"""Download C-NMC 2019 (two HF mirrors), export images to disk, write manifest.csv.

Usage: python -m src.download
Every patient ID and label is parsed from the ORIGINAL C-NMC filename and cross-checked against the
mirror's own columns. Any mismatch, overlap between sources, or duplicate image aborts the run.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import parse_cnmc_name
from src.utils import load_config, resolve


def _orig_name(row: dict) -> str:
    # train mirror: original_image_name ; test mirror: source_patient_id holds the original filename
    return row.get("original_image_name") or row["source_patient_id"]


def _mirror_label(row: dict) -> int:
    vals = {str(row.get("label", "")).lower(), str(row.get("class_label", "")).lower()}
    if vals & {"all", "cancer"} and not vals & {"hem", "healthy", "normal"}:
        return 1
    if vals & {"hem", "healthy", "normal"} and not vals & {"all", "cancer"}:
        return 0
    raise ValueError(f"Unrecognised label columns: {vals}")


def _to_pil(obj):
    """datasets may return a decoded PIL image or a {'bytes': ..., 'path': ...} dict."""
    import io

    from PIL import Image

    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, dict):
        if obj.get("bytes"):
            return Image.open(io.BytesIO(obj["bytes"]))
        if obj.get("path"):
            return Image.open(obj["path"])
    raise TypeError(f"Cannot decode image object of type {type(obj)}")


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def export(cfg: dict) -> pd.DataFrame:
    """Export images and build the manifest.

    Data-quality rules (all logged to reports/tables/):
      * unreadable / all-zero files are dropped (corrupt_dropped.csv)
      * pixel-identical copies are dropped, keeping the first (duplicates_dropped.csv)
      * patient IDs that share any identical image are merged into one `group_id`
        (patient_groups.csv). Splits are made on group_id, so aliased IDs never straddle splits.
      * identical pixels with conflicting labels abort the run
    """
    from datasets import load_dataset

    root = resolve(cfg["paths"]["data_root"])
    rows, seen_hash, dups, corrupt = [], {}, [], []
    uf = _UnionFind()
    for spec in cfg["hf_datasets"]:
        source = spec["repo"].split("/")[-1]
        ds = load_dataset(spec["repo"], split=spec["split"])
        out_dir = root / "images" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in tqdm(ds, desc=source):
            name = _orig_name(r)
            pid, label = parse_cnmc_name(name)
            if str(r["subject_id"]).upper() != pid:
                raise ValueError(f"{source}: subject_id {r['subject_id']} != filename patient {pid} ({name})")
            if _mirror_label(r) != label:
                raise ValueError(f"{source}: label column disagrees with filename for {name}")
            uf.find(pid)
            try:
                img = _to_pil(r["image"]).convert("RGB")
                if img.getbbox() is None:
                    raise ValueError("all-black image")
            except Exception as e:
                corrupt.append({"name": name, "patient_id": pid, "source": source, "error": str(e)[:80]})
                continue
            h = hashlib.sha256(img.tobytes()).hexdigest()
            if h in seen_hash:
                first_name, first_pid, first_label = seen_hash[h]
                if first_label != label:
                    raise ValueError(f"Identical image with conflicting labels: {name} vs {first_name}")
                uf.union(pid, first_pid)
                dups.append({"dropped": name, "kept": first_name, "patient_id": pid, "kept_patient_id": first_pid,
                             "source": source})
                continue
            seen_hash[h] = (name, pid, label)
            rel = Path("images") / source / (Path(name).stem + ".png")
            dst = root / rel
            if not dst.exists():
                img.save(dst, optimize=False)
            rows.append({"image_path": rel.as_posix(), "patient_id": pid, "label": label, "source": source,
                         "orig_name": name, "width": img.width, "height": img.height, "sha256": h,
                         "cnmc_fold": r.get("fold", -1)})

    df = pd.DataFrame(rows)
    df["group_id"] = df["patient_id"].map(lambda p: "G_" + uf.find(p))
    by_src = df.groupby("patient_id")["source"].nunique()
    if (by_src > 1).any():
        raise ValueError(f"Patients present in both mirrors: {list(by_src[by_src > 1].index)}")
    if (df.groupby("group_id")["label"].nunique() > 1).any():
        raise ValueError("A merged patient group has mixed labels")
    df.to_csv(root / "manifest.csv", index=False)

    audit = resolve(cfg["paths"]["reports"]) / "tables"
    audit.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dups).to_csv(audit / "duplicates_dropped.csv", index=False)
    pd.DataFrame(corrupt).to_csv(audit / "corrupt_dropped.csv", index=False)
    groups = df.groupby("group_id").agg(patients=("patient_id", lambda s: " ".join(sorted(set(s)))),
                                        n_patients=("patient_id", "nunique"), images=("image_path", "size"),
                                        label=("label", "first"))
    groups.to_csv(audit / "patient_groups.csv")
    merged = groups[groups["n_patients"] > 1]
    print(f"dropped {len(corrupt)} corrupt and {len(dups)} duplicate images; "
          f"{len(merged)} groups merge aliased patient IDs:")
    print(merged.to_string())
    return df


def main() -> None:
    cfg = load_config()
    df = export(cfg)
    print(df.groupby(["source", "label"]).agg(images=("image_path", "size"), patients=("patient_id", "nunique")))
    print("total images", len(df), "patients", df["patient_id"].nunique())
    print("sizes", df[["width", "height"]].value_counts().head())


if __name__ == "__main__":
    main()
