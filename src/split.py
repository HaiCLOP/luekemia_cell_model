"""Patient-level split: train / val / test roles plus CV folds inside train.

Usage:  python -m src.split
Writes data/splits.csv with columns image_path, patient_id, label, split, fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import load_config, resolve


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    per = df.groupby("unit")["label"].agg(["nunique", "first", "size"])
    mixed = per[per["nunique"] > 1]
    if len(mixed):
        # In C-NMC every subject is either ALL or healthy. A mixed subject means a parsing bug.
        raise ValueError(f"Patients with mixed labels (parsing bug?): {list(mixed.index)[:10]}")
    return per.rename(columns={"first": "label", "size": "n_images"})[["label", "n_images"]]


def _pick_patients(patients: pd.DataFrame, frac: float, rng: np.random.Generator) -> set[str]:
    """Stratified by patient label; chosen so the image share is close to `frac` too."""
    chosen: set[str] = set()
    for _, grp in patients.groupby("label"):
        ids = grp.index.to_numpy()
        target_imgs = frac * grp["n_images"].sum()
        best, best_err = None, np.inf
        for _ in range(500):  # random search for an image-balanced subset
            perm = rng.permutation(ids)
            cum = grp.loc[perm, "n_images"].cumsum().to_numpy()
            k = int(np.argmin(np.abs(cum - target_imgs))) + 1
            err = abs(cum[k - 1] - target_imgs)
            if err < best_err:
                best, best_err = set(perm[:k]), err
        chosen |= best
    return chosen


def make_splits(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    manifest = manifest.copy()
    if "group_id" not in manifest.columns:
        manifest["group_id"] = manifest["patient_id"]
    # Split unit = group of patient IDs that share identical images (aliases of one subject).
    manifest["unit"] = manifest["group_id"]
    patients = _patient_table(manifest)

    test_ids = _pick_patients(patients, cfg["split"]["test_frac"], rng)
    rest = patients.drop(index=list(test_ids))
    val_frac_of_rest = cfg["split"]["val_frac"] / (1 - cfg["split"]["test_frac"])
    val_ids = _pick_patients(rest, val_frac_of_rest, rng)

    out = manifest.copy()
    out["split"] = "train"
    out.loc[out["unit"].isin(val_ids), "split"] = "val"
    out.loc[out["unit"].isin(test_ids), "split"] = "test"

    out["fold"] = -1
    tr = out["split"] == "train"
    sgkf = StratifiedGroupKFold(n_splits=cfg["split"]["n_folds"], shuffle=True, random_state=cfg["seed"])
    for f, (_, idx) in enumerate(sgkf.split(out[tr], out.loc[tr, "label"], out.loc[tr, "unit"])):
        out.loc[out[tr].index[idx], "fold"] = f

    assert_no_leakage(out)
    return out[["image_path", "patient_id", "group_id", "label", "split", "fold"]]


def assert_no_leakage(df: pd.DataFrame) -> None:
    for col in ("patient_id", "group_id"):
        if col not in df.columns:
            continue
        per = df.groupby(col)["split"].nunique()
        leaked = per[per > 1]
        if len(leaked):
            raise AssertionError(f"{col} values in more than one split: {list(leaked.index)}")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("split").agg(
        images=("image_path", "size"),
        patients=("patient_id", "nunique"),
        pos_rate=("label", "mean"),
        pos_patients=("patient_id", lambda s: df.loc[s.index].groupby("patient_id")["label"].first().sum()),
    )


def main() -> None:
    cfg = load_config()
    manifest = pd.read_csv(resolve(cfg["paths"]["data_root"]) / "manifest.csv")
    splits = make_splits(manifest, cfg)
    out = resolve(cfg["paths"]["splits_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)
    print(summarize(splits).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
