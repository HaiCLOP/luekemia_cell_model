"""One-time sealed evaluation of the frozen, deployed model on the held-out test subjects (and, separately,
on an external set such as ALL-IDB2).

This script exists so the final run is a single, recorded event, not something anyone re-runs while still
tuning. It evaluates the exact ONNX artifact the API serves (models/model_meta.json), with the temperature
and band frozen in that file. Nothing here fits anything.

Usage
  python -m src.final_test --split test --confirm-final
  python -m src.final_test --external E:/datasets/allidb/ALL_IDB2/img --confirm-final

Guards
  * --confirm-final is required. Without it the script only prints what it would do.
  * The working tree must be clean, so the report points at a real commit.
  * The artifact's SHA-256 must match model_meta.json.
  * If a report for this split and artifact already exists, it refuses to run again.

Outputs  reports/final/<tag>_<artifact sha[:12]>/{summary.json, preds.npz}
  Cell-level metrics with subject-clustered 95% intervals, referral-band results, input-check rejections,
  and an exploratory subject-level table (mean cell probability; NOT a validated patient endpoint).
  For ALL-IDB2 only aggregate numbers are written (its licence forbids redistributing images).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from api.inference import Model
from api.quality import check as check_input
from src.calibrate import band_stats
from src.evaluate import binary_metrics, cluster_bootstrap, patient_level
from src.utils import ROOT, load_config, resolve


def clean_tree_sha() -> str:
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("Commit or stash your changes first: the final report must point at an exact commit.")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def internal_items(cfg, split):
    from src.data import load_split

    df = load_split(cfg, split)
    root = Path(cfg["paths"]["data_root"])
    return [(root / p, int(y), str(g)) for p, y, g in zip(df.image_path, df.label, df.group_id)]


def external_items(folder: Path):
    """ALL-IDB2 naming: ImXXX_Y.tif, Y = 1 for a lymphoblast, 0 otherwise. Each image is its own unit
    (the dataset does not publish patient IDs), so intervals here resample images, which is optimistic."""
    items = []
    for f in sorted(folder.iterdir()):
        m = re.match(r"Im\d+_([01])\.(tif|tiff|png|jpg|bmp)$", f.name, re.I)
        if m:
            items.append((f, int(m.group(1)), f.stem))
    if not items:
        raise SystemExit(f"No ALL-IDB2-style files (ImXXX_0/1.tif) found in {folder}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=["test"])
    g.add_argument("--external", type=Path)
    ap.add_argument("--confirm-final", action="store_true")
    args = ap.parse_args()
    cfg = load_config()

    model = Model(resolve("models"))  # verifies the ONNX checksum against model_meta.json
    meta = model.meta
    tag = "test" if args.split else "external_" + args.external.name.lower()
    out = resolve("reports/final") / f"{tag}_{meta['onnx_sha256'][:12]}"
    items = internal_items(cfg, args.split) if args.split else external_items(args.external)

    print(f"model {meta['name']} (sha {meta['onnx_sha256'][:12]}), T={model.temperature}, band [{model.t_low}, {model.t_high}]")
    print(f"{tag}: {len(items)} images from {len({u for *_, u in items})} units -> {out}")
    if out.exists():
        raise SystemExit("This artifact was already evaluated on this set. The result stands; do not re-run.")
    if not args.confirm_final:
        print("Dry run. Re-run with --confirm-final once every modelling choice is frozen.")
        return
    commit = clean_tree_sha()

    p_raw, p_cal, ys, units, rejected = [], [], [], [], {}
    for path, y, unit in items:
        img = Image.open(path)
        qc = check_input(img)
        if not qc.ok:  # counted, and kept out of the scored set, exactly as the API would treat them
            rejected[qc.code] = rejected.get(qc.code, 0) + 1
            continue
        pred = model.predict(img, with_heatmap=False)
        p_raw.append(pred.prob_raw); p_cal.append(pred.prob_calibrated); ys.append(y); units.append(unit)
    p_raw, p_cal, ys, units = map(np.asarray, (p_raw, p_cal, ys, units))

    iters = cfg["eval"]["bootstrap_iters"]

    def with_ci(fn, p):
        point, lo, hi = cluster_bootstrap(fn, p, ys, units, iters=iters)
        return {"value": point, "ci95": [lo, hi]}

    metrics = binary_metrics(p_cal, ys, n_bins=cfg["calibration"]["ece_bins"])
    summary = {
        "set": tag, "evaluated_at": datetime.now(timezone.utc).isoformat(), "source_commit": commit,
        "model": {k: meta.get(k) for k in ("name", "onnx_sha256", "temperature", "t_low", "t_high")},
        "n_images": len(items), "n_scored": int(len(ys)), "input_check_rejections": rejected,
        "units": int(len(np.unique(units))),
        "interval_note": "subject-clustered bootstrap" if args.split else "image-level bootstrap (no subject IDs published)",
        "cell_level": {
            "calibrated_threshold_0.5": metrics,
            "pr_auc": with_ci(lambda p, y: binary_metrics(p, y)["pr_auc"], p_cal),
            "roc_auc": with_ci(lambda p, y: binary_metrics(p, y)["roc_auc"], p_cal),
            "ece_calibrated": with_ci(lambda p, y: binary_metrics(p, y)["ece"], p_cal),
            "ece_raw": with_ci(lambda p, y: binary_metrics(p, y)["ece"], p_raw),
        },
        "referral_band": band_stats(p_cal, ys, model.t_low, model.t_high),
    }
    if args.split:
        pp, py = patient_level(p_cal, ys, units)
        summary["subject_level_exploratory"] = {
            "note": "mean cell probability per subject; not a validated patient-level endpoint",
            "n_subjects": int(len(py)), "subjects_leukemic": int(py.sum()),
            "metrics_threshold_0.5": binary_metrics(pp, py) if len(np.unique(py)) == 2 else None,
        }
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if args.split:  # internal predictions may be kept; external ones stay aggregate-only
        np.savez(out / "preds.npz", prob_raw=p_raw, prob_calibrated=p_cal, labels=ys, groups=units.astype(str))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
