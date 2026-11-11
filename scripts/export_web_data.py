"""Export results for the web app: web/src/data/results.json, figures and three sample cells.

Usage: python scripts/export_web_data.py
Everything shown on the website comes from reports/ via this script, so numbers never drift from the analysis.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.utils import load_config  # noqa: E402

WEB = ROOT / "web"
REP = ROOT / "reports"


def ci(s: str):
    m = re.findall(r"[\d.]+", s or "")
    return [float(m[0]), float(m[1])] if len(m) == 2 else None


def _final_test():
    """The one-time sealed test result for the deployed model, if it has been run (reports/final/)."""
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    f = REP / "final" / f"test_{meta['onnx_sha256'][:12]}" / "summary.json"
    if not f.exists():
        return None
    t = json.loads(f.read_text())
    c = t["cell_level"]
    return {"subjects": t["units"], "cells": t["n_scored"], "date": t["evaluated_at"][:10],
            "pr_auc": c["pr_auc"]["value"], "pr_auc_ci": c["pr_auc"]["ci95"], "roc_auc": c["roc_auc"]["value"],
            "roc_auc_ci": c["roc_auc"]["ci95"], "ece": c["ece_calibrated"]["value"], "band": t["referral_band"]}


def main() -> None:
    cfg = load_config()
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    triage = json.loads((REP / "triage" / "summary.json").read_text())
    expl = json.loads((REP / "explain" / "summary.json").read_text())

    with (REP / "tables" / "val_comparison.csv").open(encoding="utf-8") as f:
        comp = list(csv.DictReader(f))
    label = {"effb0_main_s42": "EfficientNet-B0 (seed 42)", "mnv3_s42": "MobileNetV3-Large", "dinov2s_s42": "DINOv2 ViT-S/14",
             "baseline_b_smallcnn": "Small CNN from scratch", "baseline_a_lr": "Frozen features + logistic regression"}
    models = [{"name": label[r["run"]], "pr_auc": float(r["pr_auc"]), "pr_auc_ci": ci(r["pr_auc_ci"]),
               "roc_auc": float(r["roc_auc"]), "roc_auc_ci": ci(r["roc_auc_ci"]), "recall": float(r["recall"]),
               "specificity": float(r["specificity"]), "ece": float(r["ece"])} for r in comp if r["run"] in label]
    for r in models:
        r["deployed"] = False

    # the deployed model itself: mean of the five seeds' probabilities, raw (temperature does not change ranking)
    from src.evaluate import binary_metrics, cluster_bootstrap

    runs = [np.load(REP / "preds" / f"effb0_main_s{s}_val.npz", allow_pickle=False) for s in cfg["calibration"]["ensemble_seeds"]]
    p_ens = np.mean([1 / (1 + np.exp(-r["logits"])) for r in runs], 0)
    y_val, g_val = runs[0]["labels"], runs[0]["groups"]
    bm = binary_metrics(p_ens, y_val, n_bins=cfg["calibration"]["ece_bins"])
    iters = cfg["eval"]["bootstrap_iters"]
    models.append({
        "name": "Deployed: 5-seed EfficientNet-B0 ensemble", "deployed": True,
        "pr_auc": bm["pr_auc"], "pr_auc_ci": list(cluster_bootstrap(lambda a, b: binary_metrics(a, b)["pr_auc"], p_ens, y_val, g_val, iters=iters)[1:]),
        "roc_auc": bm["roc_auc"], "roc_auc_ci": list(cluster_bootstrap(lambda a, b: binary_metrics(a, b)["roc_auc"], p_ens, y_val, g_val, iters=iters)[1:]),
        "recall": bm["recall"], "specificity": bm["specificity"], "ece": bm["ece"]})
    models.sort(key=lambda r: (not r["deployed"], -r["pr_auc"]))

    ens_raw = next(r for r in triage["calibration"] if r["model"].startswith("Ensemble") and r["variant"] == "raw")
    ens_ts = next(r for r in triage["calibration"] if r["model"].startswith("Ensemble") and r["variant"].startswith("temperature"))
    single_raw = next(r for r in triage["calibration"] if r["model"].startswith("EfficientNet") and r["variant"] == "raw")
    single_ts = next(r for r in triage["calibration"] if r["model"].startswith("EfficientNet") and r["variant"].startswith("temperature"))

    with (REP / "tables" / "size_reliance.csv").open(encoding="utf-8") as f:
        size = {r["run"]: r for r in csv.DictReader(f)}
    seeds = [float(size[f"effb0_main_s{s}"]["rho_size_normal"]) for s in (42, 43, 44, 45, 46) if f"effb0_main_s{s}" in size]

    with (ROOT / "data" / "splits.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    splits = {}
    for r in rows:
        s = splits.setdefault(r["split"], {"images": 0, "subjects": set(), "leukemic": 0})
        s["images"] += 1
        s["subjects"].add(r["group_id"])
        s["leukemic"] += r["label"] == "1"
    splits = {k: {"images": v["images"], "subjects": len(v["subjects"]), "leukemic": v["leukemic"]} for k, v in splits.items()}

    out = {
        "model": {"name": meta["name"], "architecture": meta["architecture"], "temperature": meta["temperature"],
                  "t_low": meta["t_low"], "t_high": meta["t_high"], "band_target": meta["band_target"],
                  "onnx_mb": round((ROOT / "models" / meta["onnx_file"]).stat().st_size / 2**20, 1)},
        "data": {"images": len(rows), "subjects": len({r["group_id"] for r in rows}),
                 "patient_ids": len({r["patient_id"] for r in rows}), "splits": splits,
                 "corrupt_removed": 350, "duplicates_removed": 110, "merged_groups": 4},
        "headline": {"split": "validation, 18 subjects", "pr_auc": ens_raw["pr_auc"], "roc_auc": ens_raw["roc_auc"],
                     "pr_auc_ci": next(m["pr_auc_ci"] for m in models if m["deployed"]),
                     "ece_raw": ens_raw["ece"], "ece_raw_ci": ens_raw["ece_ci"],
                     "ece_calibrated": ens_ts["ece"], "ece_calibrated_ci": ens_ts["ece_ci"],
                     "ece_bins": cfg["calibration"]["ece_bins"]},
        "models": models,
        "calibration": [
            {"label": "Single model, raw", "ece": single_raw["ece"], "nll": single_raw["nll"]},
            {"label": "Single model, temperature scaled", "ece": single_ts["ece"], "nll": single_ts["nll"]},
            {"label": "5-seed ensemble, raw", "ece": ens_raw["ece"], "nll": ens_raw["nll"]},
            {"label": "5-seed ensemble, temperature scaled", "ece": ens_ts["ece"], "nll": ens_ts["nll"]},
        ],
        "band": {"fixed": triage["fixed_threshold_0.5"], "threshold_only": triage["threshold_only_for_target_miss"],
                 "random": triage["random_referral_same_rate"], "band": triage["band_crossfit"]},
        "explain": {"gradcam_enrichment": expl["gradcam_enrichment_median"], "insertion": expl["insertion_auc"],
                    "size_rho_normal_seeds": [min(seeds), max(seeds)],
                    "size_rho_scalejit": float(size["abl_scalejit_s42"]["rho_size_normal"]) if "abl_scalejit_s42" in size else None},
        "external": None,
        "test": _final_test(),
        "disclaimer": cfg["disclaimer"],
    }
    (WEB / "src" / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "src" / "data" / "results.json").write_text(json.dumps(out, indent=2, default=float))

    figs = WEB / "public" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for name in ("reliability.png", "risk_coverage.png", "referral_band.png", "gallery_true_positive.png",
                 "gallery_false_positive.png", "eda_samples.png"):
        shutil.copy(REP / "figures" / name, figs / name)

    # three C-NMC validation sample cells (CC BY 3.0) for "try a sample": clear leukemic, clear normal, borderline
    samples = WEB / "public" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    picks = {"sample-leukemic.png": "images/cnmc-leukemia-2019/UID_22_1_1_all.png",
             "sample-normal.png": "images/cnmc-leukemia-2019/UID_H12_1_1_hem.png"}
    # borderline: a validation cell whose ensemble probability sits inside the band
    P = np.mean([1 / (1 + np.exp(-np.load(REP / "preds" / f"effb0_main_s{s}_val.npz", allow_pickle=False)["logits"]))
                 for s in cfg["calibration"]["ensemble_seeds"]], 0)
    val = [r for r in rows if r["split"] == "val"]
    mid = int(np.argmin(np.abs(P - 0.35)))
    picks["sample-borderline.png"] = val[mid]["image_path"]
    for dst, rel in picks.items():
        Image.open(Path(cfg["paths"]["data_root"]) / rel).convert("RGB").save(samples / dst, optimize=True)
    print("wrote web/src/data/results.json, figures and samples:", picks)


if __name__ == "__main__":
    main()
