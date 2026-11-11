"""Is the deployed ensemble more than a cell-size detector? Validation only; nothing is fitted here.

Usage: python -m src.size_matched

Three checks on the 5-seed ensemble (calibrated score, deployed temperature):
  1. Size alone as a classifier: how far does foreground area get on its own?
  2. Size-stratified AUC: ROC-AUC computed inside each area quintile, where size barely varies, then averaged.
  3. Size-matched pairs: each leukemic cell paired with the normal cell closest in area (without replacement,
     tolerance 2% of the frame); ROC-AUC on that matched set with subject-clustered 95% intervals.
Also: false-alarm rate among normal cells by area quintile.

If the model only read size, checks 2 and 3 would fall towards 0.5. Area comes from src.size_check
(fraction of non-black pixels), so it depends on magnification and cropping, which is exactly the worry.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.calibrate import apply_temperature
from src.evaluate import cluster_bootstrap, sigmoid
from src.size_check import cell_areas
from src.utils import load_config, resolve

TOL = 0.02


def matched_pairs(area, y, rng):
    """Greedy nearest-area matching of every leukemic cell to an unused normal cell within TOL."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    neg_sorted = neg[np.argsort(area[neg])]
    free = np.ones(len(neg_sorted), bool)
    keep = []
    for i in rng.permutation(pos):
        k = np.searchsorted(area[neg_sorted], area[i])
        best, best_d = None, TOL
        for j in range(max(0, k - 30), min(len(neg_sorted), k + 30)):
            d = abs(area[neg_sorted[j]] - area[i])
            if free[j] and d <= best_d:
                best, best_d = j, d
        if best is not None:
            free[best] = False
            keep += [i, neg_sorted[best]]
    return np.array(sorted(keep))


def main() -> None:
    cfg = load_config()
    v = cell_areas(cfg)
    runs = [np.load(resolve("reports/preds") / f"effb0_main_s{s}_val.npz", allow_pickle=False)
            for s in cfg["calibration"]["ensemble_seeds"]]
    y, g = runs[0]["labels"].astype(int), runs[0]["groups"].astype(str)
    assert (y == v.label.to_numpy()).all()
    p_ens = np.clip(np.mean([sigmoid(r["logits"]) for r in runs], 0), 1e-6, 1 - 1e-6)
    p = apply_temperature(np.log(p_ens / (1 - p_ens)), cfg["band"]["temperature"])
    area = v.area.to_numpy()
    iters = cfg["eval"]["bootstrap_iters"]
    auc = lambda pp, yy: roc_auc_score(yy, pp)  # noqa: E731

    # 1. size alone
    size_auc = cluster_bootstrap(auc, area, y, g, iters=iters)
    model_auc = cluster_bootstrap(auc, p, y, g, iters=iters)

    # 2. within-quintile AUC
    q = np.quantile(area, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(area, q)
    strata, false_alarm = [], []
    for b in range(5):
        m = bins == b
        n_pos, n_neg = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        strata.append({"quintile": b + 1, "area_range": [float(area[m].min()), float(area[m].max())],
                       "n_leukemic": n_pos, "n_normal": n_neg,
                       "roc_auc": float(roc_auc_score(y[m], p[m])) if n_pos and n_neg else None})
        false_alarm.append(float((p[m & (y == 0)] > cfg["band"]["t_high"]).mean()) if n_neg else None)
    valid = [s for s in strata if s["roc_auc"] is not None]
    w = np.array([min(s["n_leukemic"], s["n_normal"]) for s in valid], float)
    stratified_auc = float(np.average([s["roc_auc"] for s in valid], weights=w))

    # 3. matched pairs
    idx = matched_pairs(area, y, np.random.default_rng(0))
    matched_auc = cluster_bootstrap(auc, p[idx], y[idx], g[idx], iters=iters)
    matched_size_auc = float(roc_auc_score(y[idx], area[idx]))

    summary = {
        "note": "validation only; deployed ensemble, calibrated score; area = fraction of non-black pixels",
        "model_roc_auc": {"value": model_auc[0], "ci95": list(model_auc[1:])},
        "size_only_roc_auc": {"value": size_auc[0], "ci95": list(size_auc[1:])},
        "size_stratified_roc_auc": stratified_auc,
        "strata": strata,
        "false_alarm_above_band_by_area_quintile": false_alarm,
        "size_matched": {"n_pairs": int(len(idx) // 2), "tolerance": TOL, "area_auc_after_matching": matched_size_auc,
                         "roc_auc": {"value": matched_auc[0], "ci95": list(matched_auc[1:])}},
    }
    out = resolve("reports/size")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(1, 6), [s["roc_auc"] or 0 for s in strata], color="#2b3a78")
    ax[0].axhline(0.5, ls=":", color="#7d8b93"); ax[0].set_ylim(0.4, 1)
    ax[0].set_xlabel("cell-area quintile (small to large)"); ax[0].set_ylabel("ROC-AUC within quintile")
    ax[0].set_title(f"Size-stratified AUC {stratified_auc:.3f} (overall {model_auc[0]:.3f}, size alone {size_auc[0]:.3f})", fontsize=10)
    ax[1].bar(range(1, 6), [f or 0 for f in false_alarm], color="#d4573a")
    ax[1].set_xlabel("cell-area quintile (small to large)"); ax[1].set_ylabel("normal cells scored above the band")
    ax[1].set_title("False alarms concentrate in large normal cells", fontsize=10)
    fig.tight_layout(); fig.savefig(resolve("reports/figures") / "size_stratified.png", dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
