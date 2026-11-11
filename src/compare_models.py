"""Old (deployed) vs robust ensemble on outside smears, with the decision rule fixed before any result.

Rule (stated before the outside runs were read): the robust model is adopted for smear analysis only if
  1. on Taleqani, with colour normalisation and the cell-type filter, the paired difference in smear-level
     ROC-AUC (robust - old, same smears, image-level bootstrap) has a 95% interval entirely above 0, and
  2. on BCCD's healthy smears, after the filter, it calls no larger share of cells leukemic than the old model.
Otherwise the old model stays everywhere.

Usage: python -m src.compare_models
Reads reports/smear/_scores_taleqani_{old,robust}_filter.npz and bccd_classic_{filter,robust_filter}.json,
writes reports/smear/model_comparison.json.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils import resolve

OUT = resolve("reports/smear")


def paired(old: np.ndarray, new: np.ndarray, y: np.ndarray, iters: int = 2000, seed: int = 0) -> dict:
    ok = ~np.isnan(old) & ~np.isnan(new)
    o, n, yy = old[ok], new[ok], y[ok]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iters):
        i = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[i])) == 2:
            diffs.append(roc_auc_score(yy[i], n[i]) - roc_auc_score(yy[i], o[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"smears": int(ok.sum()), "auc_old": float(roc_auc_score(yy, o)), "auc_new": float(roc_auc_score(yy, n)),
            "difference": float(roc_auc_score(yy, n) - roc_auc_score(yy, o)), "difference_ci95": [float(lo), float(hi)]}


def main() -> None:
    a = np.load(OUT / "_scores_taleqani_old_filter.npz")
    b = np.load(OUT / "_scores_taleqani_robust_filter.npz")
    assert np.array_equal(a["names"], b["names"]), "the two runs must cover the same smears in the same order"
    y = a["labels"]
    res = {"rule": __doc__.split("Usage")[0].strip()}
    for arm in ("ruler_x1.0_stain_filtered", "ruler_x1.0_stain"):
        res[f"taleqani_{arm}"] = paired(a[f"mean_{arm}"], b[f"mean_{arm}"], y)
    old_b = json.loads((OUT / "bccd_classic_filter.json").read_text())["healthy_lymphoid_cells"]["stain_normalised_filtered"]
    new_b = json.loads((OUT / "bccd_classic_robust_filter.json").read_text())["healthy_lymphoid_cells"]["stain_normalised_filtered"]
    res["bccd_healthy_after_filter"] = {"old": old_b, "robust": new_b}
    main_arm = res["taleqani_ruler_x1.0_stain_filtered"]
    rule1 = main_arm["difference_ci95"][0] > 0
    rule2 = (new_b.get("called_leukemic") or 0) <= (old_b.get("called_leukemic") or 0)
    res["decision"] = {"rule1_taleqani_better": bool(rule1), "rule2_bccd_not_worse": bool(rule2),
                       "adopt_robust_for_smears": bool(rule1 and rule2)}
    (OUT / "model_comparison.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rule"}, indent=2))


if __name__ == "__main__":
    main()
