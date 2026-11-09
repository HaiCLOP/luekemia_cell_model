"""How much do the models lean on cell size? Checked on validation cells.

Usage: python -m src.size_check effb0_main_s42 abl_scalejit_s42 ...
Reports, per run: PR-AUC, ECE, Spearman(cell area, P(leukemic)) within each class, false-alarm rate,
and the mean P(leukemic) of the large-celled normal subjects that drive the errors.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from src.evaluate import ece, sigmoid
from src.utils import load_config, resolve

HARD = ["G_H34", "G_H9", "G_H46"]


def cell_areas(cfg) -> pd.DataFrame:
    cache = resolve("reports/tables") / "val_cell_area.csv"
    if cache.exists():
        return pd.read_csv(cache)
    s = pd.read_csv(resolve(cfg["paths"]["splits_csv"]))
    v = s[s.split == "val"].reset_index(drop=True)
    root = resolve(cfg["paths"]["data_root"])
    v["area"] = [(np.asarray(Image.open(root / r).convert("L")) > 15).mean() for r in v.image_path]
    v[["image_path", "group_id", "label", "area"]].to_csv(cache, index=False)
    return v


def main(runs: list[str]) -> None:
    cfg = load_config()
    v = cell_areas(cfg)
    rows = []
    for run in runs:
        d = np.load(resolve("reports/preds") / f"{run}_val.npz", allow_pickle=True)
        p, y = sigmoid(d["logits"]), d["labels"].astype(int)
        assert len(p) == len(v) and (y == v.label.to_numpy()).all()
        n, a = v.label == 0, v.label == 1
        row = {"run": run, "pr_auc": average_precision_score(y, p), "ece": ece(p, y, 15),
               "rho_size_normal": spearmanr(v.area[n], p[n])[0], "rho_size_leukemic": spearmanr(v.area[a], p[a])[0],
               "false_alarm@0.5": float((p[n] >= 0.5).mean()), "recall@0.5": float((p[a] >= 0.5).mean())}
        for g in HARD:
            row[f"p_{g[2:]}"] = float(p[(v.group_id == g).to_numpy()].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    out = resolve("reports/tables") / "size_reliance.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["effb0_main_s42"])
