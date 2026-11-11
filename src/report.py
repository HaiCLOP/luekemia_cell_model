"""Collect validation results of all runs into one table with subject-clustered bootstrap CIs.

Usage: python -m src.report [--split val]
Reads reports/preds/<run>_<split>.npz, writes reports/tables/<split>_comparison.{csv,md}.
CIs resample subject GROUPS (aliased patient IDs merged), 1000 iterations, 95% percentile.
"""
from __future__ import annotations

import argparse
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import binary_metrics, brier, cluster_bootstrap, ece, patient_level, sigmoid
from src.utils import load_config, resolve


def _recall(p, y, t=0.5):
    pos = y == 1
    return float(((p >= t) & pos).sum() / max(pos.sum(), 1))


def _spec(p, y, t=0.5):
    neg = y == 0
    return float(((p < t) & neg).sum() / max(neg.sum(), 1))


METRICS = {
    "pr_auc": lambda p, y: float(average_precision_score(y, p)),
    "roc_auc": lambda p, y: float(roc_auc_score(y, p)),
    "recall": _recall,
    "specificity": _spec,
    "ece": partial(ece, n_bins=15),
    "brier": brier,
}


def summarize_run(path, iters: int) -> dict:
    d = np.load(path, allow_pickle=False)
    p, y = sigmoid(d["logits"]), d["labels"].astype(int)
    groups = d["groups"] if "groups" in d.files else d["patients"]
    row = {"run": path.stem.rsplit("_", 1)[0], "n_images": len(y), "n_subjects": len(np.unique(groups))}
    for name, fn in METRICS.items():
        pt, lo, hi = cluster_bootstrap(fn, p, y, groups, iters=iters)
        row[name] = pt
        row[f"{name}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    pp, py = patient_level(p, y, groups)
    pm = binary_metrics(pp, py)
    row["subject_acc"] = pm["accuracy"]
    row["subject_recall"] = pm["recall"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    cfg = load_config()
    reports = resolve(cfg["paths"]["reports"])
    files = sorted((reports / "preds").glob(f"*_{args.split}.npz"))
    files = [f for f in files if not f.stem.startswith("smoke") and "logits" in np.load(f, allow_pickle=False).files]
    rows = [summarize_run(f, cfg["eval"]["bootstrap_iters"]) for f in files]
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    out = reports / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.split}_comparison.csv", index=False)

    show = ["run", "n_subjects"]
    lines = ["| " + " | ".join(["Run", "Subjects", "PR-AUC", "ROC-AUC", "Recall (ALL)", "Specificity", "ECE",
                                "Subject acc."]) + " |", "|" + "---|" * 8]
    for r in df.to_dict("records"):
        cell = lambda k: f"{r[k]:.3f} {r[k + '_ci']}"
        lines.append(f"| {r['run']} | {r['n_subjects']} | {cell('pr_auc')} | {cell('roc_auc')} | {cell('recall')} | "
                     f"{cell('specificity')} | {cell('ece')} | {r['subject_acc']:.3f} |")
    md = "\n".join(lines)
    (out / f"{args.split}_comparison.md").write_text(md + "\n\nThreshold 0.5, uncalibrated. 95% CIs resample subject groups (1000x).\n")
    print(md)


if __name__ == "__main__":
    main()
