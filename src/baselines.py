"""Baseline A: frozen ImageNet EfficientNet-B0 pooled features + logistic regression.

Usage: python -m src.baselines
C is chosen on validation PR-AUC. Writes reports/runs/baseline_a_lr.json and reports/preds/baseline_a_lr_val.npz.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import load_split, make_loader
from src.evaluate import binary_metrics
from src.model import Classifier
from src.utils import config_hash, git_sha, load_config, resolve, seed_everything


@torch.no_grad()
def extract(model, loader, device) -> np.ndarray:
    model.eval()
    return np.concatenate([model.features(x.to(device)).cpu().numpy() for x, _ in loader])


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier("efficientnet_b0", pretrained=True).to(device)

    tr_df, va_df = load_split(cfg, "train"), load_split(cfg, "val")
    # no augmentation: features of the plain resized image
    xtr = extract(model, make_loader(cfg, tr_df, train=False, batch_size=64), device)
    xva = extract(model, make_loader(cfg, va_df, train=False, batch_size=64), device)
    ytr, yva = tr_df["label"].to_numpy(), va_df["label"].to_numpy()

    scaler = StandardScaler().fit(xtr)
    xtr, xva = scaler.transform(xtr), scaler.transform(xva)
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        lr = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(xtr, ytr)
        p = lr.predict_proba(xva)[:, 1]
        m = binary_metrics(p, yva, n_bins=cfg["calibration"]["ece_bins"])
        print(f"C={c}: val pr_auc {m['pr_auc']:.4f} roc_auc {m['roc_auc']:.4f} recall {m['recall']:.3f}")
        if best is None or m["pr_auc"] > best[1]["pr_auc"]:
            best = (c, m, lr.decision_function(xva))

    c, m, logits = best
    out = resolve(cfg["paths"]["reports"])
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    np.savez(out / "preds" / "baseline_a_lr_val.npz", logits=logits, labels=yva,
             patients=va_df["patient_id"].astype(str).to_numpy(), groups=va_df["group_id"].astype(str).to_numpy())
    summary = {"name": "baseline_a_lr", "arch": "efficientnet_b0 frozen + logistic regression", "C": c,
               "seed": cfg["seed"], "git_sha": git_sha(), "config_hash": config_hash(cfg),
               "finished": datetime.now(timezone.utc).isoformat(), "val_metrics": m}
    (out / "runs" / "baseline_a_lr.json").write_text(json.dumps(summary, indent=2))
    print(f"best C={c}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})}")


if __name__ == "__main__":
    main()
