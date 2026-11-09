"""Metrics, calibration error, patient-clustered bootstrap CIs, risk-coverage."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error for a binary classifier, over the positive-class probability.

    Uses equal-width bins on p(positive) and compares mean predicted probability with the observed
    positive rate in each bin (the reliability-diagram definition).
    """
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def nll(probs, labels, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(probs, float), eps, 1 - eps)
    y = np.asarray(labels, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(probs, labels) -> float:
    return float(np.mean((np.asarray(probs, float) - np.asarray(labels, float)) ** 2))


def binary_metrics(probs, labels, threshold: float = 0.5, n_bins: int = 15) -> dict:
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    both = len(np.unique(labels)) == 2
    return {
        "n": int(len(labels)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(labels)),
        "balanced_accuracy": float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
        "roc_auc": float(roc_auc_score(labels, probs)) if both else float("nan"),
        "pr_auc": float(average_precision_score(labels, probs)) if both else float("nan"),
        "ece": ece(probs, labels, n_bins),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def patient_level(probs, labels, patients) -> tuple[np.ndarray, np.ndarray]:
    """Mean image probability per patient."""
    import pandas as pd

    df = pd.DataFrame({"p": probs, "y": labels, "pid": patients}).groupby("pid").agg(p=("p", "mean"), y=("y", "first"))
    return df["p"].to_numpy(), df["y"].to_numpy()


def cluster_bootstrap(metric_fn, probs, labels, patients, iters: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling patients (clusters) with replacement."""
    probs, labels, patients = np.asarray(probs), np.asarray(labels), np.asarray(patients)
    uniq = np.unique(patients)
    groups = {p: np.flatnonzero(patients == p) for p in uniq}
    rng = np.random.default_rng(seed)
    point = metric_fn(probs, labels)
    stats = []
    for _ in range(iters):
        idx = np.concatenate([groups[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(labels[idx])) < 2:
            continue
        stats.append(metric_fn(probs[idx], labels[idx]))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def risk_coverage(probs, labels, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, float]:
    """Selective prediction curve. Confidence = max(p, 1-p). Returns coverage, risk, AURC."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    conf = np.maximum(probs, 1 - probs)
    order = np.argsort(-conf)
    err = ((probs[order] >= threshold).astype(int) != labels[order]).astype(float)
    n = len(err)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return coverage, risk, float(trapz(risk, coverage))
