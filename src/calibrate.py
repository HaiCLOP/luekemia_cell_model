"""Temperature scaling, ensembles, MC dropout, and the refer-to-expert band.

All fitting uses VALIDATION patients only. The band is chosen to keep the leukemic miss rate at or
below `band.target_miss_rate` while minimising the referral rate.
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluate import sigmoid


# ---------------------------------------------------------------- temperature scaling
def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Single scalar T minimising NLL of sigmoid(logits / T) on validation data (LBFGS on log T)."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, t: float) -> np.ndarray:
    return sigmoid(np.asarray(logits, float) / t)


# ---------------------------------------------------------------- ensembles / MC dropout
def ensemble_probs(prob_list: list[np.ndarray]) -> np.ndarray:
    """Deep ensemble: average member probabilities (each member already temperature scaled or raw)."""
    return np.mean(np.stack(prob_list), axis=0)


@torch.no_grad()
def mc_dropout_probs(model, loader, device, passes: int = 20, amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of sigmoid outputs over `passes` stochastic forward passes (dropout on, BN frozen)."""
    from src.model import enable_mc_dropout

    enable_mc_dropout(model)
    runs = []
    for _ in range(passes):
        out = []
        for x, _ in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                out.append(torch.sigmoid(model(x.to(device)).float().squeeze(1)).cpu())
        runs.append(torch.cat(out).numpy())
    model.eval()
    runs = np.stack(runs)
    return runs.mean(0), runs.std(0)


# ---------------------------------------------------------------- refer-to-expert band
def apply_band(probs, t_low: float, t_high: float) -> np.ndarray:
    """0 = normal, 1 = leukemic, -1 = refer to expert."""
    probs = np.asarray(probs, float)
    out = np.full(len(probs), -1, dtype=int)
    out[probs < t_low] = 0
    out[probs > t_high] = 1
    return out


def band_stats(probs, labels, t_low: float, t_high: float) -> dict:
    """Missed leukemic = leukemic cells auto-labelled normal (referred cells are NOT misses: a human sees them)."""
    labels = np.asarray(labels, int)
    d = apply_band(probs, t_low, t_high)
    pos = labels == 1
    auto = d != -1
    return {
        "t_low": float(t_low), "t_high": float(t_high),
        "referral_rate": float((d == -1).mean()),
        "miss_rate": float(((d == 0) & pos).sum() / max(pos.sum(), 1)),          # missed leukemic / all leukemic
        "false_alarm_rate": float(((d == 1) & ~pos).sum() / max((~pos).sum(), 1)),
        "auto_accuracy": float((d[auto] == labels[auto]).mean()) if auto.any() else float("nan"),
        "coverage": float(auto.mean()),
    }


def choose_band(probs, labels, target_miss: float = 0.02, max_false_alarm: float | None = None,
                grid: int = 201) -> dict:
    """Grid search on validation data.

    t_low is the largest value whose miss rate stays <= target. t_high is then chosen as the smallest
    value >= t_low whose false-alarm rate is no worse than the single-threshold false-alarm rate at
    0.5 (or `max_false_alarm` if given). This makes the band referral-minimal subject to both errors.
    """
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    qs = np.unique(np.concatenate([np.linspace(0, 1, grid), np.quantile(probs, np.linspace(0, 1, grid))]))
    pos, neg = labels == 1, labels == 0

    miss = np.array([((probs < t) & pos).sum() / max(pos.sum(), 1) for t in qs])
    ok = qs[miss <= target_miss]
    t_low = float(ok.max()) if len(ok) else 0.0

    fa_cap = max_false_alarm if max_false_alarm is not None else ((probs > 0.5) & neg).sum() / max(neg.sum(), 1)
    cands = qs[qs >= t_low]
    fa = np.array([((probs > t) & neg).sum() / max(neg.sum(), 1) for t in cands])
    good = cands[fa <= fa_cap]
    t_high = float(good.min()) if len(good) else 1.0
    t_high = max(t_high, t_low)
    out = band_stats(probs, labels, t_low, t_high)
    out["target_miss_rate"] = target_miss
    return out


def random_referral(probs, labels, rate: float, threshold: float = 0.5, iters: int = 1000, seed: int = 0) -> dict:
    """Baseline H2: refer a random `rate` fraction, decide the rest with a fixed threshold."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    pos = labels == 1
    misses = []
    for _ in range(iters):
        refer = rng.random(len(probs)) < rate
        misses.append((((probs < threshold) & pos) & ~refer).sum() / max(pos.sum(), 1))
    return {"referral_rate": rate, "miss_rate": float(np.mean(misses)), "miss_rate_sd": float(np.std(misses))}
