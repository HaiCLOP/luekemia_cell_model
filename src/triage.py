"""Calibration, deep ensemble, MC dropout and the refer-to-expert band, on validation only.

The test split is never read here.

Usage:  python -m src.triage [--mc]      (--mc runs MC-dropout inference on the GPU, ~5 min)

Honesty rule: every reported validation number for a *fitted* quantity (temperature, band) is
cross-fitted over subject groups: fit on 2/3 of validation subjects, score on the held-out 1/3,
rotate. The final temperature / band (fitted on all validation subjects) is what gets deployed and
what the test set will use.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.calibrate import apply_band, apply_temperature, band_stats, choose_band, fit_temperature, random_referral
from src.evaluate import brier, cluster_bootstrap, ece, nll, risk_coverage, sigmoid
from src.utils import ROOT, load_config, resolve

EO, HX, INK, MUTED = "#d0654f", "#3f63c4", "#22272b", "#7d8b93"
SEEDS = [42, 43, 44, 45, 46]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def load(name):
    d = np.load(resolve("reports/preds") / f"{name}_val.npz", allow_pickle=False)
    return d["logits"].astype(float), d["labels"].astype(int), d["groups"].astype(str)


def folds(y, g, k=3, seed=0):
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(y, y, g))


def cal_metrics(p, y, bins=15):
    return {"ece": ece(p, y, bins), "nll": nll(p, y), "brier": brier(p, y),
            "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def crossfit_temperature(z, y, g):
    """Out-of-fold calibrated probabilities + per-fold temperatures."""
    out, temps = np.zeros_like(z), []
    for tr, te in folds(y, g):
        t = fit_temperature(z[tr], y[tr])
        temps.append(t)
        out[te] = apply_temperature(z[te], t)
    return out, temps


def crossfit_band(z, y, g, target):
    """Nested over the same subject folds: on each outer-training 2/3, fit the temperature AND choose the
    band; apply both to the held-out 1/3. No held-out subject's label touches either fitted quantity."""
    dec = np.zeros(len(y), dtype=int)
    for tr, te in folds(y, g):
        t = fit_temperature(z[tr], y[tr])
        b = choose_band(apply_temperature(z[tr], t), y[tr], target_miss=target)
        dec[te] = apply_band(apply_temperature(z[te], t), b["t_low"], b["t_high"])
    pos = y == 1
    return {"referral_rate": float((dec == -1).mean()),
            "miss_rate": float(((dec == 0) & pos).sum() / pos.sum()),
            "false_alarm_rate": float(((dec == 1) & ~pos).sum() / (~pos).sum()),
            "auto_accuracy": float((dec[dec != -1] == y[dec != -1]).mean())}, dec


def reliability(ax, p, y, label, color, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    ax.plot(xs, ys, "o-", color=color, label=label, ms=4, lw=1.6)


def mc_dropout(cfg) -> dict:
    """MC dropout on the seed-42 checkpoint: 20 passes, dropout before the head only."""
    import torch

    from src.calibrate import mc_dropout_probs
    from src.data import load_split, make_loader
    from src.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(resolve(cfg["paths"]["checkpoints"]) / "effb0_main_s42.pt",
                                     map_location=device, weights_only=True))
    loader = make_loader(cfg, load_split(cfg, "val"), train=False, batch_size=64)
    mean, std = mc_dropout_probs(model, loader, device, passes=cfg["calibration"]["mc_passes"], amp=False)
    np.savez(resolve("reports/preds") / "effb0_mcdropout_s42_val.npz", mean=mean, std=std)
    return {"mean": mean, "std": std}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", action="store_true")
    ap.add_argument("--prefix", default="effb0_main",
                    help="run-name prefix of the five seeds; anything other than effb0_main writes to its own folders "
                         "and leaves the deployed temperature and band in config.yaml untouched")
    args = ap.parse_args()
    cfg = load_config()
    bins, target = cfg["calibration"]["ece_bins"], cfg["band"]["target_miss_rate"]
    main_model = args.prefix == "effb0_main"
    suffix = "" if main_model else f"_{args.prefix}"
    figs = resolve("reports/figures") if main_model else resolve("reports/figures") / args.prefix
    tabs = resolve("reports/tables")
    figs.mkdir(parents=True, exist_ok=True)
    if not main_model:
        args.mc = False

    # ---------------- members and ensemble
    Z, y, g = [], None, None
    for s in SEEDS:
        z, y_, g_ = load(f"{args.prefix}_s{s}")
        if y is not None:
            assert np.array_equal(y, y_) and np.array_equal(g, g_), "val order differs between runs"
        Z.append(z); y, g = y_, g_
    Z = np.stack(Z)
    z_single = Z[0]
    p_ens_raw = sigmoid(Z).mean(0)
    z_ens = logit(p_ens_raw)
    z_mnv3, _, _ = load("mnv3_s42")

    candidates = {
        "EfficientNet-B0 (seed 42)": z_single,
        "MobileNetV3-L": z_mnv3,
        "Ensemble of 5 B0 seeds": z_ens,
    }

    rows, cal_probs = [], {}
    for name, z in candidates.items():
        p_raw = sigmoid(z)
        p_ts, temps = crossfit_temperature(z, y, g)
        cal_probs[name] = p_ts
        for variant, p in (("raw", p_raw), ("temperature scaled (cross-fit)", p_ts)):
            m = cal_metrics(p, y, bins)
            _, lo, hi = cluster_bootstrap(lambda pp, yy: ece(pp, yy, bins), p, y, g, iters=cfg["eval"]["bootstrap_iters"])
            rows.append({"model": name, "variant": variant, **m, "ece_ci": [lo, hi],
                         "T_folds": [round(t, 3) for t in temps] if "temperature" in variant else None})

    mc = None
    if args.mc:
        mc = mc_dropout(cfg)
    elif main_model and (resolve("reports/preds") / "effb0_mcdropout_s42_val.npz").exists():
        d = np.load(resolve("reports/preds") / "effb0_mcdropout_s42_val.npz")
        mc = {"mean": d["mean"], "std": d["std"]}
    if mc is not None:
        m = cal_metrics(mc["mean"], y, bins)
        _, lo, hi = cluster_bootstrap(lambda pp, yy: ece(pp, yy, bins), mc["mean"], y, g, iters=cfg["eval"]["bootstrap_iters"])
        rows.append({"model": "EfficientNet-B0 (seed 42)", "variant": f"MC dropout ({cfg['calibration']['mc_passes']} passes)",
                     **m, "ece_ci": [lo, hi], "mean_std": float(mc["std"].mean())})

    # ---------------- deployed choice: ensemble + temperature fitted on ALL validation subjects
    T_final = fit_temperature(z_ens, y)
    p_final_fullfit = apply_temperature(z_ens, T_final)
    band_final = choose_band(p_final_fullfit, y, target_miss=target)

    # ---------------- band evaluation (cross-fitted) vs baselines, on the cross-fit calibrated ensemble
    p_cf = cal_probs["Ensemble of 5 B0 seeds"]
    band_cf, dec = crossfit_band(z_ens, y, g, target)
    pos = y == 1
    fixed = {"referral_rate": 0.0, "miss_rate": float(((p_cf < 0.5) & pos).sum() / pos.sum()),
             "false_alarm_rate": float(((p_cf >= 0.5) & ~pos).sum() / (~pos).sum())}
    rnd = random_referral(p_cf, y, rate=band_cf["referral_rate"], threshold=0.5)
    # threshold-only alternative: lower a single threshold until miss <= target (no referral, more false alarms)
    qs = np.linspace(0, 1, 1001)
    ok = [t for t in qs if ((p_cf < t) & pos).sum() / pos.sum() <= target]
    t_only = max(ok) if ok else 0.0
    thr_only = {"threshold": float(t_only), "referral_rate": 0.0,
                "miss_rate": float(((p_cf < t_only) & pos).sum() / pos.sum()),
                "false_alarm_rate": float(((p_cf >= t_only) & ~pos).sum() / (~pos).sum())}
    # misses avoided per referred cell (H2)
    n_ref = int((dec == -1).sum())
    avoided_band = (fixed["miss_rate"] - band_cf["miss_rate"]) * pos.sum()
    avoided_rand = (fixed["miss_rate"] - rnd["miss_rate"]) * pos.sum()
    cov, risk, aurc = risk_coverage(p_cf, y)
    _, risk_raw, aurc_raw = risk_coverage(sigmoid(z_single), y)

    summary = {
        "note": "Validation only. Fitted quantities are cross-fitted over subject groups (3 folds).",
        "calibration": rows,
        "deployed": {"model": "ensemble of 5 EfficientNet-B0 seeds", "temperature": T_final, "band": band_final},
        "band_crossfit": {**band_cf, "target_miss_rate": target, "n_referred": n_ref,
                          "misses_avoided_per_100_referred": float(100 * avoided_band / max(n_ref, 1))},
        "fixed_threshold_0.5": fixed,
        "random_referral_same_rate": {**rnd, "misses_avoided_per_100_referred": float(100 * avoided_rand / max(n_ref, 1))},
        "threshold_only_for_target_miss": thr_only,
        "aurc": {"ensemble_calibrated": aurc, "single_raw": aurc_raw},
    }
    out = resolve("reports/triage" + suffix)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---------------- tables
    lines = ["| Model | Variant | ECE (95% subject CI) | NLL | Brier | PR-AUC |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['variant']} | {r['ece']:.4f} [{r['ece_ci'][0]:.3f}, {r['ece_ci'][1]:.3f}] | "
                     f"{r['nll']:.4f} | {r['brier']:.4f} | {r['pr_auc']:.4f} |")
    (tabs / f"calibration{suffix}.md").write_text("\n".join(lines) + "\n")
    b = summary
    band_md = "\n".join([
        "| Policy (validation, cross-fitted) | Referred | Missed leukemic | False alarms on normal |",
        "|---|---|---|---|",
        f"| Single threshold 0.5 | 0% | {100*b['fixed_threshold_0.5']['miss_rate']:.2f}% | {100*b['fixed_threshold_0.5']['false_alarm_rate']:.1f}% |",
        f"| Single threshold tuned to <= {100*target:.0f}% miss (t={thr_only['threshold']:.3f}) | 0% | {100*thr_only['miss_rate']:.2f}% | {100*thr_only['false_alarm_rate']:.1f}% |",
        f"| Random referral at band rate | {100*rnd['referral_rate']:.1f}% | {100*rnd['miss_rate']:.2f}% | n/a |",
        f"| **Refer-to-expert band** | {100*band_cf['referral_rate']:.1f}% | {100*band_cf['miss_rate']:.2f}% | {100*band_cf['false_alarm_rate']:.1f}% |",
    ])
    (tabs / f"referral_band{suffix}.md").write_text(band_md + "\n")

    # ---------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, name in zip(axes, candidates):
        ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
        reliability(ax, sigmoid(candidates[name]), y, "raw", EO)
        reliability(ax, cal_probs[name], y, "temperature scaled", HX)
        ax.set_title(name, fontsize=11); ax.set_xlabel("predicted P(leukemic)")
    if mc is not None:
        reliability(axes[0], mc["mean"], y, "MC dropout", "#5a9e6f")
    axes[0].set_ylabel("observed leukemic fraction"); axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Reliability on validation (temperature cross-fitted over subjects)", fontsize=12)
    fig.tight_layout(); fig.savefig(figs / "reliability.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(cov, risk, color=HX, lw=2, label=f"ensemble, calibrated (AURC {aurc:.4f})")
    ax.plot(cov, risk_raw, color=EO, lw=1.6, ls="--", label=f"single model, raw (AURC {aurc_raw:.4f})")
    ax.axvline(1 - band_cf["referral_rate"], color=MUTED, ls=":", lw=1)
    ax.text(1 - band_cf["referral_rate"], ax.get_ylim()[1] * 0.92, "  band coverage", color=MUTED, fontsize=9)
    ax.set_xlabel("coverage (fraction decided automatically)"); ax.set_ylabel("error rate on decided cells")
    ax.set_title("Risk-coverage, validation"); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "risk_coverage.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bins_h = np.linspace(0, 1, 41)
    ax.hist(p_final_fullfit[y == 0], bins=bins_h, color=HX, alpha=.75, label="normal (hem)")
    ax.hist(p_final_fullfit[y == 1], bins=bins_h, color=EO, alpha=.6, label="leukemic (ALL)")
    ax.axvspan(band_final["t_low"], band_final["t_high"], color="#e9b24f", alpha=.18, label="refer to expert")
    ax.set_yscale("log"); ax.set_xlabel("calibrated P(leukemic), 5-seed ensemble"); ax.set_ylabel("cells (log)")
    ax.set_title(f"Referral band [{band_final['t_low']:.3f}, {band_final['t_high']:.3f}] chosen for <= {100*target:.0f}% missed leukemic")
    ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig(figs / "referral_band.png", dpi=160); plt.close(fig)

    # ---------------- write the deployed temperature and band back into the config
    if not main_model:
        print(json.dumps({"deployed": summary["deployed"], "band_crossfit": summary["band_crossfit"]}, indent=2, default=float))
        return
    cfg_path = ROOT / "configs" / "config.yaml"
    txt = cfg_path.read_text(encoding="utf-8")
    txt = re.sub(r"t_low: .*", f"t_low: {band_final['t_low']:.4f}            # chosen on all validation subjects (src.triage)", txt, count=1)
    txt = re.sub(r"t_high: .*", f"t_high: {band_final['t_high']:.4f}", txt, count=1)
    if "temperature:" not in txt:
        txt = txt.replace("band:\n", f"band:\n  temperature: {T_final:.4f}       # ensemble logit temperature (src.triage)\n", 1)
    else:
        txt = re.sub(r"temperature: .*", f"temperature: {T_final:.4f}       # ensemble logit temperature (src.triage)", txt, count=1)
    cfg_path.write_text(txt, encoding="utf-8")

    print("\n".join(lines)); print(); print(band_md); print()
    print(json.dumps({"deployed": summary["deployed"], "band_crossfit": summary["band_crossfit"],
                      "random": summary["random_referral_same_rate"], "aurc": summary["aurc"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
