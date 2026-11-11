"""Two-stage fine-tuning with AMP, grad accumulation, early stopping on val PR-AUC.

Usage:
  python -m src.train                                   # main run from config
  python -m src.train --set train.arch=mobilenetv3_large_100 --name mnv3
  python -m src.train --set train.max_steps=20 --name smoke   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import load_split, make_loader
from src.evaluate import binary_metrics, sigmoid
from src.model import build_model, count_params, predict_logits
from src.utils import config_hash, git_sha, load_config, resolve, seed_everything


def parse_overrides(pairs: list[str]) -> dict:
    import yaml

    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def pos_weight(labels: np.ndarray) -> torch.Tensor:
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)


def write_live(state: dict, payload: dict) -> None:
    """Best-effort progress file for the monitor dashboard. Never interrupts training."""
    try:
        path = state["live_path"]
        best = state["best"] if np.isfinite(state["best"]) else None
        payload = {"name": state["name"], "updated": time.time(), "best": best, **payload}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=float))
        tmp.replace(path)
    except Exception:
        pass


def run_stage(model, loaders, cfg, device, epochs, lr, stage, state, log):
    t = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=t["weight_decay"])
    steps_per_epoch = max(1, -(-len(loaders["train"]) // t["grad_accum"]))  # ceil: the last partial group also steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and device.type == "cuda")
    crit = state["criterion"]
    patience = 0

    for epoch in range(epochs):
        model.train()
        t0, seen, running = time.time(), 0, 0.0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(device, non_blocking=True), y.float().to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=t["amp"] and device.type == "cuda"):
                logits = model(x).squeeze(1)
                if t["label_smoothing"]:
                    y = y * (1 - t["label_smoothing"]) + 0.5 * t["label_smoothing"]
                loss = crit(logits.float(), y) / t["grad_accum"]
            scaler.scale(loss).backward()
            # step on every full accumulation group, and on the last batch so a leftover group is not dropped
            if (i + 1) % t["grad_accum"] == 0 or i + 1 == len(loaders["train"]):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                state["step"] += 1
            running += loss.item() * t["grad_accum"] * len(y)
            seen += len(y)
            if i % 20 == 0:
                write_live(state, {"stage": stage, "epoch": epoch, "epochs": epochs, "batch": i + 1,
                                   "batches": len(loaders["train"]), "train_loss": running / max(seen, 1),
                                   "img_per_sec": round(seen / max(time.time() - t0, 1e-6), 1),
                                   "lr": opt.param_groups[0]["lr"]})
            if t["max_steps"] and state["step"] >= t["max_steps"]:
                break

        val_logits = predict_logits(model, loaders["val"], device, t["amp"])
        vm = binary_metrics(sigmoid(val_logits.numpy()), loaders["val"].dataset.labels, n_bins=cfg["calibration"]["ece_bins"])
        dt = time.time() - t0
        rec = {"stage": stage, "epoch": epoch, "train_loss": running / max(seen, 1), "sec": round(dt, 1),
               "img_per_sec": round(seen / dt, 1), **{f"val_{k}": v for k, v in vm.items()}}
        log.append(rec)
        print(f"[{stage} ep{epoch}] loss {rec['train_loss']:.4f}  val pr_auc {vm['pr_auc']:.4f}  "
              f"roc_auc {vm['roc_auc']:.4f}  recall {vm['recall']:.3f}  ece {vm['ece']:.4f}  {dt:.0f}s "
              f"({rec['img_per_sec']} img/s)", flush=True)

        score = vm[t["early_stop_metric"]]
        if score > state["best"]:
            state["best"], patience = score, 0
            torch.save(model.state_dict(), state["ckpt"])
        else:
            patience += 1
            if patience >= t["early_stop_patience"]:
                print(f"early stop in {stage} at epoch {epoch}")
                break
        if t["max_steps"] and state["step"] >= t["max_steps"]:
            break


def train(cfg: dict, name: str) -> dict:
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg["train"]

    train_df, val_df = load_split(cfg, "train"), load_split(cfg, "val")
    if t["augment"] == "robust" and device.type == "cuda":
        # the robust recipe runs on the GPU from images held in memory (src/gpu_augment.py)
        from src.gpu_augment import GpuAugLoader

        train_loader = GpuAugLoader(cfg, train_df, device, seed=cfg["seed"])
    else:
        train_loader = make_loader(cfg, train_df, train=True)
    loaders = {"train": train_loader, "val": make_loader(cfg, val_df, train=False, batch_size=64)}

    model = build_model(cfg).to(device)
    pw = pos_weight(train_df["label"].to_numpy()).to(device) if t["imbalance"] == "weighted_loss" else None
    ckpt_dir = resolve(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {"criterion": nn.BCEWithLogitsLoss(pos_weight=pw), "best": -np.inf, "step": 0,
             "ckpt": ckpt_dir / f"{name}.pt", "name": name}
    live_dir = resolve(cfg["paths"]["reports"]) / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    state["live_path"] = live_dir / f"{name}.json"
    log: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()

    if t["arch"] == "small_cnn" or not t["pretrained"]:
        run_stage(model, loaders, cfg, device, t["stage1_epochs"] + t["stage2_epochs"], t["stage1_lr"], "scratch", state, log)
    else:
        model.freeze_backbone()
        print("stage1 params (total, trainable):", count_params(model))
        run_stage(model, loaders, cfg, device, t["stage1_epochs"], t["stage1_lr"], "stage1", state, log)
        model.load_state_dict(torch.load(state["ckpt"], map_location=device, weights_only=True))
        model.unfreeze_last(t["unfreeze_last"])
        print("stage2 params (total, trainable):", count_params(model))
        run_stage(model, loaders, cfg, device, t["stage2_epochs"], t["stage2_lr"], "stage2", state, log)

    # Re-evaluate the best checkpoint on validation and keep its logits for calibration / band selection.
    model.load_state_dict(torch.load(state["ckpt"], map_location=device, weights_only=True))
    val_logits = predict_logits(model, loaders["val"], device, t["amp"]).numpy()
    val_metrics = binary_metrics(sigmoid(val_logits), val_df["label"].to_numpy(), n_bins=cfg["calibration"]["ece_bins"])
    preds_dir = resolve(cfg["paths"]["reports"]) / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez(preds_dir / f"{name}_val.npz", logits=val_logits, labels=val_df["label"].to_numpy(),
             patients=val_df["patient_id"].to_numpy(dtype=str), groups=val_df["group_id"].to_numpy(dtype=str))

    summary = {
        "val_metrics": val_metrics,
        "name": name, "arch": t["arch"], "seed": cfg["seed"], "git_sha": git_sha(), "config_hash": config_hash(cfg),
        "started": started, "finished": datetime.now(timezone.utc).isoformat(), "best_val": state["best"],
        "checkpoint": str(state["ckpt"]), "max_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0,
        "config": cfg, "log": log,
    }
    out = resolve(cfg["paths"]["reports"]) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"best val {t['early_stop_metric']} = {state['best']:.4f}; peak VRAM {summary['max_vram_mb']} MB")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.arch=mobilenetv3_large_100")
    args = ap.parse_args()
    cfg = load_config(overrides=parse_overrides(args.set))
    name = args.name or f"{cfg['train']['arch']}_s{cfg['seed']}"
    train(cfg, name)


if __name__ == "__main__":
    main()
