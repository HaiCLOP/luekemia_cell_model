"""Train the 5-seed "robust" ensemble (augment=robust) one seed after another, with plain-language progress
on the monitor's Jobs page (python monitor/server.py, http://127.0.0.1:8765/jobs.html).

Usage: python scripts/train_robust.py [--seeds 42 43 44 45 46]
Seeds whose checkpoint and run summary already exist are skipped, so the script can be re-run after a stop.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.progress import Progress  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402

STAGE_EPOCHS = {"stage1": 5, "stage2": 25}


def seed_fraction(live: dict | None) -> float:
    """How far one seed is, from train.py's live file: stage 1 is 5 short epochs, stage 2 up to 25."""
    if not live:
        return 0.0
    st = live.get("stage", "stage1")
    ep, eps = live.get("epoch", 0), max(1, live.get("epochs", 1))
    b, bs = live.get("batch", 0), max(1, live.get("batches", 1))
    within = min(1.0, (ep + b / bs) / eps)
    return 0.15 * within if st == "stage1" else 0.15 + 0.85 * within


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    cfg = load_config()
    ck = resolve(cfg["paths"]["checkpoints"])
    names = [f"effb0_robust_s{s}" for s in args.seeds]

    steps = [f"Train copy {i + 1} of {len(names)} (seed {s})" for i, s in enumerate(args.seeds)]
    with Progress("train_robust", "Retraining the model so it copes with other labs' images", total=100,
                  explain=("Teaches five copies of the network on the same C-NMC cells, but shown blurred, at lower "
                           "resolution, in shifted colours and at different sizes, so it stops depending on one lab's "
                           "look. Each copy stops early once it stops improving, so the finish time can move earlier."),
                  steps=steps) as pg:
        for i, (seed, name) in enumerate(zip(args.seeds, names)):
            pg.set_stage(f"training {name}", total=100, step=i)
            if (ck / f"{name}.pt").exists() and (resolve("reports/runs") / f"{name}.json").exists():
                pg.step(100, note=f"{name} already trained, skipped")
                continue
            cmd = [sys.executable, "-m", "src.train", "--name", name, "--set", f"seed={seed}",
                   "train.augment=robust", "data.num_workers=4"]
            log = open(resolve("reports/logs") / f"{name}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            live_path = resolve("reports/live") / f"{name}.json"
            while proc.poll() is None:
                time.sleep(5)
                try:
                    live = json.loads(live_path.read_text())
                except Exception:
                    live = None
                done = int(100 * seed_fraction(live))
                if done > pg.done:
                    pg.step(done - pg.done, note=f"{name}: {live.get('stage')} epoch {live.get('epoch', 0) + 1}" if live else name,
                            metrics={"best_val_pr_auc": live.get("best")} if live and live.get("best") else None)
            log.close()
            if proc.returncode != 0:
                raise RuntimeError(f"{name} failed (exit {proc.returncode}); see reports/logs/{name}.log")
            pg.step(100 - pg.done, note=f"{name} finished")


if __name__ == "__main__":
    main()
