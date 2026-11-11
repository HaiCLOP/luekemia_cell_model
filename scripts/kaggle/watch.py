"""Watch a Kaggle job, show it on the local progress page, and download its output when it finishes.

Usage (token in KAGGLE_API_TOKEN):
  python scripts/kaggle/watch.py haiclop/cell-triage-robust-training --expect-min 90 --out dist/kaggle_out
Kaggle does not stream logs while a job runs, so the bar shows time used against the expected run time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.progress import Progress  # noqa: E402

KAGGLE = str(Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle"))


def status(ref: str) -> str:
    r = subprocess.run([KAGGLE, "kernels", "status", ref], capture_output=True, text=True)
    text = (r.stdout + r.stderr).lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in text:
            return s
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--expect-min", type=float, default=90)
    ap.add_argument("--out", default="dist/kaggle_out")
    a = ap.parse_args()
    total = int(a.expect_min)
    with Progress("kaggle_train", "Retraining the model on Kaggle's free GPU", total=total,
                  explain=("Five copies of the network are being trained on a Kaggle T4 graphics card, with images shown "
                           "blurred, at lower resolution and in shifted colours so the model copes with other labs. Kaggle "
                           "does not report progress while it runs, so this bar shows time used against the expected "
                           "time; it jumps to the end when Kaggle says the job is done."),
                  steps=["Waiting for a Kaggle GPU", "Training on Kaggle", "Downloading the results"]) as pg:
        pg.set_stage("queued", step=0)
        t0 = time.time()
        while True:
            s = status(a.ref)
            minutes = (time.time() - t0) / 60
            if s == "running" and pg.step_index == 0:
                pg.set_stage("running on Kaggle", total=total, step=1)
            if s in ("running", "queued"):
                target = min(total - 1, int(minutes))
                if target > pg.done:
                    pg.step(target - pg.done, note=f"Kaggle status: {s}, {minutes:.0f} min so far")
                time.sleep(60)
                continue
            if s == "complete":
                pg.set_stage("downloading results", total=1, step=2)
                out = ROOT / a.out
                out.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([KAGGLE, "kernels", "output", a.ref, "-p", str(out)], capture_output=True, text=True)
                print(r.stdout[-2000:], r.stderr[-2000:])
                pg.step(note=f"saved to {out}")
                return
            raise RuntimeError(f"Kaggle job ended with status '{s}' after {minutes:.0f} min")


if __name__ == "__main__":
    main()
