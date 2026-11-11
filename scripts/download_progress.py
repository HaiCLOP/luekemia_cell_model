"""Run a download command and show its progress on the local progress page (monitor /jobs.html).

Usage: python scripts/download_progress.py --job aml_download --title "..." --watch FILE --expect-bytes N -- cmd args...
The bar follows the size of FILE (the archive being written) against the expected size.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.progress import Progress  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--explain", default="")
    ap.add_argument("--watch", required=True)
    ap.add_argument("--expect-bytes", type=int, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    total_mb = max(1, a.expect_bytes // 2**20)
    with Progress(a.job, a.title, total=total_mb, explain=a.explain, steps=["Downloading"]) as pg:
        pg.set_stage("downloading", total=total_mb, step=0)
        proc = subprocess.Popen(cmd, cwd=ROOT)
        watch = Path(a.watch)
        while proc.poll() is None:
            time.sleep(5)
            mb = watch.stat().st_size // 2**20 if watch.exists() else 0
            mb = min(mb, total_mb - 1)
            if mb > pg.done:
                pg.step(mb - pg.done, note=f"{mb:,} of {total_mb:,} MB")
        if proc.returncode != 0:
            raise RuntimeError(f"download failed (exit {proc.returncode})")
        pg.step(max(0, total_mb - pg.done), note="download complete")


if __name__ == "__main__":
    main()
