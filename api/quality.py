"""Input contract checks that run before the classifier.

The model was trained on C-NMC crops: one segmented cell, roughly centred, on a black background, in a
square frame. Anything else (a whole smear, a photo, several cells, a blank frame) is outside what it was
validated on, so it gets an explicit "not supported" answer instead of a score. These checks are cheap
heuristics on a 64 x 64 thumbnail; they catch obvious misuse, they are not an out-of-distribution detector.

Thresholds were set from the training and validation crops (see tests/test_quality.py and the numbers
printed by `python -m api.quality <folder>`), with a wide margin so that real crops are not rejected.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

THUMB = 64
DARK = 24             # a pixel counts as background when its brightest channel is below this (0-255)
MAX_ASPECT = 1.25     # long side / short side
MIN_BORDER_DARK = 0.70
MIN_FOREGROUND, MAX_FOREGROUND = 0.02, 0.60   # real crops: 0.08 to 0.26
MIN_MAIN_BLOB = 0.80  # share of foreground pixels in the largest connected blob
MIN_CONTRAST = 1.0    # grey-level std inside the eroded cell; real crops measure 1.7 to 21.5, a flat blob 0


@dataclass
class Check:
    ok: bool
    code: str = "ok"          # ok | unsupported_input | insufficient_image_quality
    reason: str = ""
    stats: dict | None = None


def _largest_blob_share(mask: np.ndarray) -> float:
    """Share of True pixels that belong to the largest 4-connected component (small grid, plain flood fill)."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    best = 0
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack, size = [(sy, sx)], 0
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, size)
    return best / total


def measure(img: Image.Image) -> dict:
    w, h = img.size
    rgb = np.asarray(img.convert("RGB").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32)
    fg = rgb.max(axis=2) >= DARK
    border = np.ones((THUMB, THUMB), dtype=bool)
    border[3:-3, 3:-3] = False
    grey = rgb.mean(axis=2)
    # interior of the cell only: two-pixel erosion drops the soft edge, which would add contrast to any blob
    inner = fg.copy()
    for _ in range(2):
        inner[1:-1, 1:-1] &= inner[:-2, 1:-1] & inner[2:, 1:-1] & inner[1:-1, :-2] & inner[1:-1, 2:]
        inner[[0, -1], :] = False
        inner[:, [0, -1]] = False
    return {
        "aspect": max(w, h) / max(1, min(w, h)),
        "border_dark": float((~fg[border]).mean()),
        "foreground": float(fg.mean()),
        "main_blob": _largest_blob_share(fg),
        "contrast": float(grey[inner].std()) if inner.any() else 0.0,
    }


def check(img: Image.Image) -> Check:
    s = measure(img)
    if s["aspect"] > MAX_ASPECT:
        return Check(False, "unsupported_input", "The image is not close to square. Upload one cropped cell in a square frame.", s)
    if s["foreground"] < MIN_FOREGROUND:
        return Check(False, "insufficient_image_quality", "The image is almost entirely black. No cell was found.", s)
    if s["border_dark"] < MIN_BORDER_DARK:
        return Check(False, "unsupported_input",
                     "This does not look like a segmented cell on a black background. Whole smears, photos and "
                     "uncropped fields are not supported.", s)
    if s["foreground"] > MAX_FOREGROUND:
        return Check(False, "unsupported_input", "The cell fills the whole frame. The model expects a cell with black background around it.", s)
    if s["main_blob"] < MIN_MAIN_BLOB:
        return Check(False, "unsupported_input", "More than one object was found. Upload a single cell per image.", s)
    if s["contrast"] < MIN_CONTRAST:
        return Check(False, "insufficient_image_quality", "The cell has almost no internal detail (flat or badly out of focus).", s)
    return Check(True, stats=s)


if __name__ == "__main__":  # python -m api.quality <folder> [limit]: pass rate and stat ranges over real crops
    from pathlib import Path

    files = sorted(Path(sys.argv[1]).rglob("*.png"))
    if len(sys.argv) > 2:
        files = files[:: max(1, len(files) // int(sys.argv[2]))]
    rows, fails = [], {}
    for f in files:
        c = check(Image.open(f))
        rows.append(c.stats)
        if not c.ok:
            fails[c.reason[:40]] = fails.get(c.reason[:40], 0) + 1
    print(f"{len(files)} images, rejected {sum(fails.values())}: {fails}")
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        print(f"{k:12s} min {v.min():.3f}  p1 {np.percentile(v, 1):.3f}  median {np.median(v):.3f}  p99 {np.percentile(v, 99):.3f}  max {v.max():.3f}")
