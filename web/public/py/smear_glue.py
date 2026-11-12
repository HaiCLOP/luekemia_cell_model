# Runs inside the visitor's browser (Pyodide). Mirrors api/smear.py up to the point where cells are cut out:
# decode -> optional downscale -> find white cells (segment.py, copied from src/vision) -> C-NMC-style crops.
# Classification happens in JavaScript with ONNX Runtime Web, like the single-cell page.
import io
import json

import cv2
import numpy as np
from PIL import Image, ImageOps
from segment import find_white_cells, to_cnmc

RBC_UM = 7.5
MAX_SIDE = 2048


def analyse(data: bytes, um_per_px: float, stain_json: str, model_size: int, filter_size: int) -> dict:
    stain = json.loads(stain_json)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(bytes(data))))
    rgb = np.asarray(img.convert("RGB"))
    k = max(rgb.shape[:2]) / MAX_SIDE
    if k > 1:
        rgb = cv2.resize(rgb, (int(rgb.shape[1] / k), int(rgb.shape[0] / k)), interpolation=cv2.INTER_AREA)
    shrink = max(1.0, k)

    override, source = None, "red_cells_in_image"
    if um_per_px and um_per_px > 0:
        override, source = RBC_UM / (um_per_px * shrink), "um_per_px"
    res = find_white_cells(rgb, rbc_override=override)

    scored = [i for i, c in enumerate(res.cells) if c.lymphoid_like and not c.touches_border]
    full = [to_cnmc(rgb, res.cells[i], res.scale, mode="ruler" if res.scale else "fit", stain=stain) for i in scored]
    crops = np.stack([np.asarray(Image.fromarray(f).convert("RGB").resize((model_size, model_size), Image.BILINEAR)) for f in full]) \
        if full else np.zeros((0, model_size, model_size, 3), np.uint8)
    fcrops = np.stack([np.asarray(Image.fromarray(f).resize((filter_size, filter_size), Image.BILINEAR)) for f in full]) \
        if full else np.zeros((0, filter_size, filter_size, 3), np.uint8)

    cells = []
    for c in res.cells:
        cnt, _ = cv2.findContours(c.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = max(cnt, key=cv2.contourArea).reshape(-1, 2).tolist() if cnt else []
        cells.append({"box": [int(v) for v in c.box], "lymphoid_like": bool(c.lymphoid_like),
                      "touches_border": bool(c.touches_border), "contour": poly})
    return {
        "image": np.ascontiguousarray(rgb, dtype=np.uint8), "height": int(rgb.shape[0]), "width": int(rgb.shape[1]),
        "shrink": float(shrink), "source": source if res.rbc_diameter_px is not None else "unknown",
        "rbc_diameter_px": None if res.rbc_diameter_px is None else float(res.rbc_diameter_px),
        "scale": None if res.scale is None else float(res.scale),
        "notes": list(res.notes), "cells_json": json.dumps(cells), "scored": scored,
        "crops": np.ascontiguousarray(crops, dtype=np.uint8), "filter_crops": np.ascontiguousarray(fcrops, dtype=np.uint8),
    }
