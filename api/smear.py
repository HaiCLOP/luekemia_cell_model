"""Smear analysis for the API: find white cells, put each lymphoid-like cell on C-NMC scale, score it with
the frozen single-cell ensemble, and summarise. Research demo. Not for clinical use.

The smear-level decision applies the single-cell band to the MEAN calibrated score of the scored cells.
That is a research aggregate (see src/smear_eval.py for how it behaves on an outside ALL dataset), not a
diagnosis and not a probability that the person has leukemia.
"""
from __future__ import annotations

import base64
import io
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from api.schemas import Band, SmearCell, SmearResponse, SmearSummary
from src.vision.segment import CNMC_RBC_PX, find_white_cells, to_cnmc

RBC_UM = 7.5            # typical red-cell diameter used when the caller gives a pixel size
MAX_SIDE = 2048         # larger images are downscaled first (keeps latency bounded)
COLOURS = {"leukemic": (212, 87, 58), "normal": (79, 122, 90), "refer_to_expert": (168, 107, 18), "not_scored": (125, 139, 147)}


@lru_cache(maxsize=4)
def _stain_target(model_dir: str) -> dict | None:
    f = Path(model_dir) / "cnmc_stain.json"
    return json.loads(f.read_text()) if f.exists() else None


@lru_cache(maxsize=4)
def _cell_filter(model_dir: str):
    """The learned lymphoid / other-white-cell filter, if this server has it (models/cell_type_meta.json)."""
    meta_f = Path(model_dir) / "cell_type_meta.json"
    if not meta_f.exists():
        return None
    import onnxruntime as ort

    meta = json.loads(meta_f.read_text())
    sess = ort.InferenceSession(str(Path(model_dir) / meta["onnx_file"]), providers=["CPUExecutionProvider"])
    return sess, int(meta["input_size"]), float(meta["threshold"])


def _filter_keep(model_dir: str, crops: list[np.ndarray]) -> np.ndarray | None:
    f = _cell_filter(model_dir)
    if f is None or not crops:
        return None
    sess, size, thr = f
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
    x = np.stack([np.asarray(Image.fromarray(c).resize((size, size), Image.BILINEAR)) for c in crops]).astype(np.float32)
    x = (x.transpose(0, 3, 1, 2) / 255.0 - mean) / std
    z = sess.run(None, {"image": x})[0].reshape(-1)
    return 1 / (1 + np.exp(-np.clip(z, -50, 50))) >= thr


def _overlay(rgb: np.ndarray, cells, decisions) -> str:
    o = rgb.copy()
    lw = max(2, int(round(max(rgb.shape[:2]) / 400)))
    for i, (c, d) in enumerate(zip(cells, decisions), 1):
        cnt, _ = cv2.findContours(c.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cnt, -1, COLOURS[d], lw)
        x0, y0 = c.box[0], c.box[1]
        cv2.putText(o, str(i), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 + lw * 0.15, COLOURS[d], max(1, lw - 1), cv2.LINE_AA)
    buf = io.BytesIO()
    Image.fromarray(o).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def analyse_smear(model, img: Image.Image, um_per_px: float | None = None, rbc_diameter_px: float | None = None) -> SmearResponse:
    t0 = time.perf_counter()
    rgb = np.asarray(img.convert("RGB"))
    k = max(rgb.shape[:2]) / MAX_SIDE
    if k > 1:
        rgb = cv2.resize(rgb, (int(rgb.shape[1] / k), int(rgb.shape[0] / k)), interpolation=cv2.INTER_AREA)
    shrink = max(1.0, k)

    warnings: list[str] = []
    override, source = None, "red_cells_in_image"
    if rbc_diameter_px:
        override, source = rbc_diameter_px / shrink, "rbc_diameter_px"
    elif um_per_px:
        override, source = RBC_UM / (um_per_px * shrink), "um_per_px"

    res = find_white_cells(rgb, rbc_override=override)
    warnings += res.notes
    if res.rbc_diameter_px is None:
        source = "unknown"
        warnings.append("Scale could not be measured; cells were resized to a typical C-NMC size, which removes size information.")
    elif source == "red_cells_in_image":
        warnings.append("Scale was measured from the red cells in this one image (about 20% error per image on BCCD). "
                        "If you know the pixel size, pass um_per_px for a steadier result.")

    scored, decisions, out_cells = [], [], []
    # colours are normalised to the C-NMC training cells: without this the network's internal scores on
    # other labs' smears run into the thousands (see the smear section of MODEL_CARD.md)
    stain = _stain_target(str(model.dir))
    if stain is None:
        warnings.append("C-NMC colour statistics missing on this server; cells were scored in their original colours.")
    crops, idx = [], []
    for i, c in enumerate(res.cells):
        if c.lymphoid_like and not c.touches_border:
            crops.append(to_cnmc(rgb, c, res.scale, mode="ruler" if res.scale else "fit", stain=stain))
            idx.append(i)
    probs, ood, dropped = {}, set(), set()
    keep = _filter_keep(str(model.dir), crops)
    if keep is not None:
        dropped = {i for i, k in zip(idx, keep) if not k}
        crops = [c for c, k in zip(crops, keep) if k]
        idx = [i for i, k in zip(idx, keep) if k]
    if crops:
        xs = np.concatenate([model.preprocess(Image.fromarray(cr))[0] for cr in crops], axis=0)
        logits, _ = model.session.run(None, {"image": xs})
        for i, l in zip(idx, logits):
            if float(np.abs(l).max()) > model.ood_logit:
                ood.add(i)
            else:
                probs[i] = model.calibrate(l)[1]

    for i, c in enumerate(res.cells):
        if i in probs:
            p = probs[i]
            d = model.decide(p)
            scored.append(p)
            reason = None
        else:
            p, d = None, "not_scored"
            reason = ("out of distribution: the model's internal scores were far outside their normal range" if i in ood
                      else "not a lymphocyte according to the cell-type filter; the model only knows lymphocytes" if i in dropped
                      else "touches the image edge" if c.touches_border
                      else "not lymphoid-like (lobed or irregular nucleus); the model only knows lymphocytes")
        decisions.append(d)
        out_cells.append(SmearCell(id=i + 1, box=[int(round(v * shrink)) for v in c.box], lymphoid_like=c.lymphoid_like,
                                   touches_border=c.touches_border, prob_calibrated=None if p is None else round(float(p), 4),
                                   decision=d, reason=reason))

    mean_p = float(np.mean(scored)) if scored else None
    smear_decision = model.decide(mean_p) if mean_p is not None else "no_scorable_cells"
    if len(scored) < 3 and scored:
        warnings.append(f"Only {len(scored)} cell(s) could be scored; the smear summary is weak.")
    summary = SmearSummary(cells_found=len(res.cells), cells_scored=len(scored),
                           leukemic=decisions.count("leukemic"), normal=decisions.count("normal"),
                           refer_to_expert=decisions.count("refer_to_expert"), not_scored=decisions.count("not_scored"),
                           mean_prob_calibrated=None if mean_p is None else round(mean_p, 4), smear_decision=smear_decision)
    return SmearResponse(model_version=model.meta["name"],
                         rbc_diameter_px=None if res.rbc_diameter_px is None else round(res.rbc_diameter_px * shrink, 1),
                         scale_source=source, scale_to_cnmc=None if res.scale is None else round(res.scale / shrink, 4),
                         band=Band(t_low=model.t_low, t_high=model.t_high), summary=summary, cells=out_cells,
                         overlay_png_base64=_overlay(rgb, res.cells, decisions), warnings=warnings,
                         latency_ms=int(round((time.perf_counter() - t0) * 1000)))
