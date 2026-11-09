"""CPU inference with ONNX Runtime. No PyTorch dependency.

Pipeline (must match training eval transform): RGB -> resize 224x224 bilinear (antialiased) -> [0,1] ->
ImageNet normalise -> ensemble logits -> mean of member probabilities -> temperature scaling on the
ensemble logit -> refer-to-expert band.
"""
from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# perceptually ordered dark-to-bright ramp for the heatmap (approximates 'inferno')
_LUT_STOPS = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                       [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=np.float32)
_LUT = np.stack([np.interp(np.linspace(0, len(_LUT_STOPS) - 1, 256), np.arange(len(_LUT_STOPS)), _LUT_STOPS[:, c])
                 for c in range(3)], axis=1).astype(np.uint8)


@dataclass
class Prediction:
    prob_raw: float
    prob_calibrated: float
    decision: str
    member_probs: list[float]
    heatmap_png_base64: str | None
    latency_ms: int


class Model:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_dir / self.meta["onnx_file"]), opts, providers=["CPUExecutionProvider"])
        inp = self.meta["input"]
        self.size = int(inp["size"])
        self.mean = np.array(inp["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(inp["std"], dtype=np.float32).reshape(3, 1, 1)
        self.temperature = float(self.meta["temperature"])
        self.t_low, self.t_high = float(self.meta["t_low"]), float(self.meta["t_high"])

    # ------------------------------------------------------------------ steps
    def preprocess(self, img: Image.Image) -> tuple[np.ndarray, Image.Image]:
        rgb = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x = (x - self.mean) / self.std
        return x[None].astype(np.float32), rgb

    def decide(self, p_cal: float) -> str:
        if p_cal < self.t_low:
            return "normal"
        if p_cal > self.t_high:
            return "leukemic"
        return "refer_to_expert"

    def calibrate(self, member_logits: np.ndarray) -> tuple[float, float]:
        p_members = 1.0 / (1.0 + np.exp(-member_logits))
        p_raw = float(np.clip(p_members.mean(), 1e-6, 1 - 1e-6))
        z = np.log(p_raw / (1 - p_raw))
        p_cal = float(1.0 / (1.0 + np.exp(-z / self.temperature)))
        return p_raw, p_cal

    def heatmap(self, cam: np.ndarray, rgb: Image.Image) -> str:
        c = cam - cam.min()
        c = c / (c.max() + 1e-8)
        cam_img = Image.fromarray((c * 255).astype(np.uint8)).resize(rgb.size, Image.BICUBIC)
        heat = _LUT[np.asarray(cam_img)]
        base = np.asarray(rgb, dtype=np.float32)
        blend = (0.55 * base + 0.45 * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(blend).save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ------------------------------------------------------------------ public
    def predict(self, img: Image.Image, with_heatmap: bool = True) -> Prediction:
        t0 = time.perf_counter()
        x, rgb = self.preprocess(img)
        logits, cam = self.session.run(None, {"image": x})
        p_raw, p_cal = self.calibrate(logits[0])
        hm = self.heatmap(cam[0], rgb) if with_heatmap else None
        return Prediction(
            prob_raw=p_raw, prob_calibrated=p_cal, decision=self.decide(p_cal),
            member_probs=[float(v) for v in 1.0 / (1.0 + np.exp(-logits[0]))],
            heatmap_png_base64=hm, latency_ms=int(round((time.perf_counter() - t0) * 1000)),
        )
