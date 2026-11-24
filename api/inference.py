"""CPU inference with ONNX Runtime. No PyTorch dependency.

Model MH51401102 (formerly DinoBloom-B-3lab): frozen DinoBloom-B with two linear heads in one ONNX graph,
`logit` (leukemic blast vs normal lymphocyte, trained on three labs) and `keep_logit` (its cell filter: keep =
lymphocyte or blast). Pipeline: RGB -> resize 224x224 bilinear (antialiased) -> [0,1] -> ImageNet normalise ->
logit -> temperature scaling -> refer-to-expert band. There is no heatmap output.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass
class Prediction:
    prob_raw: float
    prob_calibrated: float
    decision: str
    latency_ms: int
    max_abs_logit: float = 0.0
    kept: bool = True            # the cell filter's verdict: False = not a lymphocyte or blast


class Model:
    def __init__(self, model_dir: str | Path, meta_file: str = "model_meta.json"):
        model_dir = Path(model_dir)
        self.dir = model_dir
        self.meta = json.loads((model_dir / meta_file).read_text())
        self._verify(model_dir)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = self._session(model_dir / self.meta["onnx_file"], opts)
        inp = self.meta["input"]
        self.size = int(inp["size"])
        self.mean = np.array(inp["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(inp["std"], dtype=np.float32).reshape(3, 1, 1)
        self.temperature = float(self.meta["temperature"])
        self.t_low, self.t_high = float(self.meta["t_low"]), float(self.meta["t_high"])
        # out-of-distribution guard: 1.5x the largest |logit| seen on the training cells (see model_meta.json)
        self.ood_logit = float(self.meta.get("ood_max_abs_logit", 25.0))
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.has_filter = "keep_logit" in self.outputs
        self.keep_logit_threshold = float(self.meta.get("filter_threshold_logit", 0.0))

    @staticmethod
    def _session(path: Path, opts):
        return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def _verify(self, model_dir: Path) -> None:
        """Refuse to serve an artifact that does not match its metadata (wrong file, partial download)."""
        m = self.meta
        if not (0 <= m["t_low"] <= m["t_high"] <= 1) or m["temperature"] <= 0:
            raise ValueError("model_meta.json has an invalid band or temperature")
        if os.getenv("SKIP_MODEL_CHECKSUM") == "1":
            return
        h = hashlib.sha256()
        with open(model_dir / m["onnx_file"], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != m["onnx_sha256"]:
            raise ValueError(f"{m['onnx_file']} checksum does not match model_meta.json")

    # ------------------------------------------------------------------ steps
    def preprocess(self, img: Image.Image) -> tuple[np.ndarray, Image.Image]:
        rgb = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x = (x - self.mean) / self.std
        return x[None].astype(np.float32), rgb

    def logits_and_keep(self, xs: np.ndarray, batch: int = 32) -> tuple[np.ndarray, np.ndarray | None]:
        """[N, 3, size, size] -> blast logits [N] and the filter's keep mask [N] (None if the graph has no filter)."""
        zs, ks = [], []
        for i in range(0, len(xs), batch):
            out = dict(zip(self.outputs, self.session.run(None, {"image": xs[i:i + batch]})))
            # "logit" [N] for MH51401102; a retired ensemble graph (first output "logits" [N, members]) also loads
            zs.append(out["logit"] if "logit" in out else out[self.outputs[0]])
            if self.has_filter:
                ks.append(out["keep_logit"])
        return np.concatenate(zs), (np.concatenate(ks) >= self.keep_logit_threshold) if self.has_filter else None

    def cell_logits(self, xs: np.ndarray) -> np.ndarray:
        """[N, 3, size, size] -> [N, members] logits (members = 1 for MH51401102)."""
        z = self.logits_and_keep(xs)[0]
        return z if z.ndim == 2 else z[:, None]

    def decide(self, p_cal: float) -> str:
        if p_cal < self.t_low:
            return "normal"
        if p_cal > self.t_high:
            return "leukemic"
        return "refer_to_expert"

    def calibrate(self, member_logits: np.ndarray) -> tuple[float, float]:
        """Logit(s) -> (raw, calibrated) probability. One logit for MH51401102; several are averaged as probabilities."""
        p_members = 1.0 / (1.0 + np.exp(-np.clip(member_logits, -50.0, 50.0)))  # clip: no overflow warning, same result
        p_raw = float(np.clip(p_members.mean(), 1e-6, 1 - 1e-6))
        z = np.log(p_raw / (1 - p_raw))
        p_cal = float(1.0 / (1.0 + np.exp(-z / self.temperature)))
        return p_raw, p_cal

    # ------------------------------------------------------------------ public
    def predict(self, img: Image.Image) -> Prediction:
        t0 = time.perf_counter()
        x, _ = self.preprocess(img)
        z, keep = self.logits_and_keep(x)
        p_raw, p_cal = self.calibrate(np.atleast_1d(z[0]))
        return Prediction(
            prob_raw=p_raw, prob_calibrated=p_cal, decision=self.decide(p_cal), max_abs_logit=float(np.abs(z[0]).max()),
            kept=True if keep is None else bool(keep[0]), latency_ms=int(round((time.perf_counter() - t0) * 1000)),
        )
