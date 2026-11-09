"""Export the deployed model (5-seed EfficientNet-B0 ensemble) to ONNX for CPU inference.

Usage: python -m src.export [--name effb0-ens5-v1]

Graph inputs:  image  float32 [B, 3, 224, 224]  (ImageNet-normalised, same as training eval transform)
Graph outputs: logits float32 [B, 5]             (one raw logit per ensemble member)
               cam    float32 [B, 7, 7]          (mean over members of ReLU(sum_k w_k * A_k), A = last feature map)

For these models the head is linear on global-average-pooled features, so this CAM equals Grad-CAM at the
final feature map for the leukemic logit. Calibration (temperature) and the referral band are applied in
the API from model_meta.json, so they can change without re-exporting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from src.model import build_model
from src.utils import config_hash, git_sha, load_config, resolve


class EnsembleWithCAM(nn.Module):
    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        logits, cams = [], []
        for m in self.members:
            f = m.backbone.forward_features(x)                    # [B, C, 7, 7]
            pooled = f.mean(dim=(2, 3))
            logits.append(m.head(pooled))                         # [B, 1]
            w = m.head[-1].weight[0]                              # [C]
            cams.append(torch.relu(torch.einsum("c,bchw->bhw", w, f)))
        return torch.cat(logits, dim=1), torch.stack(cams, 0).mean(0)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="effb0-ens5-v1")
    args = ap.parse_args()
    cfg = load_config()
    seeds = cfg["calibration"]["ensemble_seeds"]
    members = []
    for s in seeds:
        m = build_model(cfg).eval()
        m.load_state_dict(torch.load(resolve(cfg["paths"]["checkpoints"]) / f"effb0_main_s{s}.pt", map_location="cpu", weights_only=True))
        members.append(m)
    model = EnsembleWithCAM(members).eval()

    out_dir = resolve("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{args.name}.onnx"
    size = cfg["data"]["img_size"]
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, onnx_path, input_names=["image"], output_names=["logits", "cam"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)

    # validation metrics of the deployed model for the model card / DB row
    triage = json.loads((resolve("reports/triage") / "summary.json").read_text())
    ens_rows = [r for r in triage["calibration"] if r["model"].startswith("Ensemble")]
    meta = {
        "name": args.name,
        "architecture": f"EfficientNet-B0 (timm efficientnet_b0.ra_in1k), deep ensemble of {len(seeds)} seeds",
        "seeds": seeds,
        "input": {"size": size, "mean": cfg["data"]["mean"], "std": cfg["data"]["std"], "resize": "bilinear, antialiased"},
        "temperature": cfg["band"]["temperature"],
        "t_low": cfg["band"]["t_low"], "t_high": cfg["band"]["t_high"],
        "band_target": f"leukemic miss rate <= {cfg['band']['target_miss_rate']:.0%} on validation subjects",
        "metrics": {
            "split": "validation (18 subjects); test set not yet evaluated",
            "pr_auc": next(r["pr_auc"] for r in ens_rows if r["variant"] == "raw"),
            "roc_auc": next(r["roc_auc"] for r in ens_rows if r["variant"] == "raw"),
            "ece_calibrated": next(r["ece"] for r in ens_rows if r["variant"].startswith("temperature")),
            "band_crossfit": triage["band_crossfit"],
        },
        "git_sha": git_sha(), "config_hash": config_hash(cfg),
        "onnx_file": onnx_path.name, "onnx_sha256": sha256(onnx_path),
        "exported": datetime.now(timezone.utc).isoformat(),
        "disclaimer": cfg["disclaimer"],
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    # Parity check on REAL validation cells (random noise drives activations to ~1e3 and makes float32
    # accumulation differences look large in absolute terms; real inputs are what matter).
    import onnxruntime as ort
    from PIL import Image

    from src.data import build_transforms, load_split

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tf = build_transforms(cfg, train=False)
    val = load_split(cfg, "val").sample(32, random_state=0)
    x = torch.stack([tf(Image.open(resolve(cfg["paths"]["data_root"]) / p).convert("RGB")) for p in val.image_path])
    with torch.no_grad():
        tl, tc = model(x)
    ol, oc = sess.run(None, {"image": x.numpy()})
    tp, op = torch.sigmoid(tl).mean(1).numpy(), (1 / (1 + np.exp(-ol))).mean(1)
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 2**20:.1f} MB)")
    print(f"parity on 32 val cells: max |ensemble prob diff| {np.abs(op - tp).max():.2e}, "
          f"max |logit diff| {np.abs(ol - tl.numpy()).max():.2e} (logit scale {np.abs(tl.numpy()).max():.1f}), "
          f"max |cam diff| / cam max {np.abs(oc - tc.numpy()).max() / np.abs(tc.numpy()).max():.2e}")
    assert np.abs(op - tp).max() < 1e-3, "ONNX and PyTorch disagree on real inputs"


if __name__ == "__main__":
    main()
