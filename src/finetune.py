"""Fine-tune DinoBloom's last blocks on three labs (blast vs normal lymphocyte), judged by leave one lab out.

Plan, fixed before results:
  * Start: DinoBloom-B (or -L), all blocks frozen except the last N (default 4) and the final norm; a linear head
    initialised from the frozen three-lab logistic regression of `src.multilab final` is NOT used, so the run does not
    depend on it; the head starts at zero.
  * Data: reports/multilab/manifest.csv. C-NMC test subjects never train. Each (lab, class) group gets equal weight.
  * Augmentation: flips, 90-degree turns, +-15% zoom, mild colour/brightness jitter, occasional blur (other labs'
    images are softer). No augmentation at evaluation.
  * Fixed recipe, no early stopping on the held-out lab: AdamW, lr 1e-5 (backbone) / 1e-3 (head), weight decay 0.05,
    EPOCHS epochs, cosine schedule, AMP. Same recipe for every fold.
  * --held cnmc|jhu|lmu trains on the other two labs and reports the held-out lab with the same metrics as
    `src.multilab lolo` (C-NMC: test subjects only). --held none trains on all three (for the outside tests).

Runs on the laptop GPU (GTX 1650: batch 16, gradient checkpointing) or on Kaggle (T4: batch 64). Paths come from
MULTILAB_CROPS (Hopkins/Munich crop folder) and CNMC_ROOT (C-NMC images) so the same file works in both places.

Usage
  python -m src.finetune --held jhu
  python -m src.finetune --held none --out reports/finetune/all.pt
Research demo. Not for clinical use.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
NAMES = {"b": "vit_base_patch14_dinov2", "l": "vit_large_patch14_dinov2"}


def resolve_paths(man: pd.DataFrame) -> list[str]:
    """Laptop: C-NMC originals under CNMC_ROOT, other labs' crops under MULTILAB_CROPS/<lab>/. Kaggle (MULTILAB_FLAT set):
    every crop, C-NMC included (pre-resized to 224 px, the size the model reads), at MULTILAB_FLAT/<lab>/<file name>."""
    flat = os.environ.get("MULTILAB_FLAT")
    if flat:
        return [str(Path(flat) / src / str(p).replace("\\", "/").split("/")[-1]) for src, p in zip(man.source, man.crop)]
    crops = Path(os.environ.get("MULTILAB_CROPS", "E:/cache/multilab"))
    cnmc = Path(os.environ.get("CNMC_ROOT", "E:/datasets/cnmc"))
    return [str(cnmc / p) if src == "cnmc" else str(crops / src / Path(p).name) for src, p in zip(man.source, man.crop)]


def _boot(y, p, g, iters=1000, seed=0):
    """Same metrics as src.multilab._boot (copied so this file runs alone on Kaggle)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ug} if len(ug) < len(g) else None
    roc, pr = [], []
    for _ in range(iters):
        i = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]) if idx else rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            roc.append(roc_auc_score(y[i], p[i]))
            pr.append(average_precision_score(y[i], p[i]))
    q = lambda v: [float(a) for a in np.percentile(v, [2.5, 97.5])]
    return {"roc_auc": float(roc_auc_score(y, p)), "roc_auc_ci95": q(roc),
            "pr_auc": float(average_precision_score(y, p)), "pr_auc_ci95": q(pr),
            "normal_called_leukemic_at_0.5": float((p[y == 0] > 0.5).mean()),
            "blasts_caught_at_0.5": float((p[y == 1] > 0.5).mean()),
            "cells": {"normal": int((y == 0).sum()), "blast": int((y == 1).sum())}}


class Cells:
    def __init__(self, paths, y, w, train: bool, seed: int = 0):
        self.paths, self.y, self.w, self.train = paths, y, w, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch

        im = Image.open(self.paths[i]).convert("RGB")
        if self.train:
            r = self.rng
            if r.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.rotate(90 * int(r.integers(4)))
            z = r.uniform(0.85, 1.15)                      # zoom about the centre, black border like the cut-outs
            s = im.size[0]
            n = max(8, int(round(s * z)))
            im = im.resize((n, n), Image.BILINEAR)
            canvas = Image.new("RGB", (max(n, s), max(n, s)))
            canvas.paste(im, ((max(n, s) - n) // 2, (max(n, s) - n) // 2))
            o = (max(n, s) - s) // 2
            im = canvas.crop((o, o, o + s, o + s))
            if r.random() < 0.3:
                im = im.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.2)))
        im = im.resize((224, 224), Image.BILINEAR)
        x = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
        if self.train:
            r = self.rng
            fg = x.max(0, keepdims=True) > 0.03            # jitter the cell only, keep the black background black
            x = np.where(fg, np.clip(x * r.uniform(0.9, 1.1, (3, 1, 1)) + r.uniform(-0.04, 0.04, (3, 1, 1)), 0, 1), x)
        x = (x - MEAN) / STD
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(float(self.y[i])), torch.tensor(float(self.w[i]))


def build_model(variant: str, train_blocks: int):
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from timm.models.vision_transformer import checkpoint_filter_fn

    m = timm.create_model(NAMES[variant], pretrained=False, img_size=224, num_classes=0)
    ck = torch.load(hf_hub_download("MarrLab/DinoBloom", f"pytorch_model_{variant}.bin"), map_location="cpu", weights_only=True)
    m.load_state_dict(checkpoint_filter_fn(ck, m), strict=True)
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks[-train_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in m.norm.parameters():
        p.requires_grad = True

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = m
            self.head = torch.nn.Linear(m.num_features, 1)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, x):
            return self.head(self.backbone(x))[:, 0]

    return Net()


def weights(src, y):
    w = np.zeros(len(y))
    for s in np.unique(src):
        for c in (0, 1):
            m = (src == s) & (y == c)
            if m.any():
                w[m] = 1.0 / m.sum()
    return w * len(y) / w.sum()


def predict(net, paths, dev, batch):
    import torch

    dl = torch.utils.data.DataLoader(Cells(paths, np.zeros(len(paths)), np.ones(len(paths)), False), batch_size=batch,
                                     num_workers=int(os.environ.get("WORKERS", 2)))
    out = []
    net.eval()
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        for x, _, _ in dl:
            out.append(torch.sigmoid(net(x.to(dev)).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--held", required=True, choices=["cnmc", "jhu", "lmu", "none"])
    ap.add_argument("--variant", default="b", choices=["b", "l"])
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 16)))
    ap.add_argument("--manifest", default=os.environ.get("MULTILAB_MANIFEST", "reports/multilab/manifest.csv"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(a.manifest, low_memory=False)
    paths = np.array(resolve_paths(man))
    y, src, g = man.label.to_numpy(int), man.source.to_numpy(str), man.group.to_numpy(str)
    split = man.get("cnmc_split", pd.Series([None] * len(man))).fillna("").to_numpy(str)
    trainable = ((src != "cnmc") | (split != "test")) & (src != a.held)
    test = (src == a.held) & ((src != "cnmc") | (split == "test"))

    net = build_model(a.variant, a.blocks).to(dev)
    if dev.type == "cuda" and a.batch <= 16:
        net.backbone.set_grad_checkpointing(True)
    w = weights(src[trainable], y[trainable])
    ds = Cells(paths[trainable].tolist(), y[trainable], w, True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     num_workers=int(os.environ.get("WORKERS", 2)),
                                     persistent_workers=int(os.environ.get("WORKERS", 2)) > 0)
    params = [{"params": [p for p in net.backbone.parameters() if p.requires_grad], "lr": 1e-5},
              {"params": net.head.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    log = {"args": vars(a), "train_cells": int(trainable.sum()), "epochs": []}
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        tot, n = 0.0, 0
        for i, (x, t, wt) in enumerate(dl):
            x, t, wt = x.to(dev), t.to(dev), wt.to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(x).float(), t, reduction="none") * wt).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(x)
            n += len(x)
            if i % 100 == 0:
                print(f"epoch {ep} step {i}/{len(dl)} loss {tot / n:.4f} {time.time() - t0:.0f}s", flush=True)
        log["epochs"].append({"epoch": ep, "loss": tot / n, "seconds": time.time() - t0})

    out = Path(a.out or f"reports/finetune/dinobloom_{a.variant}_ft_held_{a.held}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out)
    if a.held != "none":
        p = predict(net, paths[test].tolist(), dev, a.batch * 2)
        log["held_out"] = _boot(y[test], p, g[test])
        np.save(out.with_suffix(".probs.npy"), p)
        print(a.held, json.dumps(log["held_out"], default=float), flush=True)
    out.with_suffix(".json").write_text(json.dumps(log, indent=2, default=float))



def outside_ft(pt: str = "reports/finetune/dinobloom_b_ft_held_none.pt") -> dict:
    """Taleqani + BCCD outside tests for the fine-tuned three-lab network, with the DinoBloom cell filter (frozen features)
    deciding which cells are read, exactly as for the frozen model in `src.multilab final --dbfilter`."""
    import torch

    from src.foundation import outside
    from src.multilab import OUT, fewshot

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model("b", 4)
    net.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    net = net.to(dev).eval()

    def scorer(u8: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            for i in range(0, len(u8), 64):
                x = (u8[i:i + 64].astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - MEAN[None]) / STD[None]
                out.append(torch.sigmoid(net(torch.from_numpy(x.astype(np.float32)).to(dev)).float()).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    res_path = OUT / "outside_results_finetuned_dbfilter.json"
    res = outside(None, res_path, job="finetuned_outside", emb_filter=OUT / "dinobloom_b_filter.joblib", scorer=scorer)
    res["classifier"] = pt
    res["fewshot_cutoff"] = {arm: fewshot(res_path.with_suffix(".npz"), arm) for arm in ("ruler_x1.0_stain_filtered", "ruler_x1.0_stain")}
    res_path.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    import sys
    outside_ft(*sys.argv[2:3]) if sys.argv[1:2] == ["outside"] else main()
