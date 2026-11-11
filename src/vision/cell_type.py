"""Learned white-cell type filter: should a cell found in a smear be read by the ALL classifier at all?

The shape rule in segment.py lets many granulocytes through (most cells it kept in healthy BCCD smears were
neutrophils), and the ALL model was never trained on them. This filter answers "lymphoid, keep" vs "other
white cell, drop", from the same cut-out the classifier sees (cell on black, red-cell scale, C-NMC colours).

Training data
  keep: PBC lymphocytes (Acevedo et al. 2020, CC BY 4.0) + C-NMC training cells (normal and leukemic), so
        that lymphoblasts, which PBC does not contain, are kept too
  drop: PBC neutrophils, eosinophils, basophils, monocytes, immature granulocytes, erythroblasts
  PBC platelets are not used (the finder never returns platelets as cells).

Checks (python -m src.vision.cell_type eval)
  PBC held-out images: per-class keep rate (lymphocytes high, everything else low)
  C-NMC validation cells: keep rate for normal and for leukemic cells separately (must stay high: a filter
  that drops blasts would hide exactly the cells that matter). The keep threshold is chosen on C-NMC
  TRAINING cells and PBC validation images only; no outside smear data is used.

Usage
  python -m src.vision.cell_type prep     # cut cells out of PBC images (CPU, a few minutes)
  python -m src.vision.cell_type train    # ~10-15 min on the GTX 1650
  python -m src.vision.cell_type eval
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.progress import Progress
from src.utils import load_config, resolve
from src.vision.segment import CNMC_RBC_PX, find_white_cells, maps, rbc_diameter, to_cnmc, white_balance

PBC_REPO, PBC_FILE = "Docty/Blood-Cells", "data/train-00000-of-00001.parquet"
PBC_NAMES = {0: "monocyte", 1: "ig", 2: "neutrophil", 3: "basophil", 4: "lymphocyte", 5: "erythroblast", 6: "eosinophil", 7: "platelet"}
OUT = Path("E:/datasets/pbc_cells")
SIZE = 160          # filter input side; cell type needs less detail than ALL vs normal
CKPT = "cell_type_filter"


def _rbc(data: bytes) -> float | None:
    from scipy import ndimage as ndi

    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    st, od = maps(white_balance(rgb))
    return rbc_diameter(od, exclude=ndi.binary_dilation(st > max(np.percentile(st, 97), 0.3), iterations=3))


def _one_thread() -> None:
    """Worker start-up: one OpenCV thread per process, or the workers fight each other and the training job."""
    import cv2

    cv2.setNumThreads(1)


def _cut(args) -> np.ndarray | None:
    """The white cell nearest the image centre (PBC images are centred on one cell), as a C-NMC-style crop."""
    data, pooled, stain = args
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    res = find_white_cells(rgb, rbc_override=pooled)
    if not res.cells:
        return None
    h, w = rgb.shape[:2]
    c = min(res.cells, key=lambda c: ((c.box[0] + c.box[2]) / 2 - w / 2) ** 2 + ((c.box[1] + c.box[3]) / 2 - h / 2) ** 2)
    crop = to_cnmc(rgb, c, CNMC_RBC_PX / pooled, mode="ruler", stain=stain)
    return np.asarray(Image.fromarray(crop).resize((SIZE, SIZE), Image.BILINEAR))


def prep(workers: int = 3) -> None:
    import os
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HOME", "E:/cache/hf")
    t = pq.read_table(hf_hub_download(PBC_REPO, PBC_FILE, repo_type="dataset")).to_pandas()
    t = t[t.label != 7].reset_index(drop=True)                      # no platelets
    t["bytes"] = t["image"].map(lambda d: d["bytes"])               # images are stored as {bytes, path} records
    stain = json.loads(resolve("models/cnmc_stain.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    with Progress("pbc_prep", "Cutting single white cells out of the PBC reference photos", total=len(t),
                  explain=("Prepares training examples for the cell-type filter: 14,700 expert-labelled white cells from a "
                           "Barcelona hospital, each cut out exactly the way the smear reader cuts cells."),
                  steps=["Measure the red cells (zoom level)", "Cut out every white cell"]) as pg:
        # one microscope (CellaVision DM96), one magnification: 60 random images give the pooled ruler
        sample = t.bytes.sample(60, random_state=0).tolist()
        pg.set_stage("measuring red cells on 60 sample photos", total=len(sample), step=0)
        est = []
        for b in sample:
            e = _rbc(b)
            if e:
                est.append(e)
            pg.step()
        pooled = float(np.median(est))
        pg.metrics["red_cell_diameter_px"] = pooled
        pg.set_stage("cutting out cells", total=len(t), step=1)
        crops, keep_idx = [], []
        with ProcessPoolExecutor(workers, initializer=_one_thread) as ex:
            for i, cr in enumerate(ex.map(_cut, ((b, pooled, stain) for b in t.bytes), chunksize=16)):
                if cr is not None:
                    crops.append(cr)
                    keep_idx.append(i)
                pg.step()
    lab = t.label.to_numpy()[keep_idx]
    rng = np.random.default_rng(0)
    split = np.where(rng.random(len(lab)) < 0.8, "train", np.where(rng.random(len(lab)) < 0.5, "val", "test"))
    np.savez_compressed(OUT / "crops.npz", x=np.stack(crops), label=lab, split=split)
    found = pd.Series(lab).map(PBC_NAMES).value_counts().to_dict()
    (OUT / "prep.json").write_text(json.dumps({"rbc_pooled_px": pooled, "images": len(t), "cells_cut": len(lab),
                                               "per_class": found}, indent=2))
    print(f"cut {len(lab)} of {len(t)} cells; red cell {pooled:.1f} px; {found}")


# ----------------------------------------------------------------------------------- training
def _cnmc_cells(cfg, split: str, n: int, seed: int) -> np.ndarray:
    from src.data import load_split

    df = load_split(cfg, split).sample(n, random_state=seed)
    root = Path(cfg["paths"]["data_root"])
    return np.stack([np.asarray(Image.open(root / p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)) for p in df.image_path]), df.label.to_numpy()


class _DS:
    def __init__(self, x, y, train):
        import torch
        import torchvision.transforms.v2 as T

        from src.data import HEDJitter, RandomLowRes, RandomRot90

        self.x, self.y, self.torch = x, y.astype(np.float32), torch
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), RandomRot90(),
               T.RandomAffine(degrees=15, scale=(0.8, 1.25)),
               T.ColorJitter(0.25, 0.25, 0.3, 0.04), RandomLowRes(24, 140, p=0.6),
               T.RandomApply([T.GaussianBlur(7, (0.5, 2.0))], p=0.4), HEDJitter()] if train else []
        self.tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), *aug,
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


def _build():
    import timm
    import torch

    return timm.create_model("efficientnet_b0", pretrained=True, num_classes=1, drop_rate=0.2), torch


def _gpu_batches(x: np.ndarray, y: np.ndarray, bs: int, dev, g, train: bool):
    """Batches from in-memory uint8 crops, augmented on the GPU with the same steps as the robust recipe
    (Windows data-loader workers cannot be handed a 1 GB array)."""
    import torch

    from src.gpu_augment import _affine, _blur, _colour, _hed, _lowres

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    order = torch.randperm(len(x), generator=g).numpy() if train else np.arange(len(x))
    n = len(x) // bs if train else -(-len(x) // bs)
    for i in range(n):
        j = order[i * bs:(i + 1) * bs]
        xb = torch.from_numpy(x[j]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if train:
            b = xb.shape[0]
            xb = torch.where((torch.rand(b, generator=g) < 0.5).to(dev).view(b, 1, 1, 1), xb.flip(-1), xb)
            xb = torch.where((torch.rand(b, generator=g) < 0.5).to(dev).view(b, 1, 1, 1), xb.flip(-2), xb)
            k = torch.randint(0, 4, (b,), generator=g)
            xb = torch.stack([torch.rot90(xb[t], int(k[t]), dims=(-2, -1)) for t in range(b)])
            xb = _hed(_blur(_lowres(_colour(_affine(xb, g), g), g, p=0.6, lo=24, hi=140), g, p=0.4), g)
        yield (xb - mean) / std, torch.from_numpy(y[j].astype(np.float32)).to(dev)


def train(epochs: int = 8) -> None:
    import torch
    from sklearn.metrics import roc_auc_score

    cfg = load_config()
    d = np.load(OUT / "crops.npz")
    x, lab, split = d["x"], d["label"], d["split"]
    keep = (lab == 4).astype(int)                                   # lymphocyte = keep
    cx, _ = _cnmc_cells(cfg, "train", 3000, 0)                      # C-NMC training cells, both classes: keep
    xtr = np.concatenate([x[split == "train"], cx[:2700]])
    ytr = np.concatenate([keep[split == "train"], np.ones(2700, int)])
    xva = np.concatenate([x[split == "val"], cx[2700:]])
    yva = np.concatenate([keep[split == "val"], np.ones(300, int)])
    model, _ = _build()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    g = torch.Generator().manual_seed(0)
    pos = ytr.mean()
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos) / pos], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    bs = 32
    steps = len(xtr) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * steps)
    best, path = -1.0, resolve(cfg["paths"]["checkpoints"]) / f"{CKPT}.pt"

    with Progress("cell_type_train", "Teaching a filter to tell lymphocytes from other white cells", total=steps,
                  explain=("Trains a small network that looks at a cut-out cell and says 'lymphocyte, let the leukemia model "
                           "read it' or 'other white cell, skip it'. This stops neutrophils from being scored as leukemia."),
                  steps=[f"Training round {e + 1} of {epochs}" for e in range(epochs)]) as pg:
        for ep in range(epochs):
            pg.set_stage(f"epoch {ep + 1}", total=steps, step=ep)
            model.train()
            for xb, yb in _gpu_batches(xtr, ytr, bs, dev, g, True):
                loss = crit(model(xb).squeeze(1), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                pg.step()
            model.eval()
            ps = []
            with torch.no_grad():
                for xb, _ in _gpu_batches(xva, yva, 128, dev, g, False):
                    ps.append(torch.sigmoid(model(xb).squeeze(1)).cpu().numpy())
            auc = roc_auc_score(yva, np.concatenate(ps))
            pg.metrics[f"val_auc_round_{ep + 1}"] = float(auc)
            print(f"epoch {ep + 1}: validation AUC {auc:.4f}", flush=True)
            if auc > best:
                best = auc
                torch.save(model.state_dict(), path)
    print(f"best validation AUC {best:.4f}; saved {path}")


# ----------------------------------------------------------------------------------- evaluation and export
def _predict(model, torch, x: np.ndarray, dev) -> np.ndarray:
    ds, out = _DS(x, np.zeros(len(x)), False), []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            xb = torch.stack([ds[j][0] for j in range(i, min(len(x), i + 128))]).to(dev)
            out.append(torch.sigmoid(model(xb).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def evaluate() -> dict:
    cfg = load_config()
    model, torch = _build()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(resolve(cfg["paths"]["checkpoints"]) / f"{CKPT}.pt", map_location=dev, weights_only=True))
    model.to(dev).eval()
    d = np.load(OUT / "crops.npz")
    x, lab, split = d["x"], d["label"], d["split"]

    # threshold: the highest that still keeps >= 98% of C-NMC TRAINING cells (blasts included) and PBC val lymphocytes
    cx_tr, cy_tr = _cnmc_cells(cfg, "train", 1500, 1)
    p_c = _predict(model, torch, cx_tr, dev)
    p_lv = _predict(model, torch, x[(split == "val") & (lab == 4)], dev)
    thr = float(min(np.percentile(p_c, 2), np.percentile(p_lv, 2)))

    test = split == "test"
    p_t = _predict(model, torch, x[test], dev)
    per_class = {PBC_NAMES[k]: {"n": int((lab[test] == k).sum()), "kept": float((p_t[lab[test] == k] >= thr).mean())}
                 for k in sorted(set(lab[test]))}
    cx_va, cy_va = _cnmc_all_val(cfg)
    p_v = _predict(model, torch, cx_va, dev)
    other = np.isin(lab[test], [0, 1, 2, 3, 5, 6])
    res = {"threshold": thr,
           "threshold_rule": "keep >= 98% of C-NMC training cells and PBC validation lymphocytes",
           "pbc_test_per_class": per_class,
           "pbc_test_other_white_cells_dropped": float((p_t[other] < thr).mean()),
           "cnmc_validation_kept": {"normal": float((p_v[cy_va == 0] >= thr).mean()), "leukemic": float((p_v[cy_va == 1] >= thr).mean())}}
    (resolve("models") / "cell_type_meta.json").write_text(json.dumps({**res, "input_size": SIZE, "onnx_file": f"{CKPT}.onnx"}, indent=2))
    # export to ONNX for the API (CPU, no torch needed there)
    model.cpu()
    torch.onnx.export(model, torch.randn(1, 3, SIZE, SIZE), resolve("models") / f"{CKPT}.onnx", input_names=["image"],
                      output_names=["logit"], dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False)
    print(json.dumps(res, indent=2))
    return res


def _cnmc_all_val(cfg):
    from src.data import load_split

    df = load_split(cfg, "val")
    root = Path(cfg["paths"]["data_root"])
    return np.stack([np.asarray(Image.open(root / p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)) for p in df.image_path]), df.label.to_numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prep", "train", "eval"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    {"prep": lambda: prep(a.workers), "train": train, "eval": evaluate}[a.what]()
    print(f"{time.time() - t0:.0f}s", file=sys.stderr)
