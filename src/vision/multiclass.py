"""ALL vs AML vs normal lymphocyte, from single cells. Research demo, not for clinical use.

Classes
  0 normal lymphocyte : C-NMC "hem" cells + Munich typical lymphocytes (LYT)
  1 ALL blast         : C-NMC "all" cells (B-lineage ALL)
  2 AML blast         : Munich myeloblasts (MYO) + monoblasts (MOB)

Data
  C-NMC 2019 (TCIA, CC BY 3.0), already segmented on black; split by subject as everywhere else in this repo.
  Munich AML Cytomorphology (Matek et al. 2019, TCIA, CC BY 3.0): 400x400 photos centred on one cell. Each is cut out
  with the smear cell finder onto black. The red-cell ruler is unreliable on these small fields, so Munich is scaled to
  make its median normal lymphocyte as large as C-NMC's; blasts keep their true size relative to normal cells. The dataset publishes no patient IDs, so its split is
  random by image (fixed seed); its numbers are therefore optimistic, and are reported separately.

Every cell, from both sources, is colour-normalised to the C-NMC training statistics, so colour cannot reveal the lab.

The trap: ALL cells come only from C-NMC and AML cells only from Munich, so a model could learn "which lab" instead of
"which leukemia". Normal lymphocytes come from both labs, which gives the check: if Munich's normal lymphocytes drift
towards AML (or C-NMC's towards ALL), the model has learned the lab. evaluate() reports this explicitly.

Usage
  python -m src.vision.multiclass prep     # cut-outs and arrays (CPU, ~10-20 min)
  python -m src.vision.multiclass train    # GPU, ~30-45 min on the GTX 1650
  python -m src.vision.multiclass eval
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.progress import Progress
from src.utils import load_config, resolve
from src.vision.segment import find_white_cells, stain_normalise, to_cnmc

LMU = Path("E:/datasets/aml_lmu/images")
OUT = Path("E:/datasets/multiclass")
SIZE = 224
CLASSES = ["normal lymphocyte", "ALL blast", "AML blast"]
LMU_CLASS = {"LYT": 0, "MYO": 2, "MOB": 2}
CKPT = "multiclass_all_aml"


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _rbc(path: str) -> float | None:
    from scipy import ndimage as ndi

    from src.vision.segment import maps, rbc_diameter, white_balance

    st, od = maps(white_balance(_rgb(Path(path))))
    return rbc_diameter(od, exclude=ndi.binary_dilation(st > max(np.percentile(st, 97), 0.3), iterations=3))


def _one_thread() -> None:
    import cv2

    cv2.setNumThreads(1)


RBC_GUIDE_PX = 100.0  # rough red-cell size in Munich photos; only sets the finder's size limits, not the scale
CNMC_NORMAL_PX = 180.4  # median equivalent diameter of C-NMC normal cells in their 450 px frames


def _centre_cell(rgb: np.ndarray):
    res = find_white_cells(rgb, rbc_override=RBC_GUIDE_PX)
    if not res.cells:
        return None
    h, w = rgb.shape[:2]
    return min(res.cells, key=lambda c: ((c.box[0] + c.box[2]) / 2 - w / 2) ** 2 + ((c.box[1] + c.box[3]) / 2 - h / 2) ** 2)


def _lymph_size(path: str) -> float | None:
    c = _centre_cell(_rgb(Path(path)))
    return None if c is None else float(2 * np.sqrt(c.area_px / np.pi))


def _cut_lmu(args) -> np.ndarray | None:
    path, scale, stain = args
    rgb = _rgb(Path(path))
    c = _centre_cell(rgb)
    if c is None:
        return None
    crop = to_cnmc(rgb, c, scale, mode="ruler", stain=stain)
    return np.asarray(Image.fromarray(crop).resize((SIZE, SIZE), Image.BILINEAR))


def _cnmc_one(args) -> np.ndarray:
    path, stain = args
    crop = stain_normalise(_rgb(Path(path)), stain)
    return np.asarray(Image.fromarray(crop).resize((SIZE, SIZE), Image.BILINEAR))


def prep(workers: int = 4) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from src.data import load_split

    cfg = load_config()
    stain = json.loads(resolve("models/cnmc_stain.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    lmu = [(str(p), LMU_CLASS[p.parent.name]) for d in LMU_CLASS for p in sorted((LMU / d).glob("*.tiff"))]
    cn = load_split(cfg, ["train", "val", "test"])
    root = Path(cfg["paths"]["data_root"])

    with Progress("multiclass_prep", "Preparing cells for the ALL vs AML vs normal model", total=len(lmu),
                  explain=("Cuts each Munich cell out of its photo exactly the way the smear reader does, and brings C-NMC and "
                           "Munich cells to the same scale and colours, so the model cannot tell the labs apart by colour."),
                  steps=["Measure Munich normal lymphocytes", "Cut out Munich cells", "Prepare C-NMC cells"]) as pg:
        # Scale: the red-cell ruler is unreliable on Munich's small 400 px fields (readings of 35-101 px), so Munich
        # is scaled to make its median normal lymphocyte as large as C-NMC's. Blasts keep their size relative to
        # normal cells, which is real biology. Measured on 300 random Munich lymphocytes (sizes only, no labels used).
        rng = np.random.default_rng(0)
        lyt = [p for p, y in lmu if Path(p).parent.name == "LYT"]
        sample = [lyt[i] for i in rng.choice(len(lyt), 300, replace=False)]
        pg.set_stage("measuring Munich normal lymphocytes (scale)", total=len(sample), step=0)
        sizes = []
        with ProcessPoolExecutor(workers, initializer=_one_thread) as ex:
            for sz in ex.map(_lymph_size, sample, chunksize=8):
                if sz:
                    sizes.append(sz)
                pg.step()
        pooled = CNMC_NORMAL_PX / float(np.median(sizes))   # the scale factor applied to Munich photos
        pg.metrics["munich_scale"] = pooled
        pg.metrics["munich_lymphocyte_median_px"] = float(np.median(sizes))

        pg.set_stage("cutting out Munich cells", total=len(lmu), step=1)
        xs, ys, kept = [], [], []
        with ProcessPoolExecutor(workers, initializer=_one_thread) as ex:
            for (p, y), cr in zip(lmu, ex.map(_cut_lmu, ((p, pooled, stain) for p, _ in lmu), chunksize=16)):
                if cr is not None:
                    xs.append(cr)
                    ys.append(y)
                    kept.append(Path(p).relative_to(LMU).as_posix())
                pg.step()
        lmu_x, lmu_y = np.stack(xs), np.array(ys)
        # no patient IDs: random split by image, stratified by class, fixed seed
        lmu_split = np.empty(len(lmu_y), dtype="<U5")
        for c in np.unique(lmu_y):
            idx = rng.permutation(np.flatnonzero(lmu_y == c))
            n_tr, n_va = int(0.7 * len(idx)), int(0.15 * len(idx))
            lmu_split[idx[:n_tr]], lmu_split[idx[n_tr:n_tr + n_va]], lmu_split[idx[n_tr + n_va:]] = "train", "val", "test"

        pg.set_stage("preparing C-NMC cells", total=len(cn), step=2)
        cx = []
        with ProcessPoolExecutor(workers, initializer=_one_thread) as ex:
            for a in ex.map(_cnmc_one, ((str(root / p), stain) for p in cn.image_path), chunksize=32):
                cx.append(a)
                pg.step()
    cn_y = np.where(cn.label.to_numpy() == 1, 1, 0)
    np.savez(OUT / "cells.npz",
             x=np.concatenate([np.stack(cx), lmu_x]),
             y=np.concatenate([cn_y, lmu_y]),
             split=np.concatenate([cn.split.to_numpy(dtype=str), lmu_split]),
             source=np.array(["cnmc"] * len(cn_y) + ["munich"] * len(lmu_y)),
             group=np.concatenate([cn.group_id.to_numpy(dtype=str), np.array(["munich"] * len(lmu_y))]),
             name=np.concatenate([cn.image_path.to_numpy(dtype=str), np.array(kept)]))
    summary = {"munich_scale_factor": pooled, "munich_lymphocyte_median_px": float(np.median(sizes)), "munich_images": len(lmu), "munich_cut": len(lmu_y),
               "counts": {f"{s}/{CLASSES[c]}/{src}": int(((np.concatenate([cn.split.to_numpy(dtype=str), lmu_split]) == s)
                                                           & (np.concatenate([cn_y, lmu_y]) == c)
                                                           & (np.array(['cnmc'] * len(cn_y) + ['munich'] * len(lmu_y)) == src)).sum())
                          for s in ("train", "val", "test") for c in range(3) for src in ("cnmc", "munich")}}
    (OUT / "prep.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ----------------------------------------------------------------------------------- training
def _model():
    import timm

    return timm.create_model("efficientnet_b0", pretrained=True, num_classes=3, drop_rate=0.2)


def _batches(x, y, bs, dev, g, train):
    """In-memory batches with GPU augmentation: flips, 90-degree turns, small affine and colour jitter, blur and
    resolution loss (milder than the 'robust' recipe)."""
    import torch

    from src.gpu_augment import _affine, _blur, _colour, _lowres

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    order = torch.randperm(len(x), generator=g).numpy() if train else np.arange(len(x))
    n = len(x) // bs if train else -(-len(x) // bs)
    for i in range(n):
        j = np.sort(order[i * bs:(i + 1) * bs])
        xb = torch.from_numpy(x[j]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if train:
            b = xb.shape[0]
            xb = torch.where((torch.rand(b, generator=g) < 0.5).to(dev).view(b, 1, 1, 1), xb.flip(-1), xb)
            xb = torch.where((torch.rand(b, generator=g) < 0.5).to(dev).view(b, 1, 1, 1), xb.flip(-2), xb)
            k = torch.randint(0, 4, (b,), generator=g)
            xb = torch.stack([torch.rot90(xb[t], int(k[t]), dims=(-2, -1)) for t in range(b)])
            xb = _blur(_lowres(_colour(_affine(xb, g), g), g, p=0.3, lo=80, hi=200), g, p=0.3)
        yield (xb - mean) / std, torch.from_numpy(y[j]).long().to(dev)


def train(epochs: int = 15, seed: int = 0) -> None:
    import torch
    from sklearn.metrics import f1_score

    d = np.load(OUT / "cells.npz")
    x, y, split = d["x"], d["y"], d["split"]
    tr, va = split == "train", split == "val"
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    model = _model().to(dev)
    counts = np.bincount(y[tr], minlength=3)
    w = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=dev)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    bs = 32
    steps = int(tr.sum()) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * steps)
    g = torch.Generator().manual_seed(seed)
    best, path = -1.0, resolve(load_config()["paths"]["checkpoints"]) / f"{CKPT}.pt"
    xtr, ytr, xva, yva = x[tr], y[tr], x[va], y[va]
    with Progress("multiclass_train", "Training the ALL vs AML vs normal model", total=steps,
                  explain=("Teaches one network three answers: normal lymphocyte, ALL blast or AML blast. It keeps the round "
                           "with the best balanced score on validation cells it never trains on."),
                  steps=[f"Training round {e + 1} of {epochs}" for e in range(epochs)]) as pg:
        for ep in range(epochs):
            pg.set_stage(f"round {ep + 1}", total=steps, step=ep)
            model.train()
            for xb, yb in _batches(xtr, ytr, bs, dev, g, True):
                loss = crit(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                pg.step()
            pred = _predict(model, xva, dev).argmax(1)
            f1 = f1_score(yva, pred, average="macro")
            pg.metrics[f"val_macro_f1_round_{ep + 1}"] = float(f1)
            print(f"round {ep + 1}: validation macro F1 {f1:.4f}", flush=True)
            if f1 > best:
                best = f1
                torch.save(model.state_dict(), path)
    print(f"best validation macro F1 {best:.4f}; saved {path}")


def _predict(model, x, dev) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in _batches(x, np.zeros(len(x), np.int64), 128, dev, None, False):
            out.append(torch.softmax(model(xb), 1).cpu().numpy())
    model.train()
    return np.concatenate(out)


# ----------------------------------------------------------------------------------- evaluation
def evaluate() -> dict:
    import torch
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

    d = np.load(OUT / "cells.npz")
    x, y, split, source = d["x"], d["y"], d["split"], d["source"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model().to(dev)
    model.load_state_dict(torch.load(resolve(load_config()["paths"]["checkpoints"]) / f"{CKPT}.pt", map_location=dev, weights_only=True))
    te = split == "test"
    p = _predict(model, x[te], dev)
    yt, st, pred = y[te], source[te], p.argmax(1)
    prec, rec, f1, n = precision_recall_fscore_support(yt, pred, labels=[0, 1, 2], zero_division=0)
    res = {
        "test_cells": int(te.sum()),
        "per_class": {CLASSES[c]: {"n": int(n[c]), "precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c]),
                                   "roc_auc_one_vs_rest": float(roc_auc_score(yt == c, p[:, c]))} for c in range(3)},
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "accuracy": float((pred == yt).mean()),
        "confusion_matrix": {"rows_true_cols_pred": CLASSES, "matrix": confusion_matrix(yt, pred, labels=[0, 1, 2]).tolist()},
        # the lab check: where do normal lymphocytes from each lab go?
        "normal_lymphocytes_by_lab": {
            src: {"n": int(((yt == 0) & (st == src)).sum()),
                  "called_normal": float((pred[(yt == 0) & (st == src)] == 0).mean()),
                  "called_ALL": float((pred[(yt == 0) & (st == src)] == 1).mean()),
                  "called_AML": float((pred[(yt == 0) & (st == src)] == 2).mean())}
            for src in ("cnmc", "munich")},
        "notes": ["C-NMC test cells come from 26 held-out subjects; Munich test cells are a random 15% of images (no patient "
                  "IDs published), so Munich numbers are optimistic.",
                  "ALL cells come only from C-NMC and AML cells only from Munich: good scores can partly reflect the lab. "
                  "The normal-lymphocytes-by-lab block and the outside checks are what speak to that."]}
    out = resolve("reports/multiclass")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    np.savez(out / "test_probs.npz", probs=p, labels=yt, source=st)
    print(json.dumps(res, indent=2))
    return res


# ----------------------------------------------------------------------------------- outside-lab check
def outside(n_smears: int = 900) -> dict:
    """Third-lab check. Taleqani ALL smears (Tehran) and BCCD healthy smears were never used to build this model.
    If it learned 'which leukemia', Taleqani blasts should read mostly ALL and BCCD lymphocytes mostly normal; if it
    learned 'which lab', anything that does not look like C-NMC will tend to read AML."""
    import io as _io
    import re
    import zipfile

    import cv2
    import torch
    from huggingface_hub import hf_hub_download

    from src.smear_eval import CellTypeFilter, TALEQANI_FILE, TALEQANI_REPO

    stain = json.loads(resolve("models/cnmc_stain.json").read_text())
    filt = CellTypeFilter(resolve("models"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model().to(dev)
    model.load_state_dict(torch.load(resolve(load_config()["paths"]["checkpoints"]) / f"{CKPT}.pt", map_location=dev, weights_only=True))

    def crops_from(rgb: np.ndarray, rbc: float) -> np.ndarray:
        res = find_white_cells(rgb, rbc_override=rbc)
        cells = [c for c in res.cells if c.lymphoid_like and not c.touches_border]
        full = [to_cnmc(rgb, c, 157.0 / rbc, mode="ruler", stain=stain) for c in cells]
        if not full:
            return np.zeros((0, SIZE, SIZE, 3), np.uint8)
        keep = filt.keep(np.stack([np.asarray(Image.fromarray(f).resize((filt.size, filt.size), Image.BILINEAR)) for f in full]))
        return np.stack([np.asarray(Image.fromarray(f).resize((SIZE, SIZE), Image.BILINEAR)) for f, k in zip(full, keep) if k]) \
            if keep.any() else np.zeros((0, SIZE, SIZE, 3), np.uint8)

    def share(probs: np.ndarray) -> dict:
        if not len(probs):
            return {"cells": 0}
        pred = probs.argmax(1)
        return {"cells": int(len(pred)), **{f"called_{c.split()[0]}": float((pred == i).mean()) for i, c in enumerate(CLASSES)}}

    z = zipfile.ZipFile(hf_hub_download(TALEQANI_REPO, TALEQANI_FILE, repo_type="dataset"))
    names = [n for n in sorted(z.namelist()) if re.search(r"WBC-(Benign|Malignant-(Early|Pre|Pro))-\d+\.jpg$", n)]
    rng = np.random.default_rng(0)
    names = [names[i] for i in rng.choice(len(names), min(n_smears, len(names)), replace=False)]
    groups = {"ALL smears": [], "benign smears": []}
    with Progress("multiclass_outside", "Checking the three-class model on two outside labs", total=len(names) + 364,
                  explain=("Runs the model on cells from labs it never saw: B-ALL smears from Tehran and healthy smears from "
                           "BCCD. This shows whether it learned the disease or just which lab an image came from."),
                  steps=["Taleqani smears", "BCCD healthy smears"]) as pg:
        pg.set_stage("Taleqani", total=len(names), step=0)
        for n in names:
            rgb = np.asarray(Image.open(_io.BytesIO(z.read(n))).convert("RGB"))
            cr = crops_from(rgb, 17.34)
            if len(cr):
                groups["ALL smears" if "Malignant" in n else "benign smears"].append(_predict(model, cr, dev))
            pg.step()
        pg.set_stage("BCCD", total=364, step=1)
        bccd = []
        for f in sorted(Path("E:/datasets/bccd/BCCD/JPEGImages").glob("*.jpg")):
            cr = crops_from(cv2.imread(str(f))[..., ::-1].copy(), 87.7)
            if len(cr):
                bccd.append(_predict(model, cr, dev))
            pg.step()
    res = {"taleqani_" + k.replace(" ", "_"): share(np.concatenate(v)) if v else {"cells": 0} for k, v in groups.items()}
    res["bccd_healthy"] = share(np.concatenate(bccd)) if bccd else {"cells": 0}
    res["note"] = ("Cells pass the same steps as the smear page: lymphoid shape rule, cell-type filter, red-cell ruler and "
                   "colour normalisation. Taleqani has B-ALL only, so AML calls there are errors.")
    out = resolve("reports/multiclass")
    out.mkdir(parents=True, exist_ok=True)
    (out / "outside.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prep", "train", "eval", "outside"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()
    {"prep": lambda: prep(a.workers), "train": lambda: train(a.epochs), "eval": evaluate, "outside": outside}[a.what]()
    print(f"{time.time() - t0:.0f}s")
