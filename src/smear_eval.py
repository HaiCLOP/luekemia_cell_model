"""Smear-level evaluation of the cell finder + the frozen single-cell classifier.

Usage
  python -m src.smear_eval bccd                 # cell finding and the red-cell ruler, against BCCD boxes
  python -m src.smear_eval taleqani             # ALL vs benign smears, end to end (~30-60 min on CPU)
  python -m src.smear_eval taleqani --limit 200 # quick run on a subset

Nothing here trains or tunes anything. The analysis plan was written before the first Taleqani run:

  PRIMARY   crops scaled with the red-cell ruler (x1.0, target CNMC_RBC_PX), red-cell diameter pooled
            over the whole source (one microscope setup), each smear scored by the MEAN calibrated
            probability of its lymphoid-like cells; ROC-AUC and PR-AUC for malignant vs benign smears.
            The deployed band is applied to that mean: < t_low normal, > t_high leukemic, else refer.
  SECONDARY (reported, never used to pick settings): "ruler_x1.0_stain", the primary crops with
            Reinhard colour normalisation to C-NMC (added after BCCD showed a large colour gap, and fixed
            before any Taleqani result was seen), crop modes "fit" (every cell resized to a median
            C-NMC normal cell, which removes absolute size) and "native" (no rescaling), ruler x0.8 and
            x1.2, max and fraction-above-band aggregations, and sensitivity per ALL subtype.

Data
  BCCD (MIT licence): 364 smear images, 640 x 480, boxes for white cells, red cells and platelets.
    git clone https://github.com/Shenggan/BCCD_Dataset E:/datasets/bccd
  Taleqani Hospital ALL dataset (Aria et al. 2021, Kaggle, doi 10.34740/KAGGLE/DSV/2175623), read from
    the Hugging Face mirror pritam06/leukemia-cancer-data (224 x 224 JPEGs). Labels come from file names
    (Benign = hematogones from patients without ALL; Malignant-Early/Pre/Pro = B-ALL subtypes); the
    mirror's folder names are not used because they mix classes. No patient IDs are published, so
    intervals resample images and are optimistic. Only aggregate numbers are written.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import io
import json
import re
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from api.inference import Model
from src.vision.segment import CNMC_RBC_PX, analyse, find_white_cells, maps, rbc_diameter, to_cnmc, white_balance
from src.progress import Progress
from src.utils import resolve

BCCD = Path("E:/datasets/bccd")
TALEQANI_REPO, TALEQANI_FILE = "pritam06/leukemia-cancer-data", "leukemia.zip"
OUT = resolve("reports/smear")


# ----------------------------------------------------------------------------------- helpers
def pooled_rbc(images, pg: Progress | None = None) -> tuple[float, list[float]]:
    est = []
    for rgb in images:
        if pg:
            pg.step()
        st, od = maps(white_balance(rgb))
        e = rbc_diameter(od, exclude=ndi_dilate(st > max(np.percentile(st, 97), 0.3)))
        if e:
            est.append(e)
    return float(np.median(est)), est


def ndi_dilate(m, it=3):
    from scipy import ndimage as ndi

    return ndi.binary_dilation(m, iterations=it)


def boot_ci(fn, y, s, iters=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(fn(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


class TorchEnsemble:
    """The same five checkpoints the ONNX file was exported from, run on the GPU. Used only to make long
    evaluations fast; export parity (src/export.py) checks the two agree to < 1e-3 in probability, and
    run_taleqani re-checks agreement on real smear crops before trusting it."""

    def __init__(self, model: Model, prefix: str = "effb0_main"):
        import torch

        from src.model import build_model
        from src.utils import load_config

        cfg = load_config()
        self.torch, self.dev = torch, torch.device("cuda")
        self.members = []
        for sd in cfg["calibration"]["ensemble_seeds"]:
            m = build_model(cfg).eval().to(self.dev)
            m.load_state_dict(torch.load(resolve(cfg["paths"]["checkpoints"]) / f"{prefix}_s{sd}.pt", map_location=self.dev, weights_only=True))
            self.members.append(m)
        self.model = model

    def logits(self, xs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = self.torch.from_numpy(xs).to(self.dev)
            return self.torch.cat([m(x) for m in self.members], dim=1).float().cpu().numpy()


BACKEND: dict = {}


def score_crops(model: Model, crops: list[np.ndarray], batch: int = 32, with_maxabs: bool = False):
    """Calibrated probability per crop (same preprocessing and calibration as the API). Uses the GPU copy
    of the ensemble when one was set up in BACKEND, otherwise the ONNX file on the CPU."""
    gpu = BACKEND.get("torch")
    L = []
    if gpu is not None:
        xs = np.concatenate([model.preprocess(Image.fromarray(c))[0] for c in crops], axis=0)
        for i in range(0, len(xs), 128):
            L.append(gpu.logits(xs[i:i + 128]))
    else:
        for i in range(0, len(crops), batch):
            xs = np.concatenate([model.preprocess(Image.fromarray(c))[0] for c in crops[i:i + batch]], axis=0)
            L.append(model.session.run(None, {"image": xs})[0])
    L = np.concatenate(L, axis=0) if L else np.zeros((0, 5), np.float32)
    probs = np.array([model.calibrate(l)[1] for l in L])
    return (probs, np.abs(L).max(axis=1)) if with_maxabs else probs


def _healthy_stats(p: np.ndarray, mx: np.ndarray, model: Model) -> dict:
    if not len(p):
        return {"n": 0}
    ok = mx <= model.ood_logit
    q = p[ok]
    return {"n": int(len(p)), "flagged_out_of_distribution": float((~ok).mean()),
            "after_guard_n": int(ok.sum()),
            "called_leukemic": float((q > model.t_high).mean()) if len(q) else None,
            "referred": float(((q >= model.t_low) & (q <= model.t_high)).mean()) if len(q) else None,
            "called_normal": float((q < model.t_low).mean()) if len(q) else None}


# ----------------------------------------------------------------------------------- BCCD
def run_bccd(method: str, model_dir: str = "models", use_filter: bool = False) -> dict:
    ann = pd.read_csv(BCCD / "test.csv")
    files = sorted(ann.filename.unique())
    imgs = {f: cv2.imread(str(BCCD / "BCCD" / "JPEGImages" / f)) for f in files}
    imgs = {f: im[..., ::-1].copy() for f, im in imgs.items() if im is not None}
    pooled, per_img = pooled_rbc(imgs.values())

    rows, t0 = [], time.time()
    pg = Progress(f"bccd_{method}", f"Cell finding on BCCD ({method})", total=len(imgs), stage="matching detections to boxes")
    for f, rgb in imgs.items():
        pg.step(note=f)
        g = ann[ann.filename == f]
        wbc = g[g.cell_type == "WBC"]
        rbc = g[g.cell_type == "RBC"]
        H, W = rgb.shape[:2]
        full = rbc[(rbc.xmin > 2) & (rbc.ymin > 2) & (rbc.xmax < W - 2) & (rbc.ymax < H - 2)]
        res = find_white_cells(rgb, method=method, rbc_override=pooled)
        cents = [(np.mean([c.box[0], c.box[2]]), np.mean([c.box[1], c.box[3]])) for c in res.cells]
        inside = [[(b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax) for _, b in wbc.iterrows()] for x, y in cents]
        hit_gt = {j for row in inside for j, v in enumerate(row) if v}
        tp_det = sum(any(r) for r in inside)
        rows.append({"file": f, "gt_wbc": len(wbc), "found_gt": len(hit_gt), "detections": len(res.cells),
                     "dets_in_gt": tp_det, "gt_boxes_hit_twice": sum(sum(r[j] for r in inside) > 1 for j in hit_gt),
                     "rbc_truth": float(((full.xmax - full.xmin + full.ymax - full.ymin) / 2).median()) if len(full) >= 3 else np.nan})
    # healthy donors: every lymphoid-like cell the finder keeps should read as normal
    pg.set_stage("scoring lymphoid-like cells from healthy smears", total=len(imgs))
    model = Model(resolve(model_dir))
    stain = json.loads(resolve("models/cnmc_stain.json").read_text())
    filt = CellTypeFilter(resolve("models")) if use_filter else None
    healthy = {"raw_colour": ([], []), "stain_normalised": ([], [])}
    if filt is not None:
        healthy["stain_normalised_filtered"] = ([], [])
    n_cells, n_kept = 0, 0
    for f, rgb in imgs.items():
        res = find_white_cells(rgb, method=method, rbc_override=pooled)
        cells = [c for c in res.cells if c.lymphoid_like and not c.touches_border]
        if cells:
            for k, st in (("raw_colour", None), ("stain_normalised", stain)):
                full = [to_cnmc(rgb, c, CNMC_RBC_PX / pooled, mode="ruler", stain=st) for c in cells]
                p, mx = score_crops(model, full, with_maxabs=True)
                healthy[k][0].extend(p)
                healthy[k][1].extend(mx)
                if filt is not None and st is not None:
                    keep = filt.keep(np.stack([np.asarray(Image.fromarray(f).resize((filt.size, filt.size), Image.BILINEAR)) for f in full]))
                    n_cells += len(keep)
                    n_kept += int(keep.sum())
                    healthy["stain_normalised_filtered"][0].extend(p[keep])
                    healthy["stain_normalised_filtered"][1].extend(mx[keep])
        pg.step(note=f)
    pg.status = "finished"
    pg._write(force=True)
    df = pd.DataFrame(rows)
    est = np.array(per_img)
    truth = df.rbc_truth.dropna().to_numpy()
    summary = {
        "method": method, "images": len(df), "seconds_per_image": (time.time() - t0) / len(df),
        "white_cells": {"ground_truth": int(df.gt_wbc.sum()),
                        "recall": float(df.found_gt.sum() / df.gt_wbc.sum()),
                        "precision": float(df.dets_in_gt.sum() / max(1, df.detections.sum())),
                        "boxes_split_into_several_detections": int(df.gt_boxes_hit_twice.sum())},
        "red_cell_ruler": {"pooled_estimate_px": pooled, "truth_median_px": float(np.median(truth)),
                           "pooled_error": float(pooled / np.median(truth) - 1),
                           "per_image_within_15pct_of_truth_median": float(np.mean(np.abs(est / np.median(truth) - 1) < 0.15)),
                           "per_image_cv": float(est.std() / est.mean())},
        "healthy_lymphoid_cells": {
            "note": ("BCCD comes from healthy donors, so lymphoid-like cells should read normal. Most cells the shape "
                     "filter keeps here are in fact granulocytes, which the model was never trained on."),
            **{k: _healthy_stats(np.array(v[0]), np.array(v[1]), model) for k, v in healthy.items()},
            "filter_kept": (n_kept / n_cells) if n_cells else None},
        "model_dir": model_dir, "cell_type_filter": use_filter,
    }
    return summary


# ----------------------------------------------------------------------------------- Taleqani
ARM_SPECS = [("ruler_x1.0", "ruler", 1.0, False), ("ruler_x1.0_stain", "ruler", 1.0, True),
             ("ruler_x0.8", "ruler", 0.8, False), ("ruler_x1.2", "ruler", 1.2, False),
             ("fit", "fit", None, False), ("native", "native", None, False)]


def _rbc_one(data: bytes) -> float | None:
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    st, od = maps(white_balance(rgb))
    return rbc_diameter(od, exclude=ndi_dilate(st > max(np.percentile(st, 97), 0.3)))


def _crops_one(args) -> dict:
    """Worker: find cells in one smear and return, per arm, the crops already resized to the model's input
    size (uint8, the same PIL bilinear resize Model.preprocess does), so only small arrays cross processes."""
    data, pooled, stain, method, size, arm_names, filter_size = args
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    res = find_white_cells(rgb, method=method, rbc_override=pooled)
    cells = [c for c in res.cells if c.lymphoid_like and not c.touches_border]
    out = {"cells_found": len(res.cells), "cells_scored": len(cells), "crops": {}}
    for name, mode, mult, use_stain in ARM_SPECS:
        if name not in arm_names:
            continue
        scale = (CNMC_RBC_PX / pooled) * mult if mult else None
        full = [to_cnmc(rgb, c, scale, mode=mode, stain=stain if use_stain else None) for c in cells]
        cr = [np.asarray(Image.fromarray(f).resize((size, size), Image.BILINEAR)) for f in full]
        out["crops"][name] = np.stack(cr) if cr else np.zeros((0, size, size, 3), np.uint8)
        if filter_size and name == "ruler_x1.0_stain":
            fc = [np.asarray(Image.fromarray(f).resize((filter_size, filter_size), Image.BILINEAR)) for f in full]
            out["filter_crops"] = np.stack(fc) if fc else np.zeros((0, filter_size, filter_size, 3), np.uint8)
    return out


class CellTypeFilter:
    """The learned lymphoid / other-white-cell filter (src/vision/cell_type.py), exported to ONNX."""

    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        self.meta = json.loads((model_dir / "cell_type_meta.json").read_text())
        self.size, self.thr = int(self.meta["input_size"]), float(self.meta["threshold"])
        self.sess = ort.InferenceSession(str(model_dir / self.meta["onnx_file"]), providers=["CPUExecutionProvider"])
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

    def keep(self, crops: np.ndarray) -> np.ndarray:
        if not len(crops):
            return np.zeros(0, bool)
        x = ((crops.astype(np.float32).transpose(0, 3, 1, 2) / 255.0) - self.mean) / self.std
        z = self.sess.run(None, {"image": x})[0].reshape(-1)
        return 1 / (1 + np.exp(-np.clip(z, -50, 50))) >= self.thr


def taleqani_items(limit: int | None):
    from huggingface_hub import hf_hub_download

    z = zipfile.ZipFile(hf_hub_download(TALEQANI_REPO, TALEQANI_FILE, repo_type="dataset"))
    seen, items = set(), []
    for n in sorted(z.namelist()):
        m = re.search(r"WBC-(Benign|Malignant-(Early|Pre|Pro))-\d+", n)
        if not m:
            continue
        data = z.read(n)
        h = hashlib.sha256(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).tobytes()).hexdigest()
        if h in seen:                        # the mirror contains exact copies ("(1)" files)
            continue
        seen.add(h)
        sub = m.group(2) or "Benign"
        items.append({"name": n.split("/")[-1], "label": int(sub != "Benign"), "subtype": sub, "data": data})
    if limit:
        rng = np.random.default_rng(0)
        items = [items[i] for i in sorted(rng.choice(len(items), size=min(limit, len(items)), replace=False))]
    return items


def run_taleqani(limit: int | None, method: str, backend: str = "onnx", model_dir: str = "models",
                 prefix: str = "effb0_main", arm_names: list[str] | None = None, use_filter: bool = False,
                 job_tag: str = "") -> dict:
    arm_names = arm_names or [a[0] for a in ARM_SPECS]
    job = "taleqani_eval" + job_tag + (f"_n{limit}" if limit else "")
    pg = Progress(job, "Smear evaluation on Taleqani ALL" + (f" ({limit} images)" if limit else ""), total=1, stage="loading images")
    model = Model(resolve(model_dir))
    filt = CellTypeFilter(resolve("models")) if use_filter else None
    items = taleqani_items(limit)
    rgbs = [np.asarray(Image.open(io.BytesIO(it["data"])).convert("RGB")) for it in items[:40]]
    stain = json.loads(resolve("models/cnmc_stain.json").read_text())
    if backend == "cuda":
        BACKEND["torch"] = TorchEnsemble(model, prefix)
        # re-check GPU vs ONNX agreement on real smear crops before trusting the fast path
        chk = []
        for rgb in rgbs[:40]:
            r = find_white_cells(rgb)
            chk += [to_cnmc(rgb, c, r.scale, "ruler", stain=stain) for c in r.cells[:2]]
        p_gpu = score_crops(model, chk)
        BACKEND.pop("torch")
        p_cpu = score_crops(model, chk)
        BACKEND["torch"] = TorchEnsemble(model, prefix)
        diff = float(np.abs(p_gpu - p_cpu).max())
        pg.metrics["gpu_vs_onnx_max_diff"] = diff
        assert diff < 1e-3, f"GPU ensemble disagrees with the ONNX artifact by {diff:.2e}"
    from concurrent.futures import ProcessPoolExecutor

    workers = max(2, min(6, (os.cpu_count() or 4) - 2))
    pg.set_stage(f"measuring red cells (scale), {workers} workers", total=len(items))
    per_img = []
    with ProcessPoolExecutor(workers) as ex:
        for e in ex.map(_rbc_one, [it["data"] for it in items], chunksize=16):
            if e:
                per_img.append(e)
            pg.step()
    pooled = float(np.median(per_img))
    pg.metrics["red_cell_diameter_px"] = pooled

    pg.set_stage(f"finding and scoring cells, {len(arm_names)} crop settings, {workers} workers + {backend}", total=len(items))
    arms = [a[0] for a in ARM_SPECS if a[0] in arm_names]
    if filt is not None:
        assert "ruler_x1.0_stain" in arms, "the filter is applied to the stain-normalised ruler crops"
    report_arms = arms + (["ruler_x1.0_stain_filtered"] if filt is not None else [])
    per_arm = {a: [] for a in report_arms}
    per_arm_ood = {a: [] for a in report_arms}
    kept_frac = []
    cell_rows = []
    mean = model.mean[None].astype(np.float32)
    std = model.std[None].astype(np.float32)
    t0 = time.time()
    def bounded(ex, window: int = 24):
        """Results in order, with at most `window` images in flight. Executor.map would queue every job at
        once, and finished crops pile up in memory faster than the GPU drains them (WinError 1450)."""
        from collections import deque

        q = deque()
        for it in items:
            q.append(ex.submit(_crops_one, (it["data"], pooled, stain, method, model.size, arms,
                                            filt.size if filt is not None else None)))
            if len(q) >= window:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()

    with ProcessPoolExecutor(workers) as ex:
        for k, (it, out) in enumerate(zip(items, bounded(ex))):
            sizes = [len(out["crops"][a]) for a in arms]
            allc = np.concatenate([out["crops"][a] for a in arms], axis=0)
            if len(allc):
                xs = ((allc.astype(np.float32).transpose(0, 3, 1, 2) / 255.0) - mean) / std
                L = BACKEND["torch"].logits(xs) if "torch" in BACKEND else model.session.run(None, {"image": xs.astype(np.float32)})[0]
                probs = np.array([model.calibrate(l)[1] for l in L])
                mx = np.abs(L).max(axis=1)
            else:
                probs = mx = np.zeros(0)
            i = 0
            for a, n in zip(arms, sizes):
                per_arm[a].append(probs[i:i + n])
                per_arm_ood[a].append(mx[i:i + n] > model.ood_logit)
                if a == "ruler_x1.0_stain" and filt is not None:
                    k_ = filt.keep(out["filter_crops"])
                    kept_frac.append(k_.mean() if len(k_) else np.nan)
                    per_arm["ruler_x1.0_stain_filtered"].append(probs[i:i + n][k_])
                    per_arm_ood["ruler_x1.0_stain_filtered"].append((mx[i:i + n] > model.ood_logit)[k_])
                i += n
            cell_rows.append({"name": it["name"], "label": it["label"], "subtype": it["subtype"],
                              "cells_found": out["cells_found"], "cells_scored": out["cells_scored"]})
            pg.step(note=it["name"], metrics={"cells_scored_so_far": sum(r["cells_scored"] for r in cell_rows)})
            if k % 200 == 0:
                print(f"{k}/{len(items)} images, {time.time() - t0:.0f}s", flush=True)

    pg.set_stage("computing metrics and bootstrap intervals", total=1)
    meta = pd.DataFrame(cell_rows)
    y = meta.label.to_numpy()
    t_low, t_high = model.t_low, model.t_high
    results = {}
    def evaluate(ps):
        has = np.array([len(p) > 0 for p in ps])
        aggs = {"mean": np.array([p.mean() if len(p) else np.nan for p in ps]),
                "max": np.array([p.max() if len(p) else np.nan for p in ps]),
                "frac_above_band": np.array([(p > t_high).mean() if len(p) else np.nan for p in ps])}
        res = {}
        for agg, sc in aggs.items():
            yy, ss = y[has], sc[has]
            r = {"images_scored": int(has.sum()),
                 "roc_auc": float(roc_auc_score(yy, ss)), "roc_auc_ci95": boot_ci(roc_auc_score, yy, ss),
                 "pr_auc": float(average_precision_score(yy, ss)), "pr_auc_ci95": boot_ci(average_precision_score, yy, ss)}
            if agg == "mean":
                dec = np.where(ss < t_low, 0, np.where(ss > t_high, 1, -1))
                pos = yy == 1
                r["band"] = {"referral_rate": float((dec == -1).mean()),
                             "missed_malignant": float(((dec == 0) & pos).sum() / pos.sum()),
                             "false_alarm_benign": float(((dec == 1) & ~pos).sum() / (~pos).sum())}
                sub = meta[has].subtype.to_numpy()
                r["sensitivity_by_subtype_at_t_high"] = {st: float((ss[sub == st] > t_high).mean()) for st in ("Early", "Pre", "Pro")}
                r["benign_mean_score"] = float(ss[~pos].mean())
            res[agg] = r
        return res

    for arm in report_arms:
        flags = np.concatenate([f for f in per_arm_ood[arm] if len(f)]) if any(len(f) for f in per_arm_ood[arm]) else np.array([])
        guarded = [p[~f] for p, f in zip(per_arm[arm], per_arm_ood[arm])]
        results[arm] = {"as_planned": evaluate(per_arm[arm]),
                        "with_ood_guard": evaluate(guarded),
                        "cells_flagged_out_of_distribution": float(flags.mean()) if len(flags) else None}

    summary = {
        "dataset": "Taleqani Hospital ALL (Aria et al. 2021), HF mirror pritam06/leukemia-cancer-data, 224 x 224",
        "images": len(items), "malignant": int(y.sum()), "benign": int((1 - y).sum()),
        "method": method, "backend": backend, "seconds": time.time() - t0,
        "red_cell_diameter_px": {"pooled": pooled, "per_image_cv": float(np.std(per_img) / np.mean(per_img)),
                                 "scale_to_cnmc": CNMC_RBC_PX / pooled},
        "cells": {"images_with_no_scored_cell": float((meta.cells_scored == 0).mean()),
                  "median_cells_found": float(meta.cells_found.median()), "median_cells_scored": float(meta.cells_scored.median())},
        "model_dir": model_dir, "checkpoint_prefix": prefix, "cell_type_filter": use_filter,
        "filter_kept_fraction_median": float(np.nanmedian(kept_frac)) if kept_frac else None,
        **({"primary_as_planned": results["ruler_x1.0"]["as_planned"]["mean"],
            "primary_note": ("The planned primary arm scores cells in their raw colours. A label-free check made before any "
                             "result was read showed those inputs push member logits to 12,000+ (C-NMC max 14.9), so its "
                             "scores are numerically arbitrary. The stain-normalised arm with the out-of-distribution guard "
                             "is the meaningful result.")} if "ruler_x1.0" in results else {}),
        "main_result": results["ruler_x1.0_stain_filtered" if use_filter else "ruler_x1.0_stain"]["with_ood_guard"]["mean"],
        "all_arms": results,
        "interval_note": "image-level bootstrap; no patient IDs, so intervals are optimistic",
    }
    # per-smear scores (aggregate numbers only; no images) for the paired old-vs-new comparison
    guarded_mean = {a: np.array([float(p[~f].mean()) if len(p[~f]) else np.nan
                                 for p, f in zip(per_arm[a], per_arm_ood[a])]) for a in report_arms}
    np.savez(OUT / f"_scores_taleqani{job_tag}.npz", names=meta.name.to_numpy(dtype=str), labels=y,
             subtypes=meta.subtype.to_numpy(dtype=str), **{f"mean_{a}": v for a, v in guarded_mean.items()})
    pg.step(metrics={"main_roc_auc": summary["main_result"]["roc_auc"]})
    pg.status = "finished"
    pg._write(force=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["bccd", "taleqani"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--method", default="classic", choices=["classic", "cellpose"])
    ap.add_argument("--backend", default="cuda", choices=["cuda", "onnx"], help="cuda: same checkpoints on the GPU (fast); onnx: the deployed file on CPU")
    ap.add_argument("--model-dir", default="models", help="folder with model_meta.json and the ONNX file")
    ap.add_argument("--prefix", default="effb0_main", help="checkpoint prefix of the same five seeds (GPU path)")
    ap.add_argument("--arms", default="all", choices=["all", "stain"], help="stain: only the colour-normalised ruler crops")
    ap.add_argument("--filter", action="store_true", help="also report results after the cell-type filter")
    ap.add_argument("--tag", default="", help="suffix for the output file and job name")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import torch

    backend = args.backend if torch.cuda.is_available() else "onnx"
    arm_names = None if args.arms == "all" else ["ruler_x1.0_stain"]
    s = (run_bccd(args.method, args.model_dir, args.filter) if args.which == "bccd" else
         run_taleqani(args.limit, args.method, backend, args.model_dir, args.prefix, arm_names, args.filter, args.tag))
    tag = f"{args.which}_{args.method}{args.tag}" + (f"_n{args.limit}" if args.limit else "")
    (OUT / f"{tag}.json").write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps({k: v for k, v in s.items() if k != "all_arms"}, indent=2, default=float))


if __name__ == "__main__":
    main()
