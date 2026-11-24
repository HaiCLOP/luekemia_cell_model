"""Three-lab "leukemic blast vs normal lymphocyte" data, and leave-one-lab-out training with DinoBloom features.

Sources (each has BOTH classes, so the model cannot learn "which lab" as a shortcut):
  cnmc  AIIMS New Delhi (C-NMC 2019)          B-ALL lymphoblasts vs normal B-lymphocytes, patient IDs, already cut out
  jhu   Johns Hopkins (DeepAPL, Sidhom 2021)   "Blast, no lineage spec" vs "Lymphocyte", AML/APL patients, CellaVision
  lmu   LMU Munich (AML-Cytomorphology_LMU,    MYO (myeloblast) vs LYT (typical lymphocyte); no patient IDs
        Matek 2019, TCIA, CC BY 3.0)

Every jhu/lmu tile goes through the website's own cell pipeline (red-cell ruler pooled over 300 tiles per source, the
white cell nearest the tile centre cut onto black, colours normalised to C-NMC), so the three labs reach the model in
the same form. Crops are saved as 224 px PNGs for later fine-tuning; DinoBloom-B embeddings are cached.

Evaluation plan, fixed before results:
  * Leave one lab out: train on two labs, test on the third, for each lab in turn.
  * The classifier's C is chosen by generalisation between the two training labs (train on one, validate on the other,
    both ways, mean PR-AUC), never on the held-out lab.
  * Each training lab gets equal total weight; classes balanced within it.
  * Baseline: the C-NMC-only DinoBloom classifier (reports/foundation) on the same held-out cells.
  * Held-out C-NMC uses only its 26 test subjects (as in every earlier C-NMC result). 95% intervals resample patients
    (jhu, cnmc) or cells (lmu, no patient IDs).

Usage
  python -m src.multilab build     # crops + manifest + embeddings for jhu and lmu (C-NMC embeddings are reused)
  python -m src.multilab lolo      # leave-one-lab-out results
Research demo. Not for clinical use.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.progress import Progress
from src.utils import load_config, resolve

CROPS = Path("E:/cache/multilab")
OUT = resolve("reports/multilab")
SOURCES = {
    "jhu": (Path("E:/datasets/deepapl/Data/All"), {"Lymphocyte": 0, "Blast, no lineage spec": 1}),
    "lmu": (Path("E:/datasets/aml_lmu/images"), {"LYT": 0, "MYO": 1}),
}


def _tiles(src: str) -> pd.DataFrame:
    root, classes = SOURCES[src]
    rows = []
    for cls, y in classes.items():
        pat = f"*/Signed slides/{cls}/*.jpg" if src == "jhu" else f"{cls}/*.tiff"
        for f in sorted(root.glob(pat)):
            rows.append({"tile": str(f), "source": src, "cell_type": cls, "label": y,
                         "group": f"jhu_{f.parts[-4]}" if src == "jhu" else f"lmu_{f.stem}"})
    return pd.DataFrame(rows)


def _centre_cell(res, shape):
    cy, cx = shape[0] / 2, shape[1] / 2
    best, dist = None, 1e9
    for c in res.cells:
        x0, y0, x1, y1 = c.box
        d = np.hypot((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy)
        if d < dist:
            best, dist = c, d
    return best if best is not None and dist < 0.25 * min(shape[:2]) else None


def build() -> None:
    from src.foundation import CACHE, embed_uint8, load_dinobloom
    from src.vision.segment import CNMC_RBC_PX, find_white_cells, to_cnmc

    stain = json.loads(resolve("models/cnmc_stain.json").read_text())
    rgb_of = lambda p: np.asarray(Image.open(p).convert("RGB"))
    model, dev = load_dinobloom()
    rows, feats = [], []
    tiles = {s: _tiles(s) for s in SOURCES}
    total = sum(len(t) for t in tiles.values())
    with Progress("multilab_build", "Cutting out every cell from the Johns Hopkins and Munich datasets",
                  total=total, steps=[f"{s}: measure red cells, cut out, describe" for s in SOURCES],
                  explain=("Every lymphocyte and blast image from Johns Hopkins (CellaVision) and LMU Munich is cut out "
                           "exactly the way the website cuts cells from a smear, saved, and described with DinoBloom, so "
                           "the three labs can be trained on together.")) as pg:
        for k, (src, df) in enumerate(tiles.items()):
            pg.set_stage(src, total=len(df), step=k)
            rng = np.random.default_rng(0)
            ds = []
            for p in df.tile.to_numpy()[rng.choice(len(df), 300, replace=False)]:
                try:
                    ds.append(find_white_cells(rgb_of(p)).rbc_diameter_px)
                except OSError:
                    pass
            pooled = float(np.median([d for d in ds if d]))
            scale = CNMC_RBC_PX / pooled
            (CROPS / src).mkdir(parents=True, exist_ok=True)
            buf, meta = [], []

            def flush():
                if buf:
                    feats.append(embed_uint8(model, dev, np.stack(buf)))
                    rows.extend(meta)
                    buf.clear(); meta.clear()

            unreadable = 0
            for r in df.itertuples():
                out = CROPS / src / f"{r.Index:06d}.png"
                lymphoid = None
                if out.exists():                       # resume: a crop saved by an earlier run
                    crop = np.asarray(Image.open(out).convert("RGB"))
                    c = True
                else:
                    try:
                        rgb = rgb_of(r.tile)
                    except OSError:                    # a damaged file in the source dataset
                        unreadable += 1
                        pg.step(note=src)
                        continue
                    c = _centre_cell(find_white_cells(rgb, rbc_override=pooled), rgb.shape)
                    if c is not None:
                        lymphoid = bool(c.lymphoid_like)
                        crop = np.asarray(Image.fromarray(to_cnmc(rgb, c, scale, "ruler", stain=stain)).resize((224, 224), Image.BILINEAR))
                        Image.fromarray(crop).save(out)
                if c is not None:
                    buf.append(crop)
                    meta.append({"crop": str(out), "tile": r.tile, "source": src, "cell_type": r.cell_type, "label": r.label,
                                 "group": r.group, "lymphoid_like": lymphoid, "pooled_rbc_px": pooled})
                    if len(buf) >= 128:
                        flush()
                pg.step(note=src)
            flush()
            found = sum(1 for m in rows if m["source"] == src)
            print(f"{src}: {found}/{len(df)} tiles gave a centre cell, {unreadable} unreadable; pooled red cell {pooled:.1f} px", flush=True)

    man = pd.DataFrame(rows)
    x = np.concatenate(feats)
    # C-NMC: reuse the cached embeddings (same model, same 224 px resize of the original cut-out images)
    c = np.load(CACHE, allow_pickle=False)
    cn = pd.DataFrame({"crop": c["image_path"], "tile": c["image_path"], "source": "cnmc",
                       "cell_type": np.where(c["label"] == 1, "B-ALL blast", "normal B-lymphocyte"), "label": c["label"].astype(int),
                       "group": ["cnmc_" + str(g) for g in c["group"]], "lymphoid_like": True, "pooled_rbc_px": np.nan,
                       "cnmc_split": c["split"]})
    man = pd.concat([cn, man], ignore_index=True)
    x = np.concatenate([c["x"], x])
    OUT.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT / "manifest.csv", index=False)
    np.save(CROPS / "dinobloom_b.npy", x.astype(np.float32))
    print(man.groupby(["source", "cell_type"]).size().to_string())


# ----------------------------------------------------------------------------------- leave one lab out
def _weights(src: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Equal total weight per (lab, class) cell group: no lab and no class dominates."""
    w = np.zeros(len(y))
    for s in np.unique(src):
        for c in (0, 1):
            m = (src == s) & (y == c)
            if m.any():
                w[m] = 1.0 / m.sum()
    return w * len(y) / w.sum()


def _fit(x, y, w, c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
    clf.fit(x, y, logisticregression__sample_weight=w)
    return clf


def _boot(y, p, g, iters=1000, seed=0):
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ug} if len(ug) < len(g) else None
    roc, pr = [], []
    for _ in range(iters):
        i = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]) if idx else rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            roc.append(roc_auc_score(y[i], p[i])); pr.append(average_precision_score(y[i], p[i]))
    q = lambda v: [float(a) for a in np.percentile(v, [2.5, 97.5])]
    return {"roc_auc": float(roc_auc_score(y, p)), "roc_auc_ci95": q(roc),
            "pr_auc": float(average_precision_score(y, p)), "pr_auc_ci95": q(pr),
            "normal_called_leukemic_at_0.5": float((p[y == 0] > 0.5).mean()),
            "blasts_caught_at_0.5": float((p[y == 1] > 0.5).mean()),
            "cells": {"normal": int((y == 0).sum()), "blast": int((y == 1).sum())}}


def lolo(variant: str = "b") -> dict:
    import joblib
    from sklearn.metrics import average_precision_score

    man = pd.read_csv(OUT / "manifest.csv", low_memory=False)
    x = np.load(CROPS / f"dinobloom_{variant}.npy")
    y, src, g = man.label.to_numpy(int), man.source.to_numpy(str), man.group.to_numpy(str)
    split = man.get("cnmc_split", pd.Series([None] * len(man))).fillna("").to_numpy(str)
    # C-NMC: train/val subjects may train; only test subjects are ever tested on
    trainable = (src != "cnmc") | (split != "test")
    testable = lambda lab: (src == lab) & ((src != "cnmc") | (split == "test"))
    cnmc_only = joblib.load(resolve("reports/foundation") / "dinobloom_b_logreg.joblib")
    grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    res = {"plan": __doc__.split("Usage")[0].strip(), "labs": {}}
    for held in ("cnmc", "jhu", "lmu"):
        tr_labs = [s for s in ("cnmc", "jhu", "lmu") if s != held]
        inner = {}
        for c in grid:
            sc = []
            for a, b in ((tr_labs[0], tr_labs[1]), (tr_labs[1], tr_labs[0])):
                ma, mb = trainable & (src == a), trainable & (src == b)
                clf = _fit(x[ma], y[ma], _weights(src[ma], y[ma]), c)
                sc.append(average_precision_score(y[mb], clf.predict_proba(x[mb])[:, 1]))
            inner[c] = float(np.mean(sc))
        best = max(inner, key=inner.get)
        m = trainable & (src != held)
        clf = _fit(x[m], y[m], _weights(src[m], y[m]), best)
        t = testable(held)
        res["labs"][held] = {"trained_on": tr_labs, "C_by_cross_lab_validation": inner, "chosen_C": best,
                             "multilab": _boot(y[t], clf.predict_proba(x[t])[:, 1], g[t]),
                             "cnmc_only_baseline": None if held == "cnmc" or variant != "b"
                             else _boot(y[t], cnmc_only.predict_proba(x[t])[:, 1], g[t])}
        print(held, json.dumps(res["labs"][held]["multilab"], default=float), flush=True)
    name = "lolo_results.json" if variant == "b" else f"lolo_results_{variant}.json"
    (OUT / name).write_text(json.dumps(res, indent=2, default=float))
    return res


# ----------------------------------------------------------------------------------- cell-type filter on DinoBloom features
# keep = cells the blast/normal classifier can judge; drop = other white cells. Ambiguous types are left out entirely:
# promyelocytes (the leukemic cells in APL), promonocytes/monoblasts, plasma cells, variant/atypical lymphocytes.
OTHERS = {
    "jhu": {"Segmented neutrophils": 0, "Band neutrophils": 0, "Monocyte": 0, "Eosinophils": 0, "Basophil": 0,
            "Erythroblast": 0, "Smudge cells": 0, "Myelocyte": 0, "Metamyelocyte": 0},
    "lmu": {"NGS": 0, "NGB": 0, "MON": 0, "EOS": 0, "BAS": 0, "EBO": 0, "KSC": 0, "MMZ": 0, "MYB": 0},
    "pbc": {"neutrophil": 0, "eosinophil": 0, "ig": 0, "erythroblast": 0, "monocyte": 0, "basophil": 0, "lymphocyte": 1},
}
CAP = 500   # at most this many tiles per (lab, cell type)


def build_others() -> None:
    """Other white-cell types from Hopkins, Munich and Barcelona (PBC, Acevedo 2020, CC BY 4.0), cut out the website's
    way and described with DinoBloom-B, for a cell-type filter that has seen more than one lab."""
    import io

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from src.foundation import embed_uint8, load_dinobloom
    from src.vision.cell_type import PBC_FILE, PBC_NAMES, PBC_REPO
    from src.vision.segment import CNMC_RBC_PX, find_white_cells, to_cnmc

    stain = json.loads(resolve("models/cnmc_stain.json").read_text())
    rng = np.random.default_rng(0)
    tiles = []
    for src in ("jhu", "lmu"):
        root = SOURCES[src][0]
        for cls, k in OTHERS[src].items():
            fs = sorted(root.glob(f"*/Signed slides/{cls}/*.jpg" if src == "jhu" else f"{cls}/*.tiff"))
            for i in rng.permutation(len(fs))[:CAP]:
                f = fs[i]
                tiles.append({"tile": str(f), "source": src, "cell_type": cls, "keep": k,
                              "group": f"jhu_{f.parts[-4]}" if src == "jhu" else f"lmu_{f.stem}",
                              "read": (lambda f=f: Image.open(f))})
    t = pq.read_table(hf_hub_download(PBC_REPO, PBC_FILE, repo_type="dataset")).to_pandas()
    for lab, name in PBC_NAMES.items():
        if name not in OTHERS["pbc"]:
            continue
        for i in rng.permutation(np.flatnonzero(t.label.to_numpy() == lab))[:CAP]:
            b = t.image.iloc[i]["bytes"]
            tiles.append({"tile": f"pbc:{i}", "source": "pbc", "cell_type": name, "keep": OTHERS["pbc"][name],
                          "group": f"pbc_{i}", "read": (lambda b=b: Image.open(io.BytesIO(b)))})
    pooled = {"jhu": 69.4, "lmu": 90.9,     # from `build` (300-tile medians, see reports/logs/multilab_build.log)
              "pbc": json.loads((Path("E:/datasets/pbc_cells") / "prep.json").read_text())["rbc_pooled_px"]}
    model, dev = load_dinobloom()
    (CROPS / "others").mkdir(parents=True, exist_ok=True)
    rows, feats, buf = [], [], []
    with Progress("multilab_others", "Cutting out the other white-cell types for the new cell filter", total=len(tiles),
                  explain=("Neutrophils, monocytes, eosinophils and other cells from Johns Hopkins, Munich and Barcelona are "
                           "cut out the website's way, so a new filter can learn which cells to skip in every lab."),
                  steps=["Cut out and describe"]) as pg:
        pg.set_stage("cutting", total=len(tiles), step=0)
        for n, r in enumerate(tiles):
            try:
                rgb = np.asarray(r["read"]().convert("RGB"))
            except OSError:
                pg.step()
                continue
            c = _centre_cell(find_white_cells(rgb, rbc_override=pooled[r["source"]]), rgb.shape)
            if c is not None:
                crop = np.asarray(Image.fromarray(to_cnmc(rgb, c, CNMC_RBC_PX / pooled[r["source"]], "ruler", stain=stain))
                                  .resize((224, 224), Image.BILINEAR))
                out = CROPS / "others" / f"{n:06d}.png"
                Image.fromarray(crop).save(out)
                buf.append(crop)
                rows.append({k: v for k, v in r.items() if k != "read"} | {"crop": str(out), "lymphoid_like": bool(c.lymphoid_like)})
                if len(buf) >= 128:
                    feats.append(embed_uint8(model, dev, np.stack(buf)))
                    buf.clear()
            pg.step(note=r["source"])
        if buf:
            feats.append(embed_uint8(model, dev, np.stack(buf)))
    pd.DataFrame(rows).to_csv(OUT / "manifest_others.csv", index=False)
    np.save(CROPS / "dinobloom_b_others.npy", np.concatenate(feats).astype(np.float32))
    print(pd.DataFrame(rows).groupby(["source", "cell_type"]).size().to_string())


def _filter_data(variant: str = "b"):
    """keep/drop table: every classifier cell (keep) + the other white cells. C-NMC test subjects are never used."""
    man = pd.read_csv(OUT / "manifest.csv", low_memory=False)
    x = np.load(CROPS / f"dinobloom_{variant}.npy")
    split = man.get("cnmc_split", pd.Series([None] * len(man))).fillna("").to_numpy(str)
    ok = split != "test"
    a = man[ok].assign(keep=1)[["source", "cell_type", "group", "keep", "crop"]]
    o = pd.read_csv(OUT / "manifest_others.csv")
    xo = np.load(CROPS / f"dinobloom_{variant}_others.npy")
    return pd.concat([a, o[["source", "cell_type", "group", "keep", "crop"]]], ignore_index=True), np.concatenate([x[ok], xo])


def _keep_threshold(p, keep, src, target=0.98):
    """Highest threshold that still keeps >= target of the keep cells in EVERY training lab (blasts included)."""
    return float(min(np.quantile(p[(keep == 1) & (src == s)], 1 - target) for s in np.unique(src[keep == 1])))


def filter_lolo(variant: str = "b") -> dict:
    """Leave one lab out for the filter: per-cell-type keep rates on the unseen lab, next to the deployed ONNX filter on
    the same crops (resized 224 -> 160 px for it, close to but not exactly its own 450 -> 160 resize)."""
    from src.smear_eval import CellTypeFilter

    d, x = _filter_data(variant)
    src, keep = d.source.to_numpy(str), d.keep.to_numpy(int)
    old = CellTypeFilter(resolve("models"))
    res = {}
    for held in ("jhu", "lmu", "pbc"):
        m = src != held
        clf = _fit(x[m], keep[m], _weights(src[m], keep[m]), 0.01)
        thr = _keep_threshold(clf.predict_proba(x[m])[:, 1], keep[m], src[m])
        t = src == held
        dt = d[t].assign(new=clf.predict_proba(x[t])[:, 1] >= thr)
        crops = dt.crop.tolist()
        olds = []
        for i in range(0, len(crops), 256):
            imgs = np.stack([np.asarray(Image.open(c).convert("RGB").resize((old.size, old.size), Image.BILINEAR))
                             for c in crops[i:i + 256]])
            olds.append(old.keep(imgs))
        dt = dt.assign(old=np.concatenate(olds))
        res[held] = {"threshold": thr, "kept_by_type": {ct: {"n": int(len(g)), "new": float(g.new.mean()), "old": float(g.old.mean())}
                                                        for ct, g in dt.groupby("cell_type")}}
        print(held, json.dumps(res[held]["kept_by_type"]), flush=True)
    (OUT / f"filter_lolo_{variant}.json").write_text(json.dumps(res, indent=2))
    return res


def filter_final(variant: str = "b") -> None:
    import joblib

    d, x = _filter_data(variant)
    src, keep = d.source.to_numpy(str), d.keep.to_numpy(int)
    clf = _fit(x, keep, _weights(src, keep), 0.01)
    thr = _keep_threshold(clf.predict_proba(x)[:, 1], keep, src)
    joblib.dump({"clf": clf, "threshold": thr}, OUT / f"dinobloom_{variant}_filter.joblib")
    print("filter threshold", thr)


# ----------------------------------------------------------------------------------- DinoBloom-L features
def embed_large() -> None:
    """DinoBloom-L embeddings for every saved crop (and C-NMC's original cut-outs), in the same order as the -B files."""
    import torch

    from src.foundation import load_dinobloom

    model, dev = load_dinobloom("l")
    root = Path(load_config()["paths"]["data_root"])
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

    def run(paths, name):
        out = []
        with Progress(f"embed_l_{name}", f"Describing every {name} cell with the larger DinoBloom-L", total=len(paths),
                      steps=["DinoBloom-L"]) as pg:
            pg.set_stage("embedding", total=len(paths), step=0)
            for i in range(0, len(paths), 32):
                imgs = np.stack([np.asarray(Image.open(p if Path(p).is_absolute() else root / p).convert("RGB")
                                            .resize((224, 224), Image.BILINEAR)) for p in paths[i:i + 32]])
                xb = ((imgs.astype(np.float32).transpose(0, 3, 1, 2) / 255.0) - mean) / std
                with torch.no_grad():
                    out.append(model(torch.from_numpy(xb).to(dev)).float().cpu().numpy())
                pg.step(len(imgs))
        return np.concatenate(out).astype(np.float32)

    man = pd.read_csv(OUT / "manifest.csv", low_memory=False)
    np.save(CROPS / "dinobloom_l.npy", run(man.crop.tolist(), "main"))
    oth = pd.read_csv(OUT / "manifest_others.csv")
    np.save(CROPS / "dinobloom_l_others.npy", run(oth.crop.tolist(), "others"))


def fewshot(npz: Path, arm: str = "ruler_x1.0_stain_filtered", ks=(4, 10, 20), draws: int = 1000) -> dict:
    """A new lab labels k smears (half benign, half ALL); the cut-off is set on those (midpoint that maximises balanced
    accuracy) and judged on the lab's remaining smears. Repeated over random draws."""
    d = np.load(npz)
    s, y = d[f"mean_{arm}"], d["labels"]
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = {}
    for k in ks:
        sens, spec = [], []
        for _ in range(draws):
            pick = np.concatenate([rng.choice(pos, k // 2, replace=False), rng.choice(neg, k // 2, replace=False)])
            cand = np.unique(s[pick])
            cand = (cand[:-1] + cand[1:]) / 2 if len(cand) > 1 else cand
            ba = [((s[pick] > t)[y[pick] == 1].mean() + (s[pick] <= t)[y[pick] == 0].mean()) / 2 for t in cand]
            t = cand[int(np.argmax(ba))]
            rest = np.setdiff1d(np.arange(len(y)), pick)
            sens.append(float((s[rest][y[rest] == 1] > t).mean()))
            spec.append(float((s[rest][y[rest] == 0] <= t).mean()))
        q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out[f"k={k}"] = {"sensitivity": float(np.mean(sens)), "sensitivity_95": q(sens),
                         "specificity": float(np.mean(spec)), "specificity_95": q(spec)}
    return out


def final(variant: str = "b", use_filter: bool = False) -> dict:
    """One classifier on all three labs (C-NMC test subjects still held out), C chosen by the mean held-out-lab PR-AUC
    of the three leave-one-lab-out folds. Then the outside tests the earlier models had: Taleqani ALL smears (a fourth
    lab, real ALL, paired against the deployed model) and healthy BCCD smears. With use_filter, the DinoBloom cell
    filter (filter_final) replaces the shape rule and the ONNX filter. Then the few-shot cut-off study."""
    import joblib
    from sklearn.metrics import average_precision_score

    from src.foundation import outside

    man = pd.read_csv(OUT / "manifest.csv", low_memory=False)
    x = np.load(CROPS / f"dinobloom_{variant}.npy")
    y, src = man.label.to_numpy(int), man.source.to_numpy(str)
    split = man.get("cnmc_split", pd.Series([None] * len(man))).fillna("").to_numpy(str)
    trainable = (src != "cnmc") | (split != "test")
    grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    score = {}
    for c in grid:
        s = []
        for held in ("cnmc", "jhu", "lmu"):
            m = trainable & (src != held)
            t = (src == held) & ((src != "cnmc") | (split == "test"))
            s.append(average_precision_score(y[t], _fit(x[m], y[m], _weights(src[m], y[m]), c).predict_proba(x[t])[:, 1]))
        score[c] = float(np.mean(s))
    best = max(score, key=score.get)
    clf = _fit(x[trainable], y[trainable], _weights(src[trainable], y[trainable]), best)
    tag = variant + ("_dbfilter" if use_filter else "")
    path = OUT / f"dinobloom_{variant}_multilab.joblib"
    joblib.dump(clf, path)
    (OUT / f"final_choice_{variant}.json").write_text(json.dumps({"C_mean_lolo_pr_auc": score, "chosen_C": best}, indent=2))
    print("chosen C", best, score, flush=True)
    res_path = OUT / ("outside_results.json" if tag == "b" else f"outside_results_{tag}.json")
    res = outside(path, res_path, job=f"multilab_outside_{tag}",
                  emb_filter=OUT / f"dinobloom_{variant}_filter.joblib" if use_filter else None, variant=variant)
    res["fewshot_cutoff"] = {arm: fewshot(res_path.with_suffix(".npz"), arm)
                             for arm in ("ruler_x1.0_stain_filtered", "ruler_x1.0_stain")}
    res_path.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res["fewshot_cutoff"], indent=1))
    return res


# ----------------------------------------------------------------------------------- website export
def export_web() -> dict:
    """One ONNX graph for the website: image -> DinoBloom-B -> two linear heads, `logit` (three-lab blast vs normal
    classifier, C as chosen by `final`) and `keep_logit` (the DinoBloom cell filter). 8-bit weight-only, checked against
    the fp32 heads. Calibration uses OUT-OF-LAB predictions only: each lab is scored by a classifier fitted on the other
    two (C-NMC test subjects never used); temperature and the refer band (leukemic miss <= 2%) are fitted on those."""
    import hashlib
    import os

    import joblib
    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    from src.calibrate import apply_temperature, choose_band, fit_temperature
    from src.foundation import MEAN, STD, load_dinobloom

    man = pd.read_csv(OUT / "manifest.csv", low_memory=False)
    x = np.load(CROPS / "dinobloom_b.npy")
    y, src = man.label.to_numpy(int), man.source.to_numpy(str)
    split = man.get("cnmc_split", pd.Series([None] * len(man))).fillna("").to_numpy(str)
    trainable = (src != "cnmc") | (split != "test")
    c_best = json.loads((OUT / "final_choice_b.json").read_text())["chosen_C"]
    z_ool, y_ool = [], []
    for held in ("cnmc", "jhu", "lmu"):
        m, t = trainable & (src != held), trainable & (src == held)
        z_ool.append(_fit(x[m], y[m], _weights(src[m], y[m]), c_best).decision_function(x[t]))
        y_ool.append(y[t])
    z_ool, y_ool = np.concatenate(z_ool), np.concatenate(y_ool)
    temp = fit_temperature(z_ool, y_ool)
    band = choose_band(apply_temperature(z_ool, temp), y_ool)

    clf = joblib.load(OUT / "dinobloom_b_multilab.joblib")
    filt = joblib.load(OUT / "dinobloom_b_filter.joblib")

    def fold(pipe):
        sc, lr = pipe[0], pipe[-1]
        return lr.coef_[0] / sc.scale_, float(lr.intercept_[0] - (lr.coef_[0] * sc.mean_ / sc.scale_).sum())

    (wc, bc), (wf, bf) = fold(clf), fold(filt["clf"])
    backbone, _ = load_dinobloom()
    backbone = backbone.cpu().float().eval()

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.heads = torch.nn.Linear(len(wc), 2)
            with torch.no_grad():
                self.heads.weight.copy_(torch.from_numpy(np.stack([wc, wf]).astype(np.float32)))
                self.heads.bias.copy_(torch.tensor([bc, bf], dtype=torch.float32))

        def forward(self, image):
            h = self.heads(self.backbone(image))
            return h[:, 0], h[:, 1]

    fp32 = Path(os.environ.get("TEMP", "E:/cache")) / "MH51401102_fp32.onnx"
    final_path = resolve("models") / "MH51401102.onnx"
    torch.onnx.export(Net().eval(), torch.zeros(1, 3, 224, 224), str(fp32), input_names=["image"],
                      output_names=["logit", "keep_logit"],
                      dynamic_axes={"image": {0: "n"}, "logit": {0: "n"}, "keep_logit": {0: "n"}}, opset_version=17, dynamo=False)
    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=8, block_size=32, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(str(final_path), use_external_data_format=False)
    fp32.unlink(missing_ok=True)

    # check: 8-bit graph vs the fp32 sklearn heads on the cached embeddings, 300 cells across the three labs
    rng = np.random.default_rng(0)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(trainable & (src == s)), 100, replace=False)
                                  for s in ("cnmc", "jhu", "lmu")]))
    root = Path(load_config()["paths"]["data_root"])
    imgs = np.stack([np.asarray(Image.open(p if Path(p).is_absolute() else root / p).convert("RGB")
                                .resize((224, 224), Image.BILINEAR)) for p in man.crop.to_numpy()[idx]])
    xin = (((imgs.astype(np.float32).transpose(0, 3, 1, 2) / 255.0) - MEAN) / STD).astype(np.float32)
    sess = ort.InferenceSession(str(final_path), providers=["CPUExecutionProvider"])
    outs = [sess.run(None, {"image": xin[i:i + 25]}) for i in range(0, len(xin), 25)]
    zq, kq = np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])
    zr, kr = clf.decision_function(x[idx]), filt["clf"].decision_function(x[idx])
    thr = float(filt["threshold"])
    thr_logit = float(np.log(thr / (1 - thr)))
    pq_, pr_ = apply_temperature(zq, temp), apply_temperature(zr, temp)
    check = {"weights": "8-bit weight-only (MatMulNBits, block 32)", "cells": int(len(idx)),
             "mean_abs_prob_diff": float(np.abs(pq_ - pr_).mean()), "max_abs_prob_diff": float(np.abs(pq_ - pr_).max()),
             "filter_agreement": float(((kq >= thr_logit) == (kr >= thr_logit)).mean())}
    lolo_res = json.loads((OUT / "lolo_results.json").read_text())["labs"]
    tal = json.loads((OUT / "outside_results_b_dbfilter.json").read_text())
    meta = {"name": "MH51401102", "former_name": "DinoBloom-B-3lab",
            "architecture": "DinoBloom-B (frozen DINOv2 ViT-B/14, MarrLab) + logistic regression on three labs + DinoBloom cell filter",
            "task": "leukemic blast vs normal lymphocyte",
            "training_labs": ["C-NMC 2019 (AIIMS New Delhi)", "DeepAPL (Johns Hopkins)", "AML-Cytomorphology_LMU (LMU Munich)"],
            "input": {"size": 224, "mean": MEAN.ravel().tolist(), "std": STD.ravel().tolist(), "resize": "bilinear, antialiased"},
            "outputs": {"logit": "blast vs normal", "keep_logit": "cell filter (keep = lymphocyte or blast)"},
            "temperature": temp, "t_low": band["t_low"], "t_high": band["t_high"],
            "band_target": "leukemic miss rate <= 2% on out-of-lab predictions (each lab scored by a model fitted on the other two)",
            "band_out_of_lab": band,
            "filter_threshold_logit": thr_logit, "filter_threshold_prob": thr,
            "ood_max_abs_logit": round(1.5 * float(np.abs(clf.decision_function(x[trainable])).max()), 1),
            "quant_check": check,
            "metrics": {"leave_one_lab_out_roc_auc": {k: v["multilab"]["roc_auc"] for k, v in lolo_res.items()},
                        "taleqani_smear_paired_vs_retired_MH51401122": tal["paired_vs_deployed_model"]["main_arm_stain_filtered"],
                        "taleqani_smear_roc_auc_all_scored": tal["taleqani"]["ruler_x1.0_stain_filtered"]["roc_auc"],
                        "bccd_healthy_called_leukemic": tal["bccd_healthy"]["stain"]},
            "onnx_file": final_path.name, "onnx_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
            "onnx_mb": round(final_path.stat().st_size / 2**20, 1),
            "weights_licence": "DinoBloom weights: Apache-2.0 (Koch et al., MICCAI 2024)",
            "disclaimer": "Research demo. Not for clinical use."}
    old = resolve("models") / "model_meta.json"     # keep results recorded after export (e.g. ALL-IDB2)
    if old.exists():
        prev = json.loads(old.read_text()).get("metrics", {})
        meta["metrics"] = {**{k: v for k, v in prev.items() if k not in meta["metrics"]}, **meta["metrics"]}
    meta["exported"] = datetime.now(timezone.utc).isoformat()
    for p in (old, resolve("web/src/data/model_meta.json")):
        p.write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps({k: meta[k] for k in ("temperature", "t_low", "t_high", "filter_threshold_logit", "ood_max_abs_logit",
                                            "quant_check", "onnx_mb")}, indent=1, default=float))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "lolo", "final", "others", "filter_lolo", "filter_final", "embed_large",
                                     "export_web"])
    ap.add_argument("--variant", default="b", choices=["b", "l"])
    ap.add_argument("--dbfilter", action="store_true", help="final: use the DinoBloom cell filter")
    a = ap.parse_args()
    if a.what == "final":
        final(a.variant, a.dbfilter)
    else:
        fn = {"build": build, "lolo": lolo, "others": build_others, "filter_lolo": filter_lolo,
              "filter_final": filter_final, "embed_large": embed_large, "export_web": export_web}[a.what]
        fn(a.variant) if a.what in ("lolo", "filter_lolo", "filter_final") else fn()
