"""Outside test on Johns Hopkins CellaVision cells (DeepAPL dataset, Dr Eugene Shenderov's group).

Data: Sidhom, ..., Shenderov. Deep learning for diagnosis of acute promyelocytic leukemia via recognition of genomically
imprinted morphologic features. npj Precision Oncology 5:38 (2021), doi:10.1038/s41698-021-00179-y. Images CC BY 4.0
(figshare 10.6084/m9.figshare.14294675; also github.com/sidhomj/DeepAPL). 106 AML/APL patients, Wright-stained
peripheral blood, CellaVision DM100, every cell labelled by a pathologist ("Signed slides").

What this tests (fixed before any result was read):
  * Cells labelled "Lymphocyte" should be read as normal; cells labelled "Blast, no lineage spec" as leukemic. These are
    myeloid blasts from AML patients, not the B-lymphoblasts the models were trained on, so blast recall is a harder,
    different question. The clean measurement is the lymphocytes: a new lab's normal lymphocytes.
  * Same crop pipeline as the smear tests: the CellaVision tile's own red cells give the scale (pooled over 300 tiles,
    one microscope), the white cell nearest the tile centre is cut onto black, colours normalised to C-NMC.
  * Arms: "all" (every centre cell found) and "filtered" (also passes the shape rule and the learned cell-type filter,
    as on the website). Both models, the exact ONNX files the website uses. Threshold 0.5 and each model's own band.
  * Ensemble vs DinoBloom: paired ROC-AUC difference, bootstrap resampling whole patients.

Nothing here is used for training or model choice. Research demo. Not for clinical use.

Usage: python -m src.deepapl_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.progress import Progress
from src.utils import resolve

ROOT = Path("E:/datasets/deepapl/Data")
CLASSES = {"Lymphocyte": 0, "Blast, no lineage spec": 1}
OUT = resolve("reports/deepapl")


def tiles() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "master.csv")
    rows = []
    for cls, y in CLASSES.items():
        for f in sorted((ROOT / "All").glob(f"*/Signed slides/{cls}/*.jpg")):
            rows.append({"path": str(f), "patient": f.parts[-4], "cell_type": cls, "label": y})
    df = pd.DataFrame(rows).merge(meta[["Patient_ID", "Diagnosis", "Cohort"]], left_on="patient", right_on="Patient_ID")
    return df.drop(columns="Patient_ID")


def _centre_cell(res, shape):
    cy, cx = shape[0] / 2, shape[1] / 2
    best, dist = None, 1e9
    for c in res.cells:
        x0, y0, x1, y1 = c.box
        d = np.hypot((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy)
        if d < dist:
            best, dist = c, d
    return best if best is not None and dist < 0.25 * min(shape[:2]) else None


def _cluster_ci(y, a, b, g, iters=2000, seed=0):
    """ROC-AUC of a, of b, and b - a, with 95% intervals from resampling whole patients."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    pats = np.unique(g)
    idx = {p: np.flatnonzero(g == p) for p in pats}
    ra, rb, rd = [], [], []
    for _ in range(iters):
        i = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
        if len(np.unique(y[i])) < 2:
            continue
        sa, sb = roc_auc_score(y[i], a[i]), roc_auc_score(y[i], b[i])
        ra.append(sa); rb.append(sb); rd.append(sb - sa)
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return {"ensemble_roc_auc": float(roc_auc_score(y, a)), "ensemble_ci95": q(ra),
            "dinobloom_roc_auc": float(roc_auc_score(y, b)), "dinobloom_ci95": q(rb),
            "difference_dinobloom_minus_ensemble": float(roc_auc_score(y, b) - roc_auc_score(y, a)), "difference_ci95": q(rd)}


def _calls(p, y, t_low, t_high):
    out = {}
    for name, cls in (("lymphocytes", 0), ("blasts", 1)):
        q = p[y == cls]
        out[name] = {"cells": int(len(q)), "called_leukemic_at_0.5": float((q > 0.5).mean()),
                     "band_normal": float((q < t_low).mean()), "band_refer": float(((q >= t_low) & (q <= t_high)).mean()),
                     "band_leukemic": float((q > t_high).mean())}
    return out


def main() -> dict:
    import cv2

    from api.inference import Model
    from src.smear_eval import CellTypeFilter
    from src.vision.segment import CNMC_RBC_PX, find_white_cells, to_cnmc

    models = resolve("models")
    # the retired ensemble MH51401122 is re-created with `python -m src.export` (models/retired_effb0)
    ens, dino, filt = Model(models / "retired_effb0"), Model(models), CellTypeFilter(models)
    stain = json.loads((models / "cnmc_stain.json").read_text())
    df = tiles()
    rng = np.random.default_rng(0)

    def rgb_of(p):
        return cv2.imread(p)[..., ::-1].copy()

    with Progress("deepapl_eval", "Testing both models on Johns Hopkins CellaVision cells",
                  total=len(df), steps=["Measure red cells", "Cut out and score every cell"],
                  explain=("Dr Shenderov's dataset: 3,112 lymphocytes and 2,948 blasts labelled by pathologists at Johns "
                           "Hopkins. Each cell is cut out exactly as on the website and read by both models. Normal "
                           "lymphocytes should come out normal.")) as pg:
        pg.set_stage("red cells", total=300, step=0)
        ds = []
        for p in df.path.to_numpy()[rng.choice(len(df), 300, replace=False)]:
            d = find_white_cells(rgb_of(p)).rbc_diameter_px
            if d:
                ds.append(d)
            pg.step()
        pooled = float(np.median(ds))
        scale = CNMC_RBC_PX / pooled

        pg.set_stage("scoring", total=len(df), step=1)
        rec = []
        batch, meta_b = [], []

        def flush():
            if not batch:
                return
            c224 = np.stack([np.asarray(Image.fromarray(c).resize((224, 224), Image.BILINEAR)) for c in batch])
            cf = np.stack([np.asarray(Image.fromarray(c).resize((filt.size, filt.size), Image.BILINEAR)) for c in batch])
            x = ((c224.astype(np.float32).transpose(0, 3, 1, 2) / 255.0) - ens.mean[None]) / ens.std[None]
            x = x.astype(np.float32)
            le, ld = ens.cell_logits(x), dino.cell_logits(x)
            keep = filt.keep(cf)
            for m, a, b, k in zip(meta_b, le, ld, keep):
                m.update(ens_prob=ens.calibrate(a)[1], ens_ood=bool(np.abs(a).max() > ens.ood_logit),
                         dino_prob=dino.calibrate(b)[1], dino_ood=bool(np.abs(b).max() > dino.ood_logit),
                         filter_keep=bool(k))
                rec.append(m)
            batch.clear(); meta_b.clear()

        for r in df.itertuples():
            rgb = rgb_of(r.path)
            c = _centre_cell(find_white_cells(rgb, rbc_override=pooled), rgb.shape)
            m = {"path": Path(r.path).relative_to(ROOT).as_posix(), "patient": r.patient, "cell_type": r.cell_type,
                 "label": r.label, "diagnosis": r.Diagnosis, "found": c is not None}
            if c is None:
                rec.append(m)
            else:
                m["lymphoid_like"] = bool(c.lymphoid_like)
                batch.append(to_cnmc(rgb, c, scale, "ruler", stain=stain))
                meta_b.append(m)
                if len(batch) >= 32:
                    flush()
            pg.step(note=r.patient)
        flush()

    cells = pd.DataFrame(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT / "cells.csv", index=False)

    found = cells[cells.found]
    res = {"data": "DeepAPL (Sidhom et al., npj Precis Oncol 2021), Johns Hopkins, CellaVision DM100; CC BY 4.0",
           "pooled_red_cell_px": pooled, "scale_to_cnmc": scale,
           "tiles": {k: int(v) for k, v in cells.groupby("cell_type").size().items()},
           "centre_cell_found": {k: float(v) for k, v in cells.groupby("cell_type").found.mean().items()},
           "patients": int(cells.patient.nunique()), "arms": {}}
    arms = {"all": found, "filtered": found[found.lymphoid_like & found.filter_keep]}
    for name, d in arms.items():
        y, g = d.label.to_numpy(), d.patient.to_numpy()
        a, b = d.ens_prob.to_numpy(float), d.dino_prob.to_numpy(float)
        res["arms"][name] = {
            "cells": {k: int(v) for k, v in d.groupby("cell_type").size().items()},
            "auc_blast_vs_lymphocyte": _cluster_ci(y, a, b, g) if len(np.unique(y)) == 2 else None,
            "ensemble": {**_calls(a, y, ens.t_low, ens.t_high), "refused_ood": float(d.ens_ood.mean())},
            "dinobloom": {**_calls(b, y, dino.t_low, dino.t_high), "refused_ood": float(d.dino_ood.mean())},
        }
    kept = found.groupby("cell_type")[["lymphoid_like", "filter_keep"]].mean()
    res["kept_by_shape_rule"] = kept.lymphoid_like.to_dict()
    res["kept_by_learned_filter"] = kept.filter_keep.to_dict()
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    main()
