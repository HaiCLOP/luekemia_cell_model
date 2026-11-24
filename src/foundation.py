"""DinoBloom foundation-model features + a simple classifier, as suggested by Dr. Carsten Marr (Helmholtz Munich).

DinoBloom (Koch et al., MICCAI 2024; MarrLab/DinoBloom on Hugging Face, Apache-2.0) is a DINOv2 vision transformer
trained on 13 public blood and bone-marrow cell datasets. It is used frozen: every cell becomes one embedding, and only
a logistic-regression classifier is trained on top, which should depend far less on one laboratory's look than a
fine-tuned network.

Plan, fixed before any result was read:
  * DinoBloom-B (768-dim CLS embedding), official loading recipe, 224 px, ImageNet normalisation.
  * Classifier: standardised embeddings -> L2 logistic regression, class-weighted. The only setting, C, is chosen on
    the validation subjects (PR-AUC). Nothing is tuned on test or outside data.
  * C-NMC test: evaluated once, at cell level and at subject level (mean cell score per subject: ALL patients vs
    healthy people, the patient-level framing Dr. Marr recommended).
  * Outside: the same Taleqani smears and BCCD healthy smears, same cell finder, cell-type filter and crop settings as
    the earlier tests. Main arm: C-NMC colour normalisation + filter (same as the deployed model's main result), with a
    paired comparison against the deployed model on the same smears. Secondary arm: original colours + filter.

Usage
  python -m src.foundation embed      # DinoBloom features for all C-NMC cells (cached)
  python -m src.foundation fit        # choose C on validation, fit, evaluate on test once
  python -m src.foundation export     # one int8 ONNX graph (image -> logit) + calibration, for the web app
  python -m src.foundation outside    # Taleqani + BCCD
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.progress import Progress
from src.utils import load_config, resolve

VARIANT = "b"
DIMS = {"s": ("dinov2_vits14", 384), "b": ("dinov2_vitb14", 768), "l": ("dinov2_vitl14", 1024)}
CACHE = Path(os.environ.get("DINOBLOOM_CACHE", "E:/cache")) / f"dinobloom_{VARIANT}_cnmc.npz"
OUT = resolve("reports/foundation")
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


# ----------------------------------------------------------------------------------- model
def load_dinobloom(variant: str = VARIANT):
    """DinoBloom's official weights (MarrLab/DinoBloom) in timm's DINOv2 ViT. The model card's recipe fetches DINOv2 code
    through torch.hub, which needs GitHub at load time; timm builds the same network locally, and its DINOv2 checkpoint
    converter maps the official keys (checked against an independent timm port in src/foundation.py's self-test)."""
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from timm.models.vision_transformer import checkpoint_filter_fn

    names = {"s": "vit_small_patch14_dinov2", "b": "vit_base_patch14_dinov2", "l": "vit_large_patch14_dinov2"}
    model = timm.create_model(names[variant], pretrained=False, img_size=224, num_classes=0)
    ckpt = torch.load(hf_hub_download("MarrLab/DinoBloom", f"pytorch_model_{variant}.bin"), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint_filter_fn(ckpt, model), strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev).eval(), dev


def embed_uint8(model, dev, imgs224: np.ndarray, batch: int = 64) -> np.ndarray:
    """imgs224: uint8 [N, 224, 224, 3] -> float32 [N, dim] CLS embeddings."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(imgs224), batch):
            x = (imgs224[i:i + batch].astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - MEAN) / STD
            out.append(model(torch.from_numpy(x).to(dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, DIMS[VARIANT][1]), np.float32)


def to224(img: Image.Image) -> np.ndarray:
    # torchvision Resize((224,224)) on a PIL image is a bilinear, antialiased PIL resize
    return np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR))


# ----------------------------------------------------------------------------------- C-NMC features
def embed_cnmc() -> None:
    import pandas as pd

    cfg = load_config()
    df = pd.read_csv(resolve(cfg["paths"]["splits_csv"]))
    root = Path(cfg["paths"]["data_root"])
    model, dev = load_dinobloom()
    feats = []
    with Progress("dinobloom_embed", "Turning every C-NMC cell into a DinoBloom fingerprint", total=len(df),
                  explain=("Runs each of the 12,068 C-NMC cell images through DinoBloom, a network trained on blood cells "
                           "from 13 public datasets, and saves the 768-number summary it produces. This runs once."),
                  steps=["Describe every cell with DinoBloom"]) as pg:
        pg.set_stage("embedding", total=len(df), step=0)
        for i in range(0, len(df), 256):
            chunk = df.iloc[i:i + 256]
            imgs = np.stack([to224(Image.open(root / p)) for p in chunk.image_path])
            feats.append(embed_uint8(model, dev, imgs))
            pg.step(len(chunk))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, x=np.concatenate(feats), image_path=df.image_path.to_numpy(dtype=str), label=df.label.to_numpy(),
             split=df.split.to_numpy(dtype=str), group=df.group_id.to_numpy(dtype=str))
    print("saved", CACHE)


# ----------------------------------------------------------------------------------- classifier
def _metrics(y, p, g):
    from src.evaluate import binary_metrics, cluster_bootstrap, patient_level

    m = binary_metrics(p, y)
    pr = cluster_bootstrap(lambda a, b: binary_metrics(a, b)["pr_auc"], p, y, g, iters=1000)
    pp, py = patient_level(p, y, g)
    sub = binary_metrics(pp, py) if len(np.unique(py)) == 2 else None
    return {"cells": int(len(y)), "subjects": int(len(np.unique(g))), "pr_auc": m["pr_auc"], "pr_auc_ci95": list(pr[1:]),
            "roc_auc": m["roc_auc"], "f1_at_0.5": m["f1"], "recall_at_0.5": m["recall"], "specificity_at_0.5": m["specificity"],
            "ece": m["ece"],
            "subject_level": None if sub is None else {"roc_auc": sub["roc_auc"], "accuracy_at_0.5": sub["accuracy"],
                                                       "all_patients_flagged": f'{sub["tp"]}/{sub["tp"] + sub["fn"]}',
                                                       "healthy_flagged": f'{sub["fp"]}/{sub["fp"] + sub["tn"]}'}}


def fit() -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = np.load(CACHE, allow_pickle=False)
    x, y, split, g = d["x"], d["label"].astype(int), d["split"], d["group"]
    tr, va, te = split == "train", split == "val", split == "test"
    grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    val_scores = {}
    for c in grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
        clf.fit(x[tr], y[tr])
        val_scores[c] = float(average_precision_score(y[va], clf.predict_proba(x[va])[:, 1]))
    best = max(val_scores, key=val_scores.get)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best, class_weight="balanced", max_iter=5000))
    clf.fit(x[tr], y[tr])
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT / "dinobloom_b_logreg.joblib")
    res = {"variant": f"DinoBloom-{VARIANT.upper()}", "C_grid_val_pr_auc": val_scores, "chosen_C": best,
           "validation": _metrics(y[va], clf.predict_proba(x[va])[:, 1], g[va]),
           "test_evaluated_once": _metrics(y[te], clf.predict_proba(x[te])[:, 1], g[te])}
    (OUT / "cnmc_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return res


# ----------------------------------------------------------------------------------- export for the web app
def export() -> dict:
    """One ONNX graph, image -> leukemic logit: DinoBloom-B, then the scaler and logistic regression folded into a single
    linear layer. Matrix weights are stored as 8-bit (weight-only, blocks of 32, activations stay fp32) so the browser
    download is about 94 MB instead of 327; the graph is checked against PyTorch on validation cells. Dynamic int8
    (activations quantised too) was tried first and broke the model (mean probability shift 0.28): DINOv2 activations
    have large outliers. Temperature and the refer-to-expert band are
    fitted on the validation subjects with the same rules as the deployed model."""
    import hashlib

    import joblib
    import onnxruntime as ort
    import torch
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from sklearn.metrics import roc_auc_score

    from src.calibrate import apply_temperature, choose_band, fit_temperature

    clf = joblib.load(OUT / "dinobloom_b_logreg.joblib")
    scaler, lr = clf[0], clf[-1]
    w = lr.coef_[0] / scaler.scale_
    b = float(lr.intercept_[0] - (lr.coef_[0] * scaler.mean_ / scaler.scale_).sum())

    backbone, _ = load_dinobloom()
    backbone = backbone.cpu().float()

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.fc = torch.nn.Linear(len(w), 1)
            with torch.no_grad():
                self.fc.weight.copy_(torch.from_numpy(w.astype(np.float32))[None])
                self.fc.bias.fill_(b)

        def forward(self, image):
            return self.fc(self.backbone(image))[:, 0]

    net = Head().eval()
    models = resolve("models")
    fp32 = Path(os.environ.get("TEMP", "E:/cache")) / "dinobloom_b_fp32.onnx"
    final = models / "dinobloom_b_cnmc.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, 224, 224), str(fp32), input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "n"}, "logit": {0: "n"}}, opset_version=17, dynamo=False)
    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=8, block_size=32, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(str(final), use_external_data_format=False)
    fp32.unlink(missing_ok=True)

    # calibration on the cached validation embeddings (same numbers the classifier was chosen on)
    d = np.load(CACHE, allow_pickle=False)
    x, y, split = d["x"], d["label"].astype(int), d["split"]
    va = split == "val"
    z_val = clf.decision_function(x[va])
    t = fit_temperature(z_val, y[va])
    band = choose_band(apply_temperature(z_val, t), y[va])

    # does the int8 graph give the same answers? 400 validation cells, re-read from disk
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(np.flatnonzero(va), 400, replace=False))
    root = Path(load_config()["paths"]["data_root"])
    imgs = np.stack([to224(Image.open(root / p)) for p in d["image_path"][idx]])
    xin = ((imgs.astype(np.float32).transpose(0, 3, 1, 2) / 255.0) - MEAN) / STD
    sess = ort.InferenceSession(str(final), providers=["CPUExecutionProvider"])
    z_q = np.concatenate([sess.run(None, {"image": xin[i:i + 32]})[0] for i in range(0, len(xin), 32)])
    z_ref = clf.decision_function(x[idx])
    p_q, p_ref = apply_temperature(z_q, t), apply_temperature(z_ref, t)
    check = {"weights": "8-bit weight-only (MatMulNBits, block 32, symmetric)", "cells": len(idx), "max_abs_prob_diff": float(np.abs(p_q - p_ref).max()),
             "mean_abs_prob_diff": float(np.abs(p_q - p_ref).mean()),
             "band_decision_agreement": float((np.digitize(p_q, [band["t_low"], band["t_high"]]) ==
                                               np.digitize(p_ref, [band["t_low"], band["t_high"]])).mean()),
             "roc_auc_int8": float(roc_auc_score(y[idx], p_q)), "roc_auc_reference": float(roc_auc_score(y[idx], p_ref))}

    sha = hashlib.sha256(final.read_bytes()).hexdigest()
    fitted = json.loads((OUT / "cnmc_results.json").read_text())
    meta = {"name": "DinoBloom-B-LR", "architecture": "DinoBloom-B (frozen DINOv2 ViT-B/14, MarrLab) + logistic regression",
            "input": {"size": 224, "mean": MEAN.ravel().tolist(), "std": STD.ravel().tolist(), "resize": "bilinear, antialiased"},
            "temperature": t, "t_low": band["t_low"], "t_high": band["t_high"],
            "band_target": "leukemic miss rate <= 2% on validation cells",
            "band_validation": band, "int8_check": check,
            # out-of-distribution guard, same idea as the deployed model's: refuse cells whose logit is far beyond
            # anything seen on C-NMC training and validation cells
            "ood_max_abs_logit": round(1.5 * float(np.abs(clf.decision_function(x[split != "test"])).max()), 1),
            "metrics": {"validation": fitted["validation"], "test": fitted["test_evaluated_once"]},
            "onnx_file": final.name, "onnx_sha256": sha, "onnx_mb": round(final.stat().st_size / 2**20, 1),
            "weights_licence": "DinoBloom weights: Apache-2.0 (Koch et al., MICCAI 2024)",
            "disclaimer": "Research demo. Not for clinical use."}
    # an experiment, not the deployed model: its metadata stays beside its ONNX file and is not shipped to the website
    (models / "dinobloom_b_cnmc_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps(meta, indent=2, default=float))
    return meta


# ----------------------------------------------------------------------------------- outside tests
def outside(clf_path: Path | None = None, out_path: Path | None = None, job: str = "dinobloom_outside",
            emb_filter: Path | None = None, variant: str = VARIANT, scorer=None) -> dict:
    """Taleqani + BCCD outside tests for a DinoBloom classifier. With `emb_filter` (a {clf, threshold} joblib on DinoBloom
    features) the shape rule and the ONNX cell-type filter are replaced: every cell off the image edge is cut out, and
    the learned filter on the same DinoBloom features decides which cells are read. Per-smear scores are saved next to
    the JSON (.npz) for later threshold studies."""
    import joblib
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque

    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.compare_models import paired
    from src.smear_eval import BCCD, CellTypeFilter, _crops_one, taleqani_items
    from src.vision.segment import CNMC_RBC_PX, find_white_cells, to_cnmc

    clf = joblib.load(clf_path or OUT / "dinobloom_b_logreg.joblib")
    model, dev = load_dinobloom(variant)
    filt = CellTypeFilter(resolve("models"))
    ef = joblib.load(emb_filter) if emb_filter else None
    efkeep = lambda e: ef["clf"].predict_proba(e)[:, 1] >= ef["threshold"] if len(e) else np.zeros(0, bool)
    stain = json.loads(resolve("models/cnmc_stain.json").read_text())
    pooled = json.loads((resolve("reports/smear") / "taleqani_classic_old_filter.json").read_text())["red_cell_diameter_px"]["pooled"]
    arms = ["ruler_x1.0_stain", "ruler_x1.0"]
    items = taleqani_items(None)
    scores = {f"{a}{s}": [] for a in arms for s in ("", "_filtered")}

    with Progress(job, "Testing the DinoBloom classifier on another hospital's smears", total=len(items),
                  explain=("Finds every white cell in 3,216 Taleqani smear photos, describes each with DinoBloom, and asks the "
                           "new classifier about it, so it can be compared smear by smear with the current model."),
                  steps=["Taleqani smears", "Healthy BCCD smears"]) as pg:
        pg.set_stage("Taleqani", total=len(items), step=0)
        with ProcessPoolExecutor(6) as ex:
            q = deque()

            def results():
                for it in items:
                    q.append(ex.submit(_crops_one, (it["data"], pooled, stain, "classic", 224, arms, filt.size, ef is None)))
                    if len(q) >= 24:
                        yield q.popleft().result()
                while q:
                    yield q.popleft().result()

            for it, out in zip(items, results()):
                emb = {a: embed_uint8(model, dev, out["crops"][a]) for a in arms if len(out["crops"][a])}
                if ef is not None:
                    keep = efkeep(emb["ruler_x1.0_stain"]) if "ruler_x1.0_stain" in emb else None
                else:
                    keep = filt.keep(out["filter_crops"]) if "filter_crops" in out else None
                for a in arms:
                    if a not in emb:
                        p = np.zeros(0)
                    else:   # scorer: a fine-tuned network reading the crops directly (the filter still uses frozen features)
                        p = scorer(out["crops"][a]) if scorer else clf.predict_proba(emb[a])[:, 1]
                    scores[a].append(float(p.mean()) if len(p) else np.nan)
                    pf = p[keep] if keep is not None and len(p) else np.zeros(0)
                    scores[f"{a}_filtered"].append(float(pf.mean()) if len(pf) else np.nan)
                pg.step(note=it["name"])

        y = np.array([it["label"] for it in items])
        names = np.array([it["name"] for it in items])
        npz = (out_path or OUT / "outside_results.json").with_suffix(".npz")
        np.savez(npz, names=names, labels=y, **{f"mean_{k}": np.array(v) for k, v in scores.items()})
        taleqani = {}
        for k, v in scores.items():
            v = np.array(v)
            ok = ~np.isnan(v)
            taleqani[k] = {"smears": int(ok.sum()), "roc_auc": float(roc_auc_score(y[ok], v[ok])),
                           "pr_auc": float(average_precision_score(y[ok], v[ok])),
                           "benign_called_leukemic_at_0.5": float((v[ok][y[ok] == 0] > 0.5).mean()),
                           "all_smears_missed_at_0.5": float((v[ok][y[ok] == 1] <= 0.5).mean())}
        old = np.load(resolve("reports/smear") / "_scores_taleqani_old_filter.npz")
        assert np.array_equal(old["names"], names)
        cmp = {"main_arm_stain_filtered": paired(old["mean_ruler_x1.0_stain_filtered"], np.array(scores["ruler_x1.0_stain_filtered"]), y),
               "stain_unfiltered": paired(old["mean_ruler_x1.0_stain"], np.array(scores["ruler_x1.0_stain"]), y)}

        # healthy BCCD smears: every kept lymphoid cell should read normal
        import cv2
        import pandas as pd

        pg.set_stage("BCCD", total=1, step=1)
        bccd_pooled = json.loads((resolve("reports/smear") / "bccd_classic_filter.json").read_text())["red_cell_ruler"]["pooled_estimate_px"]
        files = sorted(pd.read_csv(BCCD / "test.csv").filename.unique())
        probs = {"stain": [], "raw": []}
        for f in files:
            im = cv2.imread(str(BCCD / "BCCD" / "JPEGImages" / f))
            if im is None:
                continue
            rgb = im[..., ::-1].copy()
            cells = [c for c in find_white_cells(rgb, rbc_override=bccd_pooled).cells
                     if (c.lymphoid_like or ef is not None) and not c.touches_border]
            if not cells:
                continue
            full_s = [to_cnmc(rgb, c, CNMC_RBC_PX / bccd_pooled, "ruler", stain=stain) for c in cells]
            e_s = embed_uint8(model, dev, np.stack([to224(Image.fromarray(fc)) for fc in full_s]))
            if ef is not None:
                keep = efkeep(e_s)
            else:
                keep = filt.keep(np.stack([np.asarray(Image.fromarray(fc).resize((filt.size, filt.size), Image.BILINEAR)) for fc in full_s]))
            for k, st in (("stain", stain), ("raw", None)):
                crops = np.stack([to224(Image.fromarray(fc)) for fc in full_s]) if st is not None else np.stack(
                    [to224(Image.fromarray(to_cnmc(rgb, c, CNMC_RBC_PX / bccd_pooled, "ruler", stain=None))) for c in cells])
                if scorer:
                    probs[k] += list(scorer(crops)[keep])
                else:
                    e = e_s if st is not None else embed_uint8(model, dev, crops)
                    probs[k] += list(clf.predict_proba(e)[:, 1][keep])
        pg.step()
    bccd = {k: {"cells_after_filter": len(v), "called_leukemic_at_0.5": float(np.mean(np.array(v) > 0.5)) if v else None}
            for k, v in probs.items()}
    res = {"taleqani": taleqani, "paired_vs_deployed_model": cmp, "bccd_healthy": bccd,
           "note": "DinoBloom-B frozen + logistic regression; same smears, finder, filter and crops as the deployed model's tests"}
    res["classifier"] = str(clf_path or OUT / "dinobloom_b_logreg.joblib")
    res["cell_filter"] = str(emb_filter) if emb_filter else "shape rule + ONNX cell-type filter (deployed)"
    res["dinobloom_variant"] = variant
    (out_path or OUT / "outside_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["embed", "fit", "export", "outside"])
    a = ap.parse_args()
    t0 = time.time()
    {"embed": embed_cnmc, "fit": fit, "export": export, "outside": outside}[a.what]()
    print(f"{time.time() - t0:.0f}s")
