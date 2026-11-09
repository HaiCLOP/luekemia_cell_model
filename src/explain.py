"""Explainability on validation images (the test split is never read here).

Usage: python -m src.explain [--n 160] [--smoke]

* Grad-CAM (last conv block) and Integrated Gradients (black baseline) for EfficientNet-B0 seed 42.
* Galleries: true positive, true negative, false positive, false negative, referred (band), hard subjects.
  Case categories use the deployed decision rule: 5-seed ensemble, temperature-scaled, referral band.
* Faithfulness: deletion and insertion curves (black-out / reveal top-k% salient pixels), compared with
  random pixel order and with random order restricted to the cell. Lower deletion AUC and higher
  insertion AUC = more faithful.
* Focus check: share of saliency mass on the cell (non-black pixels). Maps with < 50% on the cell are
  flagged as background/border-focused.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src.calibrate import apply_band, apply_temperature
from src.data import build_transforms, load_split
from src.evaluate import sigmoid
from src.model import build_model
from src.utils import load_config, resolve, seed_everything

STEPS = np.linspace(0, 1, 21)  # fraction of pixels deleted / inserted


class PosLogit(torch.nn.Module):
    """Wrap the 1-logit classifier so attribution libraries see a [B, 1] output."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def load_model(cfg, device):
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(torch.load(resolve(cfg["paths"]["checkpoints"]) / "effb0_main_s42.pt",
                                     map_location=device, weights_only=True))
    return model


def raw_image(cfg, rel) -> np.ndarray:
    s = cfg["data"]["img_size"]
    img = Image.open(resolve(cfg["paths"]["data_root"]) / rel).convert("RGB").resize((s, s), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def cell_mask(rgb01: np.ndarray) -> np.ndarray:
    return rgb01.sum(-1) > 0.06


def gradcam_maps(model, x, cls):
    cam = GradCAM(model=model, target_layers=[model.backbone.conv_head])
    targets = [BinaryClassifierOutputTarget(int(c)) for c in cls]
    return cam(input_tensor=x, targets=targets)  # [B, H, W] in [0, 1]


def ig_maps(model, x, cls, baseline):
    ig = IntegratedGradients(PosLogit(model))
    attr = ig.attribute(x, baselines=baseline.expand_as(x), target=0, n_steps=32, internal_batch_size=16)
    a = attr.detach().cpu().numpy()
    # explain the predicted class: positive evidence for leukemic, negative evidence for normal
    sign = np.where(np.asarray(cls) == 1, 1.0, -1.0)[:, None, None, None]
    a = np.clip(a * sign, 0, None).sum(1)
    a = a / (a.reshape(len(a), -1).max(1)[:, None, None] + 1e-12)
    return a


@torch.no_grad()
def prob_of_class(model, xs, cls):
    p = torch.sigmoid(model(xs).float().squeeze(1)).cpu().numpy()
    return p if cls == 1 else 1 - p


@torch.no_grad()
def curves(model, x, sal, cls, baseline, rng, cell=None):
    """Deletion and insertion curves for one image. `sal` [H,W]; order pixels by descending saliency."""
    H, W = sal.shape
    if sal is None:
        order = rng.permutation(H * W)
    else:
        order = np.argsort(-sal.ravel(), kind="stable")
    n = H * W
    dels, inss = [], []
    for f in STEPS:
        k = int(round(f * n))
        m = np.zeros(n, dtype=bool)
        m[order[:k]] = True
        m = torch.from_numpy(m.reshape(H, W)).to(x.device)
        dels.append(torch.where(m, baseline[0], x[0]))
        inss.append(torch.where(m, x[0], baseline[0]))
    d = prob_of_class(model, torch.stack(dels), cls)
    i = prob_of_class(model, torch.stack(inss), cls)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return d, i, float(trapz(d, STEPS)), float(trapz(i, STEPS))


def random_cell_order(mask, rng):
    idx = np.arange(mask.size)
    on, off = idx[mask.ravel()], idx[~mask.ravel()]
    return np.concatenate([rng.permutation(on), rng.permutation(off)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160, help="validation images for faithfulness curves")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    tf = build_transforms(cfg, train=False)
    mean = torch.tensor(cfg["data"]["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg["data"]["std"]).view(1, 3, 1, 1).to(device)
    baseline = ((torch.zeros(1, 3, 1, 1, device=device) - mean) / std).expand(1, 3, cfg["data"]["img_size"], cfg["data"]["img_size"])

    # ---- deployed decisions on validation (ensemble + temperature + band) to pick gallery cases
    val = load_split(cfg, "val")
    P = []
    for s in cfg["calibration"]["ensemble_seeds"]:
        d = np.load(resolve("reports/preds") / f"effb0_main_s{s}_val.npz", allow_pickle=True)
        P.append(sigmoid(d["logits"]))
    p_ens = np.mean(P, 0)
    z = np.log(np.clip(p_ens, 1e-6, 1 - 1e-6) / np.clip(1 - p_ens, 1e-6, 1))
    p_cal = apply_temperature(z, cfg["band"]["temperature"])
    dec = apply_band(p_cal, cfg["band"]["t_low"], cfg["band"]["t_high"])
    val = val.assign(p=p_cal, decision=dec)
    val["case"] = np.select(
        [dec == -1, (dec == 1) & (val.label == 1), (dec == 0) & (val.label == 0), (dec == 1) & (val.label == 0), (dec == 0) & (val.label == 1)],
        ["referred", "true positive", "true negative", "false positive", "false negative"], "other")

    # ---- galleries
    per = 1 if args.smoke else 5
    picks = []
    for case in ["true positive", "true negative", "false positive", "false negative", "referred"]:
        sub = val[val.case == case]
        picks.append(sub.sample(min(per, len(sub)), random_state=1))
    hard = val[val.group_id.isin(["G_H34", "G_H9"])]
    picks.append(hard.sample(min(per, len(hard)), random_state=2).assign(case="hard subject (H34/H9)"))
    gal = pd.concat(picks).reset_index(drop=True)

    def explain_rows(df):
        xs = torch.stack([tf(Image.open(resolve(cfg["paths"]["data_root"]) / r).convert("RGB")) for r in df.image_path]).to(device)
        with torch.no_grad():
            pm = torch.sigmoid(model(xs).squeeze(1)).cpu().numpy()
        cls = (pm >= 0.5).astype(int)
        cams = gradcam_maps(model, xs, cls)
        igs = ig_maps(model, xs, cls, baseline)
        return xs, pm, cls, cams, igs

    xs, pm, cls, cams, igs = explain_rows(gal)
    figs = resolve("reports/figures")
    gal = gal.assign(_i=np.arange(len(gal)))
    cases = list(dict.fromkeys(gal.case))
    # one wide figure per case type: columns = cells, rows = image / Grad-CAM / IG
    for case in cases:
        sub = gal[gal.case == case]
        k = len(sub)
        fig, axes = plt.subplots(3, k, figsize=(2.3 * k, 7.4), squeeze=False)
        for c, (_, r) in enumerate(sub.iterrows()):
            i = r._i
            img = raw_image(cfg, r.image_path)
            axes[0, c].imshow(img)
            axes[0, c].set_title(f"{r.group_id} · {'ALL' if r.label else 'hem'}\nensemble p={r.p:.2f}", fontsize=9)
            axes[1, c].imshow(show_cam_on_image(img, cams[i], use_rgb=True))
            axes[2, c].imshow(img)
            axes[2, c].imshow(igs[i], cmap="magma", alpha=0.7, vmin=0, vmax=np.percentile(igs[i], 99.5) + 1e-9)
            for a in axes[:, c]:
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_visible(False)
        axes[0, 0].set_ylabel("cell", fontsize=10)
        axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)
        axes[2, 0].set_ylabel("Integrated\nGradients", fontsize=10)
        fig.suptitle(f"{case.capitalize()}  ·  EfficientNet-B0 seed 42, validation", fontsize=11, x=0.02, ha="left")
        fig.tight_layout()
        fig.savefig(figs / f"gallery_{case.split(' (')[0].replace(' ', '_')}.png", dpi=130)
        plt.close(fig)

    # ---- faithfulness + focus on a stratified validation sample
    n = 12 if args.smoke else args.n
    samp =pd.concat([d.sample(n // 2, random_state=3) for _, d in val.groupby("label")]).reset_index(drop=True)
    recs, curves_store = [], {k: {"del": [], "ins": []} for k in ("gradcam", "ig", "random", "random_cell")}
    for start in range(0, len(samp), 16):
        chunk = samp.iloc[start:start + 16]
        xs, pm, cls, cams, igs = explain_rows(chunk)
        for j in range(len(chunk)):
            img = raw_image(cfg, chunk.image_path.iloc[j])
            mask = cell_mask(img)
            rec = {"image_path": chunk.image_path.iloc[j], "group_id": chunk.group_id.iloc[j], "label": int(chunk.label.iloc[j]),
                   "pred": int(cls[j]), "p": float(pm[j]), "cell_frac": float(mask.mean())}
            for name, sal in (("gradcam", cams[j]), ("ig", igs[j])):
                rec[f"{name}_on_cell"] = float((sal * mask).sum() / (sal.sum() + 1e-12))
                d_, i_, da, ia = curves(model, xs[j:j + 1], sal, cls[j], baseline, rng)
                rec[f"{name}_del_auc"], rec[f"{name}_ins_auc"] = da, ia
                curves_store[name]["del"].append(d_); curves_store[name]["ins"].append(i_)
            rnd = rng.permutation(mask.size).astype(float)
            d_, i_, da, ia = curves(model, xs[j:j + 1], rnd.reshape(mask.shape), cls[j], baseline, rng)
            rec["random_del_auc"], rec["random_ins_auc"] = da, ia
            curves_store["random"]["del"].append(d_); curves_store["random"]["ins"].append(i_)
            order = random_cell_order(mask, rng)
            rank = np.empty(mask.size); rank[order] = np.arange(mask.size)[::-1]
            d_, i_, da, ia = curves(model, xs[j:j + 1], rank.reshape(mask.shape), cls[j], baseline, rng)
            rec["random_cell_del_auc"], rec["random_cell_ins_auc"] = da, ia
            curves_store["random_cell"]["del"].append(d_); curves_store["random_cell"]["ins"].append(i_)
            recs.append(rec)
        print(f"faithfulness {min(start + 16, len(samp))}/{len(samp)}", flush=True)

    df = pd.DataFrame(recs)
    tabs = resolve("reports/tables")
    df.to_csv(tabs / "explain_per_image.csv", index=False)
    # Grad-CAM is a 7x7 map upsampled to 224 px, so it always spills past the cell edge. Judge focus by
    # enrichment = share of saliency on the cell / share of the image the cell covers (1.0 = no focus).
    df["gradcam_enrichment"] = df.gradcam_on_cell / df.cell_frac
    flagged = df[df.gradcam_enrichment < 1.0]
    summary = {
        "n_images": len(df), "mean_cell_area": df.cell_frac.mean(),
        "on_cell_share": {"gradcam": df.gradcam_on_cell.mean(), "ig": df.ig_on_cell.mean()},
        "gradcam_enrichment_median": float(df.gradcam_enrichment.median()),
        "flagged_gradcam_not_focused_on_cell": int(len(flagged)),
        "gradcam_lt_50pct_on_cell": int((df.gradcam_on_cell < 0.5).sum()),
        "deletion_auc": {k: df[f"{k}_del_auc"].mean() for k in ("gradcam", "ig", "random", "random_cell")},
        "insertion_auc": {k: df[f"{k}_ins_auc"].mean() for k in ("gradcam", "ig", "random", "random_cell")},
        "deletion_auc_sd": {k: df[f"{k}_del_auc"].std() for k in ("gradcam", "ig", "random", "random_cell")},
        "note": "Black baseline for deletion/insertion and for IG. IG gives zero attribution to pixels equal to the baseline, "
                "so its on-cell share is high by construction; Grad-CAM's on-cell share is the informative focus check.",
    }
    resolve("reports/explain").mkdir(parents=True, exist_ok=True)
    (resolve("reports/explain") / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    flagged.to_csv(tabs / "explain_flagged_background_focus.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"gradcam": "#d0654f", "ig": "#3f63c4", "random": "#9aa1a9", "random_cell": "#5a9e6f"}
    names = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "random": "random pixels", "random_cell": "random pixels on cell"}
    for ax, kind, title in ((axes[0], "del", "Deletion (lower = more faithful)"), (axes[1], "ins", "Insertion (higher = more faithful)")):
        for k in colors:
            arr = np.array(curves_store[k][kind])
            ax.plot(STEPS * 100, arr.mean(0), color=colors[k], lw=2 if k in ("gradcam", "ig") else 1.4,
                    ls="-" if k in ("gradcam", "ig") else "--",
                    label=f"{names[k]} (AUC {summary['deletion_auc' if kind == 'del' else 'insertion_auc'][k]:.3f})")
        ax.set_xlabel("% of pixels removed" if kind == "del" else "% of pixels revealed")
        ax.set_ylabel("P(predicted class)"); ax.set_title(title); ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"Faithfulness on {len(df)} validation cells (black baseline)", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "faithfulness.png", dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
