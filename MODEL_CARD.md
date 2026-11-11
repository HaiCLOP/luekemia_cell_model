---
license: cc-by-4.0
library_name: onnx
pipeline_tag: image-classification
tags:
  - medical-imaging
  - hematology
  - leukemia
  - calibration
  - selective-prediction
  - research-only
datasets:
  - dwb2023/cnmc-leukemia-2019
  - dwb2023/cnmc-leukemia-2019-test
metrics:
  - average_precision
  - roc_auc
  - expected_calibration_error
---

# effb0-ens5-v1: ALL vs normal single-cell classifier with a refer-to-expert band

**Research demo. Not for clinical use.** This model has not been clinically validated and must not be used to make or
support diagnostic or treatment decisions.

## Summary

| | |
|---|---|
| Task | Binary classification of a single segmented white blood cell image: B-lineage acute lymphoblastic leukemia (ALL) vs normal |
| Model | Deep ensemble of 5 EfficientNet-B0 (timm `efficientnet_b0.ra_in1k`) fine-tuned with different seeds |
| Output | Calibrated P(leukemic), a decision (`leukemic`, `normal`, `refer_to_expert`) and a class activation map |
| Calibration | Temperature 1.0728 on the ensemble logit |
| Referral band | P < 0.20: normal. P > 0.50: leukemic. Between: refer to an expert. Chosen for a leukemic miss rate of 2% or less on validation subjects |
| Format | ONNX (opset 17), 75.6 MB, CPU inference about 45 ms per cell (65 ms with heatmap) |
| Author | Arnav Srivastava, HaiCLOP Labs (independent research) |

## Intended use

Research and education on calibrated uncertainty and selective prediction in medical imaging. Inputs should be single
segmented lymphocyte crops similar to C-NMC 2019 (cell on a black background, about 450 px). Out of scope: whole slides,
other cell types, other stains or microscopes without re-validation, and any clinical use.

The API enforces part of this: images that are not near-square, have no black background around the object, contain more
than one object or show no internal detail are refused with `unsupported_input` or `insufficient_image_quality` and get
no score (`api/quality.py`). These are coarse checks, not an out-of-distribution detector. The output is a research score
for one cropped cell, not a probability that a person has leukemia, and it is not a patient-level diagnosis.

## Training data

C-NMC 2019 (ISBI challenge) from The Cancer Imaging Archive, CC BY 3.0, via two Hugging Face mirrors. The data audit removed
350 all-zero (corrupt) files and 110 pixel-identical copies, and found 4 groups of patient IDs that shared identical images
(for example IDs 20 and 21 shared 71 images). Those IDs were merged before splitting. Result: 12,068 cells from 92
independent subjects, split by subject into train (48 subjects, 7,844 cells), validation (18, 1,810) and test (26, 2,414).
No subject or identical image appears in more than one split (checked by automated tests).

## Training procedure

ImageNet initialisation; stage 1 trains the head with the backbone frozen (5 epochs, AdamW 1e-3); stage 2 unfreezes the
last three blocks (up to 25 epochs, 1e-4, cosine). Batch 16 with gradient accumulation 2, full precision (fp16 produced NaNs
on the GTX 1650), class-weighted BCE, flips, 90-degree rotations, small affine and brightness/contrast jitter. Early
stopping on validation PR-AUC. Seeds 42 to 46.

## Evaluation (validation subjects only; test set not yet evaluated)

95% intervals resample whole subjects 1,000 times.

| Model | PR-AUC | ROC-AUC | Recall (ALL) @0.5 | Specificity @0.5 | ECE |
|---|---|---|---|---|---|
| **5-seed ensemble (this model)** | **0.974** [0.906, 0.994] | 0.952 | 0.925 | 0.818 | 0.021 |
| EfficientNet-B0, single seed (mean of 5) | 0.970 (sd 0.002) | 0.946 | 0.915 | 0.821 | 0.037 |
| MobileNetV3-Large | 0.970 | 0.946 | 0.884 | 0.859 | 0.065 |
| DINOv2 ViT-S/14 | 0.961 | 0.934 | 0.873 | 0.849 | 0.080 |
| Small CNN from scratch | 0.948 | 0.915 | 0.936 | 0.722 | 0.026 |
| Frozen ImageNet features + logistic regression | 0.892 | 0.850 | 0.857 | 0.684 | 0.069 |

Refer-to-expert band, nested cross-fit over validation subjects (temperature and band both fitted on two thirds of the
subjects, scored on the held-out third, same three folds; earlier versions of this card used two different fold splits
and reported 2.90%):

| Policy | Referred | Missed leukemic | False alarms on normal |
|---|---|---|---|
| Single threshold 0.5 | 0% | 7.55% | 18.2% |
| Single threshold lowered to meet 2% miss | 0% | 1.99% | 34.3% |
| Random referral at the same rate | 8.1% | 6.94% | n/a |
| **Referral band [0.20, 0.50]** | 8.1% | **2.74%** | 18.4% |

The 2% target is met in-sample only; out of sample the estimate is 2.74%. These are still validation numbers: the
checkpoints themselves were selected on the same subjects. Per 100 referred cells the band avoids 39.7 missed leukemic cells, against 5.0 for random referral.

## Known failure mode: cell size

Among normal validation cells the Spearman correlation between cell area and predicted P(leukemic) is 0.63 to 0.72 across
seeds (0.53 to 0.78 across all architectures tested). Normal cells that trigger false alarms are as large as leukemic cells.
One validation subject (H34) with large normal cells is classified leukemic at the subject level. Scale-jitter augmentation
did not remove this (0.62). Grad-CAM concentrates on the cell (median 2.5x enrichment) but does not localise features within
the cell better than random cell pixels. Expect reduced performance on images with different magnification.

How much of the score is size? (`python -m src.size_matched`, deployed ensemble, validation, subject-clustered 95% CIs)

| Check | ROC-AUC |
|---|---|
| Deployed model, all validation cells | 0.952 (0.856 to 0.978) |
| Cell area alone as the classifier | 0.781 (0.482 to 0.860) |
| Model, inside each area quintile (weighted mean) | 0.918 |
| Model, 334 leukemic/normal pairs matched on area (area AUC after matching 0.504) | 0.883 (0.767 to 0.938) |

The model keeps most of its ranking ability when size is held fixed, so it is not only a size detector, but size is worth
roughly 0.03 to 0.07 AUC of the headline number. The errors follow size closely: the share of normal cells scored above
the referral band rises from 2.5% in the smallest area quintile to 51% in the largest. Any change of magnification or
crop policy moves cells between these quintiles, which is why external validation comes before any new disease class.

## Smear pipeline (cell finder in front of the classifier)

`src/vision/segment.py` turns a stained peripheral-blood smear image into C-NMC-style inputs. It is classical image
processing with no training: white balance, a stain map, white-cell nuclei as the top class of a three-class Otsu split,
watershed only for touching round nuclei, and a thin cytoplasm rim. Red cells (about 7.5 µm) act as a ruler, so every
cell is rescaled to the magnification of C-NMC before it is cut onto a black 450 px frame. The red-cell size is pooled over
all images from one microscope, or taken from a pixel size the user enters. Cells with lobed or irregular nuclei are
outlined but not scored. The API exposes this as `POST /analyze-smear`, and the website as the Smear page.

Checked on BCCD (364 smear images from healthy donors, MIT licence; `python -m src.smear_eval bccd`):

| Check | Result |
|---|---|
| White cells found (372 boxed) | recall 99.7%, precision 96.7%, 0.1 s per image |
| Boxed cells split into several detections | 7 of 372 |
| Red-cell ruler, pooled over the set | 87.7 px against 104.5 px from the boxes (−16%; boxes include margin, so the true error is smaller) |
| Ruler, single image | 43% of images within 15% of the box median; per-image CV 24% |
| Cellpose 3 instead of the classical finder (61-image subset) | recall 82.5% vs 98.4%, 12x slower: not used |

**Negative result.** On BCCD's healthy smears, 92% of the cells the finder kept as lymphoid-like were called leukemic. Two
causes: (1) the shape filter lets many granulocytes through, because BCCD is mostly neutrophils and some lobed nuclei look
compact once lobes merge; (2) the classifier does not transfer. Even a clear normal lymphocyte scored 0.80 after colour
normalisation to C-NMC (Reinhard, `models/cnmc_stain.json`). BCCD images are softer and lower in resolution, and the
model learned C-NMC-specific colour, texture and size. The pipeline finds cells reliably; the single-cell model is the
weak link outside C-NMC. Results on an ALL dataset from another hospital are below.

**Out-of-distribution guard.** On smear crops in their original colours, ensemble members produced logits of 12,000 to
36,000, against a maximum of 14.9 over all 1,810 C-NMC validation cells. At that size each member's answer is 0 or 1, and
ONNX on CPU, PyTorch on CPU and PyTorch on GPU disagreed by up to 0.21 in probability: the score was decided by rounding,
not by the cell. After normalising colours to C-NMC the logits fell back into range (median 1.4, max 7.5) and the backends
agreed to 0.0003. Two consequences, both in the code now:

- smear crops are always colour-normalised to C-NMC before scoring (`models/cnmc_stain.json`);
- any input where a member's |logit| exceeds 25 (`ood_max_abs_logit` in `models/model_meta.json`) is refused as
  `out_of_distribution`, by both `/predict` and `/analyze-smear`, instead of getting a confident-looking number.

### Outside test: Taleqani Hospital ALL smears

3,216 peripheral-blood smear images (after removing exact copies) from Taleqani Hospital, Tehran (Aria et al. 2021,
doi 10.34740/KAGGLE/DSV/2175623): 2,712 from B-ALL patients (Early, Pre and Pro subtypes) and 504 "benign" smears from
patients without ALL whose lymphoid cells are hematogones, harmless B-cell precursors that look very like blasts. The
images are only 224 x 224 px, so a red cell is about 17 px wide and every cut-out is enlarged about 9 times. No patient IDs
are published, so the intervals below resample images and are optimistic. `python -m src.smear_eval taleqani`
(26 minutes on the GTX 1650; GPU scores matched the deployed ONNX file to 0.0003). The analysis plan is in the script's
docstring and was fixed before the first result was read.

| Crop setting (smear score = mean of its cells) | Cells refused by the guard | Smear ROC-AUC (95% CI) | Benign smears called leukemic | ALL smears missed |
|---|---|---|---|---|
| **Ruler scale + colour normalised to C-NMC (main result)** | 0% | **0.688 (0.661 to 0.713)** | 66.8% | 0.3% |
| Ruler scale, original colours (the pre-planned primary) | 87% | 0.708 unguarded; meaningless, see above | 80.9% | 0.0% |
| Ruler x0.8, original colours | 77% | 0.751 unguarded | 64.0% | 0.2% |
| Ruler x1.2, original colours | 92% | 0.694 unguarded | 86.7% | 0.0% |
| Every cell resized to a typical C-NMC cell, original colours | 93% | 0.730 unguarded | 91.1% | 0.0% |
| No rescaling | 0.1% | 0.506 | 0.0% | 57.4% |

With 84% of smears malignant, a PR-AUC of 0.895 for the main result is only modestly above the 0.843 a coin would get.
Only the main row is a valid measurement: in every original-colour row most cells push the network far outside its
normal range, so those AUCs come from scores that rounding decides. The honest summary is that **the C-NMC model, put
behind a working cell finder, ranks ALL smears above hematogone smears somewhat better than chance, but at its deployed
thresholds it calls two thirds of benign smears leukemic.** It is not usable on another hospital's smears as it stands.
On BCCD's healthy smears the same setting calls 90% of the kept cells leukemic (`reports/smear/bccd_classic.json`).

What this points to, in order: (1) a learned white-cell type filter, so granulocytes and monocytes never reach the
classifier; (2) retraining with colour, blur and resolution augmentation, or a stain-invariant backbone, then a fresh
outside evaluation; (3) a properly resolved external set with patient IDs, so smear-level and patient-level results can
be stated with honest intervals.

## Limitations

- One dataset, one acquisition setup; external evaluation on ALL-IDB2 is pending.
- 92 independent subjects; most differences between models are within the confidence intervals.
- Validation numbers only until the one-time test-set evaluation is published.

## Citation

Data: Mourya S., Kant S., Kumar P., Gupta A., Gupta R. (2019). ALL Challenge dataset of ISBI 2019 (C-NMC 2019) (Version 1).
The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.dc64i46r

Weights are released under CC BY 4.0 with the data attribution above.
