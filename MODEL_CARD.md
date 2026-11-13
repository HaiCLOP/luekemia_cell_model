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

# MH51401122: ALL vs normal single-cell classifier with a refer-to-expert band

**Research demo. Not for clinical use.**

Model ID **MH51401122** (formerly `effb0-ens5-v1`; renamed only, the file and its sha256 `bad021f6…` are unchanged). This model has not been clinically validated and must not be used to make or
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

## Final test (26 held-out subjects, evaluated once)

Run once on 2026-11-11 with `python -m src.final_test --split test --confirm-final`, after every modelling
choice was frozen: model MH51401122 (then named `effb0-ens5-v1`; same file, ONNX sha256 `bad021f68226`), temperature and band unchanged,
source commit `8d1d789`. These subjects were never used for training, checkpoint choice, calibration or
the band. 2,414 cells, all passed the input checks. 95% intervals resample whole subjects.

| Metric | Test (26 subjects) | Validation (18 subjects) |
|---|---|---|
| PR-AUC | **0.932** (0.820 to 0.973) | 0.974 |
| ROC-AUC | **0.890** (0.830 to 0.926) | 0.952 |
| ECE (calibrated) | **0.032** | 0.023 |
| Recall / specificity at 0.5 | 0.903 / 0.712 | 0.925 / 0.818 |
| Referral band: leukemic cells missed | **2.80%** | 2.74% (nested cross-fit) |
| Referral band: cells referred | 12.3% | 8.1% |
| Referral band: normal cells called leukemic | 28.8% | 18.4% |

The drop from validation to test is expected (validation chose the checkpoints and fitted the calibration), and the
intervals are wide because there are only 26 subjects. The referral band held its miss rate on new subjects (2.8%) at
the cost of referring more cells and raising more false alarms than on validation. Exploratory subject level (mean cell
score, not a validated patient endpoint): all 13 ALL subjects scored above 0.5, and 3 of
13 normal subjects did too.

## Evaluation on validation subjects

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

### On the website, without a server

The live site (https://mh51401122.vercel.app) runs the whole smear pipeline in the visitor's browser: `segment.py` itself
runs under Pyodide 0.29.5 (Python 3.13 with numpy, SciPy, scikit-image, OpenCV and Pillow), and the cell-type filter and
the MH51401122 ensemble run with ONNX Runtime Web. Checked against the Python API on one Taleqani smear:

- PNG input: identical cells, red-cell size (15.3242 px) and lymphoid flags; scores within 0.005 (0.822 / 0.952 / 0.950
  vs 0.817 / 0.949 / 0.949; same decisions), the small gap coming from OpenCV 4.11 vs 5.0 resizing and colour conversion;
- JPEG input: Pyodide's Pillow 11.3 decodes JPEG pixels slightly differently from Pillow 13, which moved the red-cell
  estimate by 5% and changed whether one cell was scored. PNG uploads give the closest match.

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

### Two fixes tried, and what they did

**A learned cell-type filter** (`src/vision/cell_type.py`): an EfficientNet-B0 trained on PBC (Acevedo et al. 2020,
CC BY 4.0; 14,744 expert-labelled white cells from a Barcelona hospital, cut out exactly as the smear reader cuts cells)
plus C-NMC training cells, to answer "lymphocyte, read it" or "other white cell, skip it". Threshold chosen to keep 98%
of C-NMC training cells and PBC validation lymphocytes; no outside smear data was used.

| Held-out check | Kept |
|---|---|
| PBC lymphocytes | 96.9% |
| PBC neutrophils, eosinophils, basophils, monocytes | 0% |
| PBC immature granulocytes / erythroblasts | 0.3% / 0.6% |
| C-NMC validation cells, normal and leukemic | 100% / 100% |
| BCCD healthy smears: cells the shape rule had let through | 16% kept (84% dropped as non-lymphoid) |
| BCCD: 8 clear lymphocytes picked by eye | **2 kept, 6 wrongly dropped** |

The filter is reliable on its own microscope and, like the ALL classifier, much less so on another lab's images: on BCCD
it drops most granulocytes but also most real lymphocytes (8 checked by eye; a small sample, but a clear signal). It is
still part of the smear pipeline and the API: when it is wrong it makes the system refuse to score a cell, which is a
safer failure than scoring a neutrophil with a lymphocyte model. What it did to the numbers is itself a finding: on Taleqani, the old model's smear
ROC-AUC fell from 0.688 to **0.573** (0.534 to 0.612) and benign smears called leukemic rose from 66.8% to 79.2%. Part of
the earlier 0.69 came from non-lymphoid cells pulling benign smears' averages down. On the cells it is meant to read,
the model barely separates ALL smears from hematogone smears.

**A "robust" retrain** (`augment: robust`, `src/gpu_augment.py`): the same five-seed EfficientNet-B0 recipe, trained on
C-NMC with heavy blur, resolution loss (to 28-160 px), colour and stain jitter and +-25% zoom, fixed before training.
Trained on a Kaggle T4 (47 min; `scripts/kaggle/build_kernel.py`, every image checked pixel-identical to the laptop
copy), calibrated and exported to `models/robust/` without touching the deployed model.

| | Old (deployed) | Robust retrain |
|---|---|---|
| C-NMC validation PR-AUC | **0.974** | 0.909 |
| C-NMC validation ROC-AUC | **0.952** | 0.850 |
| C-NMC: normal cells called leukemic at the band | **18.4%** | 40.9% |
| Taleqani smear ROC-AUC, colour-normalised + filter | **0.573** | 0.477 |
| Taleqani smear ROC-AUC, colour-normalised, no filter | **0.688** | 0.611 |
| BCCD healthy lymphocytes called leukemic (22 cells, after filter) | 100% | 100% |

The decision rule was fixed before these runs (`src/compare_models.py`): adopt the retrain for smears only if its
paired Taleqani ROC-AUC difference has a 95% interval above zero and it is no worse on BCCD. The difference was
**-0.096 (-0.150 to -0.043)**, so the old model stays everywhere. Augmentation alone did not buy robustness: the
retrain lost accuracy on clean cells and did not gain it on other labs' smears. The likely reason is that the hard part
is not blur or colour but biology and source: Taleqani's benign smears are full of hematogones, which C-NMC never shows,
and its cells come from a different camera at about a ninth of C-NMC's resolution. The next step that could actually
help is training data from more than one lab, with patient-level splits, rather than more augmentation of one lab.

### Second smear model: DinoBloom + a linear classifier (selectable on the Smear page)

Suggested by Dr Carsten Marr (Helmholtz Munich). DinoBloom-B (Koch et al., MICCAI 2024; MarrLab/DinoBloom, Apache-2.0) is
a DINOv2 ViT-B/14 trained on 13 public blood and bone-marrow datasets. It is used frozen; only a standardised L2 logistic
regression is trained on its 768-number embedding of each C-NMC training cell, with C chosen on validation PR-AUC
(`src/foundation.py`, plan fixed before results). Calibration: temperature 1.12; band 0.15 to 0.50, chosen on validation
by the same rule as the deployed model. For the website it is exported as one ONNX graph with 8-bit weight-only
quantisation (94 MB; mean probability change 0.004 against full precision on 400 validation cells; dynamic int8 was
tried first and broke it).

| | Ensemble (deployed) | DinoBloom + LR |
|---|---|---|
| C-NMC test PR-AUC (26 subjects, cells) | **0.932** | 0.858 |
| C-NMC test, healthy subjects flagged (mean cell score > 0.5) | **3/13** | 7/13 |
| Taleqani smear ROC-AUC, colour-normalised + filter | 0.573 | **0.631** |
| Taleqani smear ROC-AUC, colour-normalised, no filter | 0.688 | **0.763** |
| Taleqani benign smears called leukemic at 0.5 (+ filter) | 79% | **38%** |
| Taleqani ALL smears missed at 0.5 (+ filter) | not recorded | 48% |
| BCCD healthy lymphocytes called leukemic (22 cells, after filter) | 100% | **86%** |

Paired Taleqani ROC-AUC difference (same 2,705 smears): **+0.058 (0.016 to 0.102)**; without the filter +0.075 (0.048
to 0.102). So the frozen foundation model gives up accuracy on C-NMC and gains some on another hospital's smears, which
is what a less lab-specific representation should do. It is still far from usable: at 0.63 it misses about half the
ALL smears and still calls most healthy BCCD lymphocytes leukemic. Both models are offered on the Smear page so the
difference can be seen; the single-cell page keeps the ensemble. Results: `reports/foundation/`.

### Outside test: Johns Hopkins CellaVision cells (DeepAPL)

Data: Sidhom, ..., Shenderov, *npj Precision Oncology* 5:38 (2021), CC BY 4.0. 106 AML/APL patients, Wright stain,
CellaVision DM100, every cell labelled by a pathologist. 3,112 cells labelled "Lymphocyte" (should read normal) and
2,948 "Blast, no lineage spec" (myeloid blasts, not the B-lymphoblasts either model was trained on). Each tile went
through the website's own pipeline (`src/deepapl_eval.py`; plan fixed before results): red-cell ruler pooled over 300
tiles, the centre cell cut onto black, colours normalised to C-NMC.

| All 6,043 cells found | Ensemble | DinoBloom + LR |
|---|---|---|
| ROC-AUC, blast vs lymphocyte (95% CI, resampling patients) | 0.17 (0.13 to 0.22) | 0.25 (0.21 to 0.30) |
| Normal lymphocytes called leukemic at 0.5 | 99.0% | 94.2% |
| Blasts called leukemic at 0.5 | 98.1% | 76.0% |

**Negative result, and the strongest one so far.** Both models call almost every Hopkins lymphocyte leukemic, and both
rank the lymphocytes as *more* leukemic than the blasts (AUC well below 0.5). After the cell-type filter (which keeps
63% of lymphocytes but only 9% of blasts) the picture is the same (0.28 and 0.19). A model that learned "leukemic vs
normal" from one lab's cells has learned that lab's normal cells, not what makes a cell normal. This is the motivation
for the multi-lab training in `src/multilab.py`. Results: `reports/deepapl/`.

### Experiment: three labs, leave one lab out (blast vs normal lymphocyte)

`src/multilab.py`, plan fixed before results. Task widened to "leukemic blast vs normal lymphocyte" so that AML
datasets count. Three labs, each with both classes: C-NMC (AIIMS Delhi; B-ALL blasts 8,043, normal B-lymphocytes
4,025), Johns Hopkins (DeepAPL; blasts 2,928, lymphocytes 3,111) and LMU Munich (AML-Cytomorphology_LMU, Matek 2019,
CC BY 3.0; myeloblasts 3,261, typical lymphocytes 3,935). Hopkins and Munich tiles go through the website's crop
pipeline, so all three labs reach the model the same way. Model: frozen DinoBloom-B + logistic regression, every
(lab, class) group given equal weight. C is chosen by training on one training lab and validating on the other, never
on the held-out lab. C-NMC is only ever tested on its 26 test subjects.

| Held-out lab | Trained on | ROC-AUC (95% CI) | Normal called leukemic at 0.5 | Blasts caught at 0.5 | C-NMC-only model, same cells |
|---|---|---|---|---|---|
| Johns Hopkins | C-NMC + Munich | **0.940** (0.913 to 0.960) | 35% | 97% | 0.252, 94% normal called leukemic |
| LMU Munich | C-NMC + Hopkins | **0.831** (0.820 to 0.840) | 16% | 67% | 0.447, 95% normal called leukemic |
| C-NMC (B-ALL) | Hopkins + Munich | 0.652 (0.524 to 0.750) | 11% | 20% | n/a |

Adding a second lab with both classes changed an inverted, useless model into a usable one on the unseen lab. What it
does not show: Hopkins and Munich blasts are both myeloid, so their two folds partly test "myeloid blast vs lymphocyte"
learned from the other myeloid lab. The C-NMC fold is the honest test for ALL: trained without a single lymphoblast it
ranks B-ALL blasts above normal B-cells only modestly (0.65) and catches 20% at 0.5. B-lymphoblasts look much more like
lymphocytes than myeloblasts do; ALL needs lymphoblasts from more than one lab. Munich has no patient IDs, so its
interval resamples cells and is too narrow. Results: `reports/multilab/`.

**All three labs, tested on a fourth.** One classifier on all three labs (C chosen by mean held-out-lab PR-AUC: 0.001),
then the same Taleqani (Tehran, real ALL) and BCCD tests as every earlier model, same smears, finder, filter and crops:

| Taleqani smears, colour-normalised + filter (2,705 smears) | Deployed ensemble | DinoBloom, C-NMC only | **DinoBloom, three labs** |
|---|---|---|---|
| Smear ROC-AUC | 0.573 | 0.631 | **0.789** |
| Benign smears called leukemic at 0.5 | 79% | 38% | **6%** |
| ALL smears missed at 0.5 | | 48% | 89% |
| BCCD healthy lymphocytes called leukemic (22 cells) | 100% | 86% | **36%** |

Paired against the deployed model on the same 2,705 smears: **+0.216 (0.158 to 0.265)**. Without the filter the gain is
+0.031 (-0.008 to 0.068). This is the best outside result of any model so far, on a lab and a disease subtype (real ALL)
it never saw labelled. The ranking is good but the scale is not: at 0.5 it calls almost nothing leukemic in Tehran's
smears. A new lab needs its own threshold, set on a few of its own labelled smears, before any verdict means anything.

### Three follow-ups: a multi-lab cell filter, DinoBloom-L, and a few-shot cut-off

**Cell filter on DinoBloom features** (`src.multilab others / filter_lolo / filter_final`). Keep = lymphocytes and blasts;
drop = neutrophils, monocytes, eosinophils, basophils, erythroblasts, smudge cells, myelocytes, metamyelocytes and
immature granulocytes from Hopkins, Munich and Barcelona (PBC, CC BY 4.0), up to 500 per lab and type. Promyelocytes
(the leukemic cells of APL), promonocytes, plasma cells and atypical lymphocytes are left out of training. Cut-off: keep
>= 98% of keep cells in every training lab. Leave one lab out, against the deployed ONNX filter on the same crops:

| Unseen lab | Blasts kept | Normal lymphocytes kept | Most-passed wrong cell |
|---|---|---|---|
| Johns Hopkins | **90%** (old 9%) | **95%** (old 64%) | myelocytes 64%, smudge 46%, monocytes 34% (old under 13%) |
| LMU Munich | **89%** (old 67%) | **91%** (old 71%) | smudge 69% (13 cells), monocytes 19% (old 20%) |
| Barcelona (PBC) | n/a | 98% (old 97%) | erythroblasts 12%, immature granulocytes 11% (old under 2%) |

It keeps the cells that matter far better; the price is more monocytes, myelocytes and smudge cells let through.

**Taleqani and BCCD with the new filter** (it replaces both the shape rule and the ONNX filter; every cell off the
image edge is offered to it). Three-lab DinoBloom-B classifier:

| | Shape rule + old filter | **DinoBloom filter** |
|---|---|---|
| Taleqani smears that get a score | 2,705 | 2,942 |
| ROC-AUC on the 2,705 smears both can score, paired vs the deployed model (0.567) | 0.789 | **0.828** (+0.261, 0.214 to 0.307) |
| ROC-AUC on every smear it scores | 0.789 | 0.739 |
| BCCD healthy lymphocytes called leukemic | 36% of 22 | **26% of 53** |

Better on the same smears, and it now scores 237 more. Those extra smears (where the shape rule found no lymphoid
cell) are read poorly and pull the all-smear figure down: a smear whose only "kept" cells are borderline should be
refused, not scored. Not yet done.

**DinoBloom-L** (1,024 features) instead of -B, same leave-one-lab-out protocol: C-NMC 0.676 (B 0.652), Hopkins 0.919
(B 0.940), Munich 0.712 (B 0.831). Worse on two of three labs; the larger model is not used.

**Few-shot cut-off for a new lab.** With k of Taleqani's smears labelled (half benign, half ALL; 1,000 random draws), the
cut-off is set on those and judged on the rest (main arm, DinoBloom filter):

| Labelled smears | Sensitivity (95% range over draws) | Specificity |
|---|---|---|
| 4 | 75% (23 to 98) | 62% (32 to 86) |
| 10 | 82% (36 to 98) | 57% (30 to 80) |
| 20 | 83% (53 to 98) | 57% (33 to 75) |

Ten labelled smears turn "misses 88% of ALL smears at 0.5" into "catches about 82%", but at a specificity near 57%
and with wide luck-of-the-draw. With a ranking of 0.74 to 0.83, no cut-off is both sensitive and specific. Results:
`reports/multilab/`.

### Fine-tuning DinoBloom's last blocks (Kaggle T4): better on myeloid labs, worse on ALL

`src/finetune.py`, recipe fixed before results: last 4 of 12 blocks + final norm trainable, linear head from zero, AdamW
(1e-5 backbone, 1e-3 head), 4 epochs, cosine, fp16, flips / 90-degree turns / +-15% zoom / mild colour and blur
jitter, equal weight per (lab, class). No early stopping on any held-out lab. Run privately on Kaggle (code
`haiclop/cell-triage-dinobloom-finetune`, data a private copy of the 224 px crops), 34 GPU-minutes for four runs.

| | Fine-tuned | Frozen DinoBloom + LR |
|---|---|---|
| Leave one lab out: Johns Hopkins ROC-AUC | **0.961** | 0.940 |
| Leave one lab out: LMU Munich | **0.919** | 0.831 |
| Leave one lab out: C-NMC (B-ALL) | 0.651 | 0.652 |
| Taleqani (real ALL), DinoBloom filter, all 2,942 smears | 0.669 | **0.739** |
| Taleqani, paired on the deployed model's 2,705 smears (deployed 0.567) | 0.740 | **0.828** |
| BCCD healthy lymphocytes called leukemic (53 cells) | 26% | 26% |

Fine-tuning sharpened "myeloid blast vs lymphocyte" (the two labs that supply only myeloid blasts improved) and lost
some of what carries over to lymphoblasts in a lab it never saw. It is not used. The best model remains frozen
DinoBloom-B + three-lab logistic regression + the DinoBloom cell filter. Results: `reports/finetune/`.

## Experiment: ALL vs AML vs normal (not deployed)

`src/vision/multiclass.py`. Three classes from two public datasets: C-NMC (normal cells and B-ALL blasts) and the Munich
AML Cytomorphology set (Matek et al. 2019, TCIA, CC BY 3.0: 3,294 myeloblasts and monoblasts, 3,937 typical lymphocytes).
Munich photos were cut out with the smear cell finder and scaled so their median normal lymphocyte matches C-NMC's (the
red-cell ruler was unreliable on those 400 px fields); every cell was colour-normalised to C-NMC. One EfficientNet-B0,
15 rounds on the GTX 1650, best validation macro F1 0.878.

| Held-out test (3,501 cells) | Precision | Recall | F1 |
|---|---|---|---|
| Normal lymphocyte | 0.958 | 0.573 | 0.717 |
| ALL blast | 0.750 | 0.978 | 0.849 |
| AML blast | 0.873 | 1.000 | 0.932 |
| Macro F1 | | | 0.833 |

**It learned the lab, not the leukemia.** ALL cells came only from C-NMC and AML cells only from Munich, and the checks
show the model used that:

- normal lymphocytes' errors never cross labs: C-NMC normals were called ALL 65% of the time (AML 0.2%), Munich normals
  were called AML 12% of the time (ALL 0%);
- on a third lab, Taleqani's B-ALL smears, **0% of 2,815 cells were called ALL** and 61% were called AML;
- healthy BCCD lymphocytes were called AML 87% of the time;
- the perfect AML recall is partly Munich's random split by image (no patient IDs are published).

So the model is not used anywhere. Telling ALL from AML needs data where both come from the same lab, ideally several
labs, with patient-level splits (see `docs/data_request_guide.md`).

## Limitations

- One dataset, one acquisition setup; ALL-IDB2 access was never granted. The outside smear tests above (Taleqani, BCCD)
  show the model does not transfer to other labs.
- 92 independent subjects; most differences between models are within the confidence intervals.

## Citation

Data: Mourya S., Kant S., Kumar P., Gupta A., Gupta R. (2019). ALL Challenge dataset of ISBI 2019 (C-NMC 2019) (Version 1).
The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.dc64i46r

Weights are released under CC BY 4.0 with the data attribution above.
