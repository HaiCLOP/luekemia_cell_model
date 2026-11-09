# Explainability findings (validation only)

Model explained: EfficientNet-B0 seed 42. Case categories use the deployed rule (5-seed ensemble, T = 1.0728, band [0.20, 0.50]).
Faithfulness sample: 200 validation cells (100 per class). Figures: `reports/figures/gallery_*.png` and `faithfulness.png`.

## 1. The model attends to the cell, not the background
- Grad-CAM puts a median **2.46x** more saliency on the cell than its area would predict (cells cover 16% of the image).
- 17 of 200 maps (8.5%) are not concentrated on the cell (enrichment < 1). Listed in `reports/tables/explain_flagged_background_focus.csv`.
- Flagged maps are more often wrong (76.5% correct vs 88.5% overall), and wrong predictions have lower enrichment (median 2.01 vs 2.55).
- Visual review of the galleries: errors show attention on the cell border, on detached fragments (H49) or on the black background (one H9 cell, G_22 and G_44 false negatives).

## 2. Saliency maps localise the cell, but not features inside it
- Insertion AUC: Grad-CAM 0.860 vs revealing random cell pixels 0.868 vs random pixels anywhere 0.531.
- So Grad-CAM is faithful at the level of "the cell matters", but no better than random at ranking pixels within the cell.
- Integrated Gradients with a black baseline is weak here (insertion 0.650): the baseline equals the background, so attributions are speckle over the cell.
- Pixel-level deletion is unreliable on this data: blacking out even 5% of random pixels creates out-of-distribution noise and drops every curve to about 0.53. Deletion AUCs (Grad-CAM 0.491, random-on-cell 0.508, random 0.538, IG 0.548) should be read with that caveat. Patch-level or blur-baseline deletion would be a better protocol.

## 3. Cell size is a dominant cue (likely shortcut)
- Among **normal** validation cells, Spearman correlation between cell area and P(leukemic) is **0.70** (n = 604, p < 1e-90). Among leukemic cells it is 0.10.
- Normal cells that trigger false alarms are as large as leukemic cells (mean area fraction 0.181 vs 0.178); correctly classified normal cells average 0.130.
- The three hard normal subjects (H34, H9, H46) have the largest normal cells in validation (0.198, 0.181, 0.177) and the highest mean P(leukemic) (0.70, 0.51, 0.44).
- Lymphoblasts are genuinely larger, so size is partly a real biological signal, but the model over-relies on it. Size also depends on magnification and acquisition, so this cue is a specific risk for the external (ALL-IDB2) evaluation.

## 4. Mitigation attempt: scale-jitter augmentation (negative result)
Run `abl_scalejit_s42`: main recipe with zoom widened from +-10% to +-25% (cell area x0.56 to x1.56). Table: `reports/tables/size_reliance.csv`.

| | Val PR-AUC | ECE | rho(size, p) normal cells | False alarms @0.5 | Mean p, H34 |
|---|---|---|---|---|---|
| B0, 5 seeds (range) | 0.967 to 0.972 | 0.026 to 0.043 | 0.627 to 0.722 (mean 0.673) | 15.1% to 19.7% | 0.67 to 0.75 |
| B0 + scale jitter | 0.967 | 0.047 | 0.619 | 17.9% | 0.73 |

- Size reliance moved only from about 0.67 to 0.62, just below the seed range: not a meaningful reduction from one run.
- The hard subject H34 is still classified leukemic (0.73). Accuracy is unchanged within seed noise.
- Size reliance is present in every architecture tested (rho 0.53 to 0.78), so it reflects the data, not one model.
- Likely reasons: resizing the whole image also rescales texture and nuclear detail, so relative cues survive; and the size difference between classes is real biology, so the model may be right to use some of it. A stronger test would hold out size explicitly (size-matched evaluation subsets) or normalise each cell to a fixed diameter.

## Implications for the paper
- Report size reliance as the main failure mode; it explains the subject-level error on H34.
- Candidate mitigation for future work: scale-jitter augmentation or size-normalised crops, evaluated by the size-p correlation among normal cells and by external performance.
