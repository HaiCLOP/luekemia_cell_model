# Cell Triage: leukemia cell classifier with calibrated refer-to-expert triage

**Research demo. Not for clinical use.**

A single-cell classifier for B-lineage acute lymphoblastic leukemia (ALL) vs normal lymphocytes, built around honest
evaluation rather than headline accuracy: patient-level splits with a data audit, subject-clustered confidence intervals,
calibrated probabilities, and a referral band that hands uncertain cells to a person.

## Results (validation, 18 subjects not used for training)

| | |
|---|---|
| Deployed model | 5-seed EfficientNet-B0 ensemble, ONNX on CPU, about 45 ms per cell |
| PR-AUC | **0.974** (95% subject CI 0.906 to 0.994) |
| Calibration (ECE, 15 bins) | 0.023 deployed (temperature scaled), 0.021 before scaling |
| Missed leukemic cells | 7.55% with a 0.5 threshold, **2.74% with the referral band** while referring 8.1% of cells (nested cross-fit; the 2% target is met in-sample only) |
| Referral efficiency | 39.7 misses avoided per 100 referred cells vs 5.0 for random referral |

Full tables, ablations and failure analysis: [MODEL_CARD.md](MODEL_CARD.md), `reports/`. The test split (26 subjects) is
untouched and will be evaluated once at the end, together with the external ALL-IDB2 set.

**Whole smears and another hospital.** A cell finder (`src/vision/segment.py`) now cuts every white cell out of a stained
smear and puts it on C-NMC scale, using red cells as a ruler. It finds 99.7% of the 372 boxed white cells in BCCD. The
single-cell model behind it does not transfer: on 3,216 Taleqani Hospital smears it separates ALL from benign hematogone
smears with ROC-AUC 0.69 and calls two thirds of the benign smears leukemic. A learned cell-type filter (trained on the
PBC dataset) now keeps neutrophils and other non-lymphocytes out; on the lymphocytes that remain the ROC-AUC is 0.57. A
retrain with heavy blur, resolution and colour augmentation (5 seeds, Kaggle T4) lost accuracy on C-NMC (PR-AUC 0.909 vs
0.974) and did worse outside it (0.48 vs 0.57), so the original model stays deployed. Details and the full comparison
are in the smear section of the model card.

### Findings worth knowing

- **Data audit.** The public C-NMC mirror contains 350 corrupt files and 110 duplicates, and 4 groups of patient IDs share
  identical images (likely the same people). Splitting by raw patient ID would leak them across train and test.
- **Cell size shortcut.** Among normal cells, size correlates with the leukemia score at about 0.67 in every architecture
  tested; large normal cells cause most false alarms. Scale-jitter augmentation did not fix it.
- **GTX 16-series fp16 bug.** Mixed precision produced NaN outputs on the GTX 1650; all training uses fp32.

## Repository

```
configs/config.yaml        all settings; band and temperature written by src/triage.py
src/                       download+audit, split, data, model, train, evaluate, calibrate, triage, explain, export, report
src/vision/segment.py      smear -> white cells -> C-NMC-style crops (red cells as the ruler)
src/smear_eval.py          cell finding on BCCD, end-to-end smear evaluation on an outside ALL dataset
src/final_test.py          one-time sealed test / external evaluation
api/                       FastAPI service (ONNX Runtime), Dockerfile for Hugging Face Spaces
web/                       Next.js site (design notes in web/DESIGN_NOTES.md)
supabase/migrations/       database schema
monitor/                   local dashboard: training console, plus /jobs.html for any long job (src/progress.py)
reports/                   tables, figures, per-run JSON, validation predictions
tests/                     leakage, transforms, metrics, API contract
scripts/run_all.ps1        reproduces every number and figure
```

## Reproduce 

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install timm scikit-learn pandas matplotlib pyyaml datasets huggingface_hub torchstain grad-cam captum onnx onnxruntime fastapi uvicorn python-multipart slowapi supabase pytest
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1            # about 8 h
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -SkipTraining  # about 15 min, reuses checkpoints
```

Heavy runs can also use `notebooks/colab_train.ipynb`.

## Run the app locally

```powershell
copy .env.example .env    # fill SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY yourself, or leave empty to run without a database
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
cd web; npm install; npm run dev    # http://localhost:3000
```

API: `GET /health`, `GET /models/active`, `POST /predict` (multipart `file`, optional `store_image`), `POST /feedback`,
`POST /analyze-smear` (multipart `file`, optional `um_per_px`): finds every white cell in a smear, scores the lymphoid-like
ones and returns an annotated image. Read the smear section of [MODEL_CARD.md](MODEL_CARD.md) before trusting it: the cell
finder is reliable, the single-cell model does not transfer well to other microscopes.
Images are never stored unless the user opts in (private bucket, deleted after 24 hours); predictions log only a SHA-256.

## Licence and credit

- **Code:** MIT ([LICENSE](LICENSE)). Use it freely, including commercially; keep the copyright notice.
- **Trained weights:** CC BY 4.0. Credit "Arnav Srivastava, Cell Triage" and the C-NMC 2019 dataset.
- **How to cite:** see [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## Data

C-NMC 2019, The Cancer Imaging Archive, CC BY 3.0. Mourya S., Kant S., Kumar P., Gupta A., Gupta R. (2019).
https://doi.org/10.7937/tcia.2019.dc64i46r. See [DATA_LICENSE.md](DATA_LICENSE.md). ALL-IDB2 (external test only) is
used under its own terms and is never redistributed. No raw data or weights are committed.

Author: Arnav Srivastava, HaiCLOP Labs.
