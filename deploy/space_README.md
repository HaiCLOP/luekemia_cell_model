---
title: Cell Triage API
emoji: 🔬
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: ALL vs normal cell classifier with refer-to-expert band
---

# Cell Triage API

**Research demo. Not for clinical use.**

FastAPI service for a 5-seed EfficientNet-B0 ensemble (ONNX, CPU) that classifies a single segmented white blood cell
image as leukemic (ALL), normal, or refer to expert, with a calibrated probability and a class activation map.

- `GET /health`
- `GET /models/active`
- `POST /predict` (multipart `file`: PNG, JPEG or BMP up to 5 MB; optional `store_image`)
- `POST /feedback` (`prediction_id`, `user_label`, optional `comment`)
- Interactive docs at `/docs`

Space secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `ALLOWED_ORIGIN` (the website origin). Without the Supabase
secrets the API still predicts; feedback returns 503.

Data: C-NMC 2019, The Cancer Imaging Archive, CC BY 3.0, https://doi.org/10.7937/tcia.2019.dc64i46r
