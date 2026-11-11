# CPU inference image for the leukemia cell classifier API (Hugging Face Spaces, Docker SDK, port 7860).
# Build from the repository root:  docker build -f api/Dockerfile -t leukemia-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models PORT=7860 TRUST_PROXY=1

# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 app
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

COPY api/ api/
# the cell finder used by /analyze-smear
COPY src/vision/ src/vision/
# Weights are not in git. Copy them into models/ before building, or fetch them from the HF model repo.
COPY models/effb0-ens5-v1.onnx models/model_meta.json models/cnmc_stain.json models/

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health',timeout=4)"
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
