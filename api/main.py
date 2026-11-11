"""FastAPI inference service for the leukemia cell classifier. Research demo. Not for clinical use.

Run locally:  .venv\\Scripts\\python.exe -m uvicorn api.main:app --port 8000
Env: MODEL_DIR (default ./models), ALLOWED_ORIGIN (comma-separated), SUPABASE_URL,
     SUPABASE_SERVICE_ROLE_KEY, RATE_LIMIT (default 20/minute), TRUST_PROXY (1 behind a proxy such as HF Spaces)
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.db import Database
from api.inference import Model
from api.quality import check as check_input
from api.schemas import (DISCLAIMER, Band, FeedbackRequest, FeedbackResponse, HealthResponse, ModelCard,
                         PredictResponse, SmearResponse)
from api.smear import analyse_smear

def _load_dotenv() -> None:
    """Read KEY=VALUE lines from the repo-root .env (local dev only; real deployments set env vars)."""
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


if os.getenv("API_SKIP_DOTENV") != "1":
    _load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

MAX_BYTES = 5 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP"}
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")
_infer_slots = asyncio.Semaphore(int(os.getenv("INFER_CONCURRENCY", "2")))


def client_key(request: Request) -> str:
    """Rate-limit key. Behind a proxy (TRUST_PROXY=1) use the RIGHT-most X-Forwarded-For entry: that one is
    appended by our own proxy, while anything to its left is whatever the client chose to send."""
    if os.getenv("TRUST_PROXY") == "1":
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[-1].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=client_key)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = Model(MODEL_DIR)
    state["db"] = Database()
    try:
        state["db"].ensure_model_version(state["model"].meta)
    except Exception as e:
        log.error("could not register model version: %s", e)
        state["db"].status = "error"
    log.info("model %s loaded, database %s", state["model"].meta["name"], state["db"].status)
    yield


app = FastAPI(title="Leukemia Cell Classifier API", version="1.0.0",
              description=f"ALL vs normal single-cell classifier with calibrated refer-to-expert triage. {DISCLAIMER}",
              lifespan=lifespan)
app.state.limiter = limiter
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGIN", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


# --------------------------------------------------------------------- errors always carry the disclaimer
class InputRejected(Exception):
    """The image is outside the model's input contract. No score is produced for it."""

    def __init__(self, code: str, reason: str):
        self.code, self.reason = code, reason


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "disclaimer": DISCLAIMER})


@app.exception_handler(InputRejected)
async def input_rejected(_: Request, exc: InputRejected):
    return JSONResponse(status_code=422, content={"detail": exc.reason, "code": exc.code, "disclaimer": DISCLAIMER})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    errs = [{"loc": e.get("loc"), "msg": e.get("msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errs, "disclaimer": DISCLAIMER})


@app.exception_handler(RateLimitExceeded)
async def rate_limited(_: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}", "disclaimer": DISCLAIMER})


# --------------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_version=state["model"].meta["name"], database=state["db"].status)


@app.get("/models/active", response_model=ModelCard)
def active_model():
    m = state["model"].meta
    return ModelCard(name=m["name"], architecture=m["architecture"], temperature=m["temperature"],
                     band=Band(t_low=m["t_low"], t_high=m["t_high"]), band_target=m["band_target"],
                     metrics=m["metrics"], input=m["input"], exported=m["exported"])


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(RATE_LIMIT)
async def predict(request: Request, file: UploadFile = File(...), store_image: bool = Form(False),
                  session_id: str | None = Form(None, max_length=64)):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 5 MB.")
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, "Unsupported or corrupt image. Use PNG, JPEG or BMP.")
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, f"Unsupported image type {fmt}. Use PNG, JPEG or BMP.")
    if min(img.size) < 32:
        raise InputRejected("insufficient_image_quality", "Image is too small (minimum 32 x 32 px).")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    qc = check_input(img)
    if not qc.ok:
        raise InputRejected(qc.code, qc.reason)

    # ONNX Runtime is blocking CPU work; keep it off the event loop. The semaphore bounds concurrent runs.
    async with _infer_slots:
        pred = await run_in_threadpool(state["model"].predict, img)
    if pred.max_abs_logit > state["model"].ood_logit:
        raise InputRejected("out_of_distribution",
                            "The model's internal scores for this image are far outside anything it produced on its training "
                            "data, so any answer would be meaningless. Usually the stain colours, magnification or image "
                            "source differ from C-NMC.")
    sha = hashlib.sha256(data).hexdigest()
    db: Database = state["db"]
    prediction_id = db.log_prediction(sha, pred, session_id)
    stored = db.store_image(prediction_id, data, fmt) if store_image else False
    m = state["model"]
    return PredictResponse(prediction_id=prediction_id, model_version=m.meta["name"], prob_raw=round(pred.prob_raw, 4),
                           prob_calibrated=round(pred.prob_calibrated, 4), decision=pred.decision,
                           band=Band(t_low=m.t_low, t_high=m.t_high), heatmap_png_base64=pred.heatmap_png_base64,
                           latency_ms=pred.latency_ms, image_stored=stored)


SMEAR_MAX_BYTES = 10 * 1024 * 1024


@app.post("/analyze-smear", response_model=SmearResponse)
@limiter.limit(os.getenv("SMEAR_RATE_LIMIT", "10/minute"))
async def analyze_smear(request: Request, file: UploadFile = File(...),
                        um_per_px: float | None = Form(None, gt=0.01, lt=5.0),
                        rbc_diameter_px: float | None = Form(None, gt=4, lt=2000)):
    """Find white cells in a stained peripheral-blood smear image, score each lymphoid-like cell with the
    single-cell model after putting it on C-NMC scale, and summarise. Nothing is stored."""
    data = await file.read(SMEAR_MAX_BYTES + 1)
    if len(data) > SMEAR_MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
        img = ImageOps.exif_transpose(img)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, "Unsupported or corrupt image. Use PNG, JPEG or BMP.")
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, f"Unsupported image type {fmt}. Use PNG, JPEG or BMP.")
    if min(img.size) < 128:
        raise InputRejected("insufficient_image_quality", "A smear image should be at least 128 x 128 px.")
    async with _infer_slots:
        out = await run_in_threadpool(analyse_smear, state["model"], img, um_per_px, rbc_diameter_px)
    return out


@app.post("/feedback", response_model=FeedbackResponse)
@limiter.limit("30/minute")
def feedback(request: Request, body: FeedbackRequest):
    db: Database = state["db"]
    if not db.enabled:
        raise HTTPException(503, "Feedback storage is not configured on this server.")
    if not db.prediction_exists(body.prediction_id):
        raise HTTPException(404, "Unknown prediction_id.")
    fid = db.add_feedback(body.prediction_id, body.user_label, body.comment)
    return FeedbackResponse(status="recorded", feedback_id=fid)
