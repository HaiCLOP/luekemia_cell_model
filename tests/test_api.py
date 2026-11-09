"""API contract tests. They run without a database (SUPABASE_* unset)."""
import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)
os.environ["RATE_LIMIT"] = "1000/minute"
os.environ["API_SKIP_DOTENV"] = "1"  # tests run without the database even if a local .env exists

from pathlib import Path  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(not list(MODEL_DIR.glob("*.onnx")), reason="no ONNX weights (run python -m src.export)")

DISCLAIMER = "Research demo. Not for clinical use."


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def _png(size=(450, 450), fmt="PNG") -> bytes:
    rng = np.random.default_rng(0)
    arr = np.zeros((*size, 3), dtype=np.uint8)
    yy, xx = np.ogrid[: size[0], : size[1]]
    disk = (yy - size[0] / 2) ** 2 + (xx - size[1] / 2) ** 2 < (size[0] / 4) ** 2
    arr[disk] = rng.integers(60, 140, (disk.sum(), 3))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["disclaimer"] == DISCLAIMER
    assert r.json()["database"] == "disabled"


def test_model_card(client):
    r = client.get("/models/active").json()
    assert 0 <= r["band"]["t_low"] <= r["band"]["t_high"] <= 1
    assert r["temperature"] > 0 and r["disclaimer"] == DISCLAIMER


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP"])
def test_predict_valid_image_returns_schema(client, fmt):
    r = client.post("/predict", files={"file": (f"cell.{fmt.lower()}", _png(fmt=fmt), "application/octet-stream")})
    assert r.status_code == 200, r.text
    j = r.json()
    for key in ("prediction_id", "model_version", "prob_raw", "prob_calibrated", "decision", "band",
                "heatmap_png_base64", "latency_ms", "disclaimer"):
        assert key in j
    assert j["decision"] in {"leukemic", "normal", "refer_to_expert"}
    assert 0 <= j["prob_raw"] <= 1 and 0 <= j["prob_calibrated"] <= 1
    assert j["disclaimer"] == DISCLAIMER and j["image_stored"] is False
    import base64

    Image.open(io.BytesIO(base64.b64decode(j["heatmap_png_base64"]))).verify()


def test_decision_consistent_with_band(client):
    j = client.post("/predict", files={"file": ("c.png", _png(), "image/png")}).json()
    p, b = j["prob_calibrated"], j["band"]
    expected = "normal" if p < b["t_low"] else "leukemic" if p > b["t_high"] else "refer_to_expert"
    assert j["decision"] == expected


def test_oversized_file_rejected(client):
    big = b"\x89PNG" + os.urandom(5 * 1024 * 1024 + 10)
    r = client.post("/predict", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413 and r.json()["disclaimer"] == DISCLAIMER


def test_wrong_type_rejected(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image at all", "text/plain")})
    assert r.status_code == 415 and r.json()["disclaimer"] == DISCLAIMER
    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(buf, format="GIF")
    r = client.post("/predict", files={"file": ("x.gif", buf.getvalue(), "image/gif")})
    assert r.status_code == 415


def test_missing_file_is_4xx_with_disclaimer(client):
    r = client.post("/predict")
    assert r.status_code == 422 and r.json()["disclaimer"] == DISCLAIMER


def test_feedback_validation_and_no_db(client):
    r = client.post("/feedback", json={"prediction_id": "x", "user_label": "maybe"})
    assert r.status_code == 422 and r.json()["disclaimer"] == DISCLAIMER
    r = client.post("/feedback", json={"prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                                       "user_label": "normal", "comment": "a" * 501})
    assert r.status_code == 422
    r = client.post("/feedback", json={"prediction_id": "123e4567-e89b-12d3-a456-426614174000", "user_label": "normal"})
    assert r.status_code == 503 and r.json()["disclaimer"] == DISCLAIMER
