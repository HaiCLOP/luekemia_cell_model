"""API contract tests. They run in CI without the trained weights: a fake ONNX session returns fixed member
logits and a CAM of the right shape, so routes, validation, input checks and decision logic are all exercised.
No database (SUPABASE_* unset)."""
import base64
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)
os.environ["RATE_LIMIT"] = "1000/minute"
os.environ["API_SKIP_DOTENV"] = "1"

DISCLAIMER = "Research demo. Not for clinical use."
FAKE_LOGIT = 0.4  # same logit from every member; the test recomputes the calibrated score and decision


class FakeSession:
    def __init__(self, logit: float):
        self.logit = logit

    def run(self, _outputs, feeds):
        x = feeds["image"]
        assert x.shape[1:] == (3, 224, 224) and x.dtype == np.float32
        cam = np.zeros((x.shape[0], 7, 7), np.float32)
        cam[:, 3, 3] = 1.0
        return [np.full((x.shape[0], 5), self.logit, np.float32), cam]


@pytest.fixture(scope="module")
def model_dir():
    d = Path(tempfile.mkdtemp())
    blob = b"fake onnx bytes"
    (d / "fake.onnx").write_bytes(blob)
    meta = {"name": "fake-v0", "architecture": "fake", "input": {"size": 224, "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]}, "temperature": 1.0728, "t_low": 0.2, "t_high": 0.5,
            "band_target": "test", "metrics": {}, "onnx_file": "fake.onnx",
            "onnx_sha256": hashlib.sha256(blob).hexdigest(), "exported": "2026-01-01T00:00:00Z"}
    (d / "model_meta.json").write_text(json.dumps(meta))
    return d


@pytest.fixture(scope="module")
def client(model_dir):
    from fastapi.testclient import TestClient

    import api.main as main
    from api.inference import Model

    main.MODEL_DIR = model_dir
    orig = Model._session
    Model._session = staticmethod(lambda path, opts: FakeSession(FAKE_LOGIT))
    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        Model._session = orig


def cell(size=(450, 450), fmt="PNG", radius=0.25, two=False, background=0) -> bytes:
    """A synthetic cell: textured disk on black, like a C-NMC crop."""
    rng = np.random.default_rng(0)
    arr = np.full((*size, 3), background, dtype=np.uint8)
    yy, xx = np.ogrid[: size[0], : size[1]]
    centres = [(size[0] * 0.3, size[1] * 0.3), (size[0] * 0.7, size[1] * 0.7)] if two else [(size[0] / 2, size[1] / 2)]
    r = min(size) * (radius if not two else 0.12)
    for cy, cx in centres:
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        arr[disk] = rng.integers(60, 200, (disk.sum(), 3))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def post(client, data, name="c.png"):
    return client.post("/predict", files={"file": (name, data, "application/octet-stream")})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["disclaimer"] == DISCLAIMER
    assert r.json()["database"] == "disabled" and r.json()["model_version"] == "fake-v0"


def test_model_card(client):
    r = client.get("/models/active").json()
    assert 0 <= r["band"]["t_low"] <= r["band"]["t_high"] <= 1
    assert r["temperature"] > 0 and r["disclaimer"] == DISCLAIMER


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP"])
def test_predict_valid_cell_returns_schema(client, fmt):
    r = post(client, cell(fmt=fmt), f"cell.{fmt.lower()}")
    assert r.status_code == 200, r.text
    j = r.json()
    for key in ("prediction_id", "model_version", "prob_raw", "prob_calibrated", "decision", "band",
                "heatmap_png_base64", "latency_ms", "disclaimer"):
        assert key in j
    assert j["decision"] in {"leukemic", "normal", "refer_to_expert"}
    assert j["disclaimer"] == DISCLAIMER and j["image_stored"] is False
    Image.open(io.BytesIO(base64.b64decode(j["heatmap_png_base64"]))).verify()


def test_calibration_and_decision_match_meta(client):
    j = post(client, cell()).json()
    p_raw = 1 / (1 + np.exp(-FAKE_LOGIT))
    p_cal = 1 / (1 + np.exp(-np.log(p_raw / (1 - p_raw)) / 1.0728))
    assert abs(j["prob_raw"] - p_raw) < 1e-4 and abs(j["prob_calibrated"] - p_cal) < 1e-4
    b = j["band"]
    expected = "normal" if p_cal < b["t_low"] else "leukemic" if p_cal > b["t_high"] else "refer_to_expert"
    assert j["decision"] == expected


@pytest.mark.parametrize("data,code", [
    (lambda: cell(size=(300, 600)), "unsupported_input"),             # not square
    (lambda: cell(background=200), "unsupported_input"),              # bright background: a photo or smear, not a crop
    (lambda: cell(two=True), "unsupported_input"),                    # two objects
    (lambda: cell(radius=0.02), "insufficient_image_quality"),        # practically empty
])
def test_out_of_contract_images_get_no_score(client, data, code):
    r = post(client, data())
    assert r.status_code == 422, r.text
    j = r.json()
    assert j["code"] == code and "decision" not in j and j["disclaimer"] == DISCLAIMER


def test_flat_blob_is_rejected(client):
    arr = np.zeros((300, 300, 3), np.uint8)
    yy, xx = np.ogrid[:300, :300]
    arr[(yy - 150) ** 2 + (xx - 150) ** 2 < 70 ** 2] = 120
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    r = post(client, buf.getvalue())
    assert r.status_code == 422 and r.json()["code"] == "insufficient_image_quality"


def test_oversized_file_rejected(client):
    big = b"\x89PNG" + os.urandom(5 * 1024 * 1024 + 10)
    r = post(client, big, "big.png")
    assert r.status_code == 413 and r.json()["disclaimer"] == DISCLAIMER


def test_wrong_type_rejected(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image at all", "text/plain")})
    assert r.status_code == 415 and r.json()["disclaimer"] == DISCLAIMER
    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(buf, format="GIF")
    assert post(client, buf.getvalue(), "x.gif").status_code == 415


def test_missing_file_is_4xx_with_disclaimer(client):
    r = client.post("/predict")
    assert r.status_code == 422 and r.json()["disclaimer"] == DISCLAIMER


def test_feedback_validation_and_no_db(client):
    r = client.post("/feedback", json={"prediction_id": "x", "user_label": "maybe"})
    assert r.status_code == 422 and r.json()["disclaimer"] == DISCLAIMER
    pid = "123e4567-e89b-12d3-a456-426614174000"
    assert client.post("/feedback", json={"prediction_id": pid, "user_label": "normal", "comment": "a" * 501}).status_code == 422
    r = client.post("/feedback", json={"prediction_id": pid, "user_label": "normal"})
    assert r.status_code == 503 and r.json()["disclaimer"] == DISCLAIMER


def test_rate_limit_key_ignores_spoofed_forwarded_prefix(monkeypatch):
    from starlette.requests import Request

    from api.main import client_key

    monkeypatch.setenv("TRUST_PROXY", "1")
    req = Request({"type": "http", "headers": [(b"x-forwarded-for", b"6.6.6.6, 10.0.0.7")], "client": ("1.1.1.1", 1)})
    assert client_key(req) == "10.0.0.7"


def test_checksum_mismatch_refuses_to_start(model_dir, tmp_path):
    from api.inference import Model

    for f in model_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    (tmp_path / "fake.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        Model(tmp_path)


def test_real_weights_if_present():
    """Optional smoke test against the real exported model (skipped in CI, which has no weights)."""
    real = Path(__file__).resolve().parents[1] / "models"
    if not list(real.glob("*.onnx")):
        pytest.skip("no ONNX weights (run python -m src.export)")
    from api.inference import Model

    m = Model(real)
    pred = m.predict(Image.open(io.BytesIO(cell())))
    assert pred.decision in {"leukemic", "normal", "refer_to_expert"} and len(pred.member_probs) == 5
