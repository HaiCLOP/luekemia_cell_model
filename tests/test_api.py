"""API contract tests. They run in CI without the trained weights: a fake ONNX session returns a fixed blast logit
and cell-filter logit, the two outputs of MH51401102, so routes, validation, input checks and decision logic are all
exercised.
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
FAKE_LOGIT = 0.4  # the test recomputes the calibrated score and decision from it


class _Out:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, logit: float, keep_logit: float = 5.0):
        self.logit, self.keep_logit = logit, keep_logit

    def get_outputs(self):
        return [_Out("logit"), _Out("keep_logit")]

    def run(self, _outputs, feeds):
        x = feeds["image"]
        assert x.shape[1:] == (3, 224, 224) and x.dtype == np.float32
        n = x.shape[0]
        return [np.full(n, self.logit, np.float32), np.full(n, self.keep_logit, np.float32)]


class FakeEnsemble:
    """MH51401122's graph: five member logits and a CAM."""

    def __init__(self, logit: float):
        self.logit = logit

    def get_outputs(self):
        return [_Out("logits"), _Out("cam")]

    def run(self, _outputs, feeds):
        x = feeds["image"]
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
            "onnx_sha256": hashlib.sha256(blob).hexdigest(), "exported": "2026-01-01T00:00:00Z",
            "filter_threshold_logit": 0.0, "ood_max_abs_logit": 12.9}
    (d / "model_meta.json").write_text(json.dumps(meta))
    (d / "fake122.onnx").write_bytes(b"fake ensemble")
    meta2 = {**meta, "name": "MH51401122", "onnx_file": "fake122.onnx", "onnx_sha256": hashlib.sha256(b"fake ensemble").hexdigest()}
    del meta2["filter_threshold_logit"]
    (d / "MH51401122_meta.json").write_text(json.dumps(meta2))
    return d


@pytest.fixture(scope="module")
def client(model_dir):
    from fastapi.testclient import TestClient

    import api.main as main
    from api.inference import Model

    main.MODEL_DIR = model_dir
    orig = Model._session
    Model._session = staticmethod(lambda path, opts: FakeEnsemble(FAKE_LOGIT) if "122" in str(path) else FakeSession(FAKE_LOGIT))
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
                "latency_ms", "disclaimer"):
        assert key in j
    assert j["decision"] in {"leukemic", "normal", "refer_to_expert"}
    assert j["disclaimer"] == DISCLAIMER and j["image_stored"] is False


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


def test_second_model_can_be_chosen(client):
    r = client.post("/predict", files={"file": ("c.png", cell(), "image/png")}, data={"model": "MH51401122"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model_version"] == "MH51401122" and j["image_stored"] is False
    Image.open(io.BytesIO(base64.b64decode(j["heatmap_png_base64"]))).verify()
    assert post(client, cell()).json()["heatmap_png_base64"] is None          # the default model has no heatmap
    assert client.post("/predict", files={"file": ("c.png", cell(), "image/png")}, data={"model": "nope"}).status_code == 422
    # smears with MH51401122: lymphoid-looking cells only (the lobed one is skipped by the shape rule)
    j = client.post("/analyze-smear", files={"file": ("s.png", _smear_png(n_wbc=3, lobed=1), "image/png")},
                    data={"model": "MH51401122"}).json()
    assert j["model_version"] == "MH51401122" and j["summary"]["cells_found"] == 4
    lobed = [c for c in j["cells"] if not c["lymphoid_like"]]
    assert lobed and all(c["decision"] == "not_scored" for c in lobed)


def test_cell_the_filter_drops_gets_no_score(client):
    """A neutrophil or monocyte: the model's own cell filter says no, so there is no score."""
    import api.main as main

    sess = main.state["model"].session
    sess.keep_logit = -5.0
    try:
        r = post(client, cell())
        assert r.status_code == 422 and r.json()["code"] == "not_lymphoid" and "decision" not in r.json()
        j = client.post("/analyze-smear", files={"file": ("s.png", _smear_png(n_wbc=2), "image/png")}).json()
        assert j["summary"]["cells_scored"] == 0
        assert all("cell filter" in (c["reason"] or "") for c in j["cells"] if not c["touches_border"])
    finally:
        sess.keep_logit = 5.0


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
        pytest.skip("no ONNX weights (python -m src.multilab export_web)")
    from api.inference import Model

    m = Model(real)
    pred = m.predict(Image.open(io.BytesIO(cell())))
    assert pred.decision in {"leukemic", "normal", "refer_to_expert"} and 0 <= pred.prob_calibrated <= 1


# ------------------------------------------------------------------ /analyze-smear
def _smear_png(**kw) -> bytes:
    from tests.test_segment import smear

    img, _ = smear(**kw)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_smear_finds_and_scores_cells(client):
    r = client.post("/analyze-smear", files={"file": ("s.png", _smear_png(n_wbc=3, lobed=1), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    s = j["summary"]
    # every cell off the edge goes to the model; its own filter (here: keep everything) decides, not the shape rule
    assert s["cells_found"] == 4 and s["cells_scored"] == 4 and s["not_scored"] == 0
    assert j["scale_source"] == "red_cells_in_image" and j["rbc_diameter_px"] > 0
    assert s["smear_decision"] in {"leukemic", "normal", "refer_to_expert"}
    assert j["disclaimer"] == DISCLAIMER
    Image.open(io.BytesIO(base64.b64decode(j["overlay_png_base64"]))).verify()


def test_smear_uses_given_pixel_size(client):
    r = client.post("/analyze-smear", files={"file": ("s.png", _smear_png(n_wbc=2), "image/png")}, data={"um_per_px": "0.27"})
    j = r.json()
    assert j["scale_source"] == "um_per_px" and abs(j["rbc_diameter_px"] - 7.5 / 0.27) < 0.2


def test_smear_without_white_cells(client):
    j = client.post("/analyze-smear", files={"file": ("s.png", _smear_png(n_wbc=0), "image/png")}).json()
    assert j["summary"]["cells_found"] == 0 and j["summary"]["smear_decision"] == "no_scorable_cells"


def test_smear_rejects_tiny_and_bad_files(client):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(buf, format="PNG")
    assert client.post("/analyze-smear", files={"file": ("t.png", buf.getvalue(), "image/png")}).status_code == 422
    assert client.post("/analyze-smear", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 415


def test_out_of_distribution_logits_get_no_score(client):
    """If the network's internal scores explode (inputs unlike anything it trained on), refuse to answer."""
    import api.main as main

    sess = main.state["model"].session
    old = sess.logit
    sess.logit = 900.0
    try:
        r = post(client, cell())
        assert r.status_code == 422 and r.json()["code"] == "out_of_distribution"
        j = client.post("/analyze-smear", files={"file": ("s.png", _smear_png(n_wbc=2), "image/png")}).json()
        assert j["summary"]["cells_scored"] == 0
        assert all("out of distribution" in (c["reason"] or "") for c in j["cells"] if not c["touches_border"])
    finally:
        sess.logit = old
