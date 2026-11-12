"""The website runs copies of a few repo files in the browser. They must not drift from the originals."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_browser_cell_finder_is_the_real_one():
    # web/scripts/copy-ort.mjs copies it at build time; the copy is committed because Vercel builds web/ only
    assert (ROOT / "web/public/py/segment.py").read_bytes() == (ROOT / "src/vision/segment.py").read_bytes()


def test_browser_model_metadata_matches_the_deployed_model():
    web = json.loads((ROOT / "web/src/data/model_meta.json").read_text())
    real = json.loads((ROOT / "models/model_meta.json").read_text())
    for k in ("name", "onnx_sha256", "temperature", "t_low", "t_high", "ood_max_abs_logit", "input"):
        assert web[k] == real[k], k


def test_browser_colour_and_filter_settings_match():
    assert json.loads((ROOT / "web/src/data/cnmc_stain.json").read_text()) == json.loads((ROOT / "models/cnmc_stain.json").read_text())
    web = json.loads((ROOT / "web/src/data/cell_type_meta.json").read_text())
    real = json.loads((ROOT / "models/cell_type_meta.json").read_text())
    assert web["threshold"] == real["threshold"] and web["input_size"] == real["input_size"]
