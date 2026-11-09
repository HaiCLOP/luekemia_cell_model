import numpy as np
import torch
from PIL import Image

from src.data import HEDJitter, build_transforms, parse_cnmc_name
from src.utils import load_config


def _img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (450, 450, 3), dtype=np.uint8))


def test_eval_transform_shape_dtype_deterministic():
    cfg = load_config()
    tf = build_transforms(cfg, train=False)
    a, b = tf(_img()), tf(_img())
    assert a.shape == (3, 224, 224) and a.dtype == torch.float32
    assert torch.equal(a, b)


def test_train_transform_shape():
    cfg = load_config()
    for aug in ("none", "geometric", "geometric_hed", "geometric_scale"):
        x = build_transforms(cfg, train=True, augment=aug)(_img())
        assert x.shape == (3, 224, 224) and torch.isfinite(x).all()


def test_hed_jitter_identity_when_zero():
    x = torch.rand(3, 32, 32) * 0.9 + 0.05
    y = HEDJitter(sigma=0, bias=0)(x)
    assert torch.allclose(x, y, atol=1e-4)


def test_parse_cnmc_name():
    assert parse_cnmc_name("UID_11_10_1_all.bmp") == ("11", 1)
    assert parse_cnmc_name("UID_H10_3_2_hem.bmp") == ("H10", 0)
    import pytest

    with pytest.raises(ValueError):
        parse_cnmc_name("random.bmp")
