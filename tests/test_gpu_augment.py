"""The batched "robust" augmentation (src/gpu_augment.py), checked on CPU tensors."""
import torch

from src.gpu_augment import _affine, _blur, _colour, _hed, _lowres


def batch(b=6, s=64):
    x = torch.zeros(b, 3, s, s)
    yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
    disk = ((yy - s / 2) ** 2 + (xx - s / 2) ** 2) < (s / 4) ** 2
    x[:, 0][:, disk], x[:, 1][:, disk], x[:, 2][:, disk] = 0.45, 0.2, 0.5
    return x


def test_shapes_and_range():
    g = torch.Generator().manual_seed(0)
    x = batch()
    for f in (_affine, _colour, _lowres, _blur, _hed):
        y = f(x, g)
        assert y.shape == x.shape
        assert float(y.min()) >= 0 and float(y.max()) <= 1


def test_black_background_stays_dark():
    g = torch.Generator().manual_seed(1)
    x = batch()
    for f in (_affine, _lowres, _blur, _hed):
        y = f(x, g)
        assert float(y[:, :, :4, :4].max()) < 0.05      # a corner far from the cell


def test_reproducible_with_seed():
    a = _lowres(batch(), torch.Generator().manual_seed(3))
    b = _lowres(batch(), torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_lowres_and_blur_soften_edges():
    g = torch.Generator().manual_seed(4)
    x = batch(b=1)
    soft = _blur(_lowres(x, g, p=1.0, lo=16, hi=16), g, p=1.0)
    grad = lambda t: (t[..., 1:, :] - t[..., :-1, :]).abs().max()  # noqa: E731
    assert grad(soft) < grad(x)
