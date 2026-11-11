"""The "robust" training augmentation, run on the GPU in batches.

Same recipe as build_transforms(augment="robust") in src/data.py (flips, 90-degree turns, affine with
+-15 deg / 5% shift / x0.75-1.25 zoom, colour jitter, random loss of resolution, Gaussian blur, HED stain
jitter), but on the GPU: on the CPU the blur and resolution steps fed only ~77 img/s and, with ten loader
workers, used ~6 GB of RAM. Training images are decoded once into memory (7,844 x 224 x 224 x 3 bytes,
about 1.2 GB) and cached on disk for the next seed.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CACHE = Path(os.environ.get("CNMC_UINT8_CACHE", "E:/cache/cnmc_train_224_uint8.npz"))


def load_uint8(cfg, df) -> torch.Tensor:
    """All training images as uint8 [N, 3, 224, 224], resized exactly like the CPU transform (antialiased)."""
    import torchvision.transforms.v2.functional as TF

    size = cfg["data"]["img_size"]
    import hashlib

    key = np.array([int(hashlib.sha1("|".join(df.image_path).encode()).hexdigest()[:15], 16)], dtype=np.int64)
    if CACHE.exists():
        d = np.load(CACHE)
        if d["key"][0] == key[0] and len(d["x"]) == len(df):
            return torch.from_numpy(d["x"])
    root = Path(cfg["paths"]["data_root"])
    out = np.empty((len(df), 3, size, size), np.uint8)
    for i, p in enumerate(df.image_path):
        t = TF.pil_to_tensor(Image.open(root / p).convert("RGB"))
        out[i] = TF.resize(t, [size, size], antialias=True).numpy()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, x=out, key=key)
    return torch.from_numpy(out)


def _affine(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, generator=g) * 30 - 15) * math.pi / 180
    sc = torch.rand(b, generator=g) * 0.5 + 0.75
    tx, ty = (torch.rand(b, generator=g) * 0.1 - 0.05) * 2, (torch.rand(b, generator=g) * 0.1 - 0.05) * 2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc          # inverse map: output -> input coordinates
    theta = torch.stack([torch.stack([cos, -sin, -tx], 1), torch.stack([sin, cos, -ty], 1)], 1).to(dev)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]


def _colour(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    b, dev = x.shape[0], x.device
    u = lambda lo, hi: (torch.rand(b, 1, 1, 1, generator=g) * (hi - lo) + lo).to(dev)  # noqa: E731
    x = (x * u(0.75, 1.25)).clamp(0, 1)                                              # brightness
    m = _gray(x).mean(dim=(2, 3), keepdim=True)
    x = ((x - m) * u(0.75, 1.25) + m).clamp(0, 1)                                    # contrast
    gr = _gray(x)
    x = ((x - gr) * u(0.7, 1.3) + gr).clamp(0, 1)                                    # saturation
    # hue: rotate chroma in YIQ space by up to +-0.04 of a turn
    h = (torch.rand(b, generator=g) * 0.08 - 0.04).to(dev) * 2 * math.pi
    yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    rgb = torch.linalg.inv(yiq)
    c, s = torch.cos(h), torch.sin(h)
    rot = torch.zeros(b, 3, 3, device=dev)
    rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = c, -s, s, c
    mtx = rgb @ rot @ yiq                                                            # [b, 3, 3]
    return torch.einsum("bij,bjhw->bihw", mtx, x).clamp(0, 1)


def _lowres(x: torch.Tensor, g: torch.Generator, p=0.7, lo=28, hi=160) -> torch.Tensor:
    h, w = x.shape[-2:]
    out = x.clone()
    for i in torch.nonzero(torch.rand(x.shape[0], generator=g) < p).flatten().tolist():
        side = int(torch.randint(lo, hi + 1, (1,), generator=g))
        small = F.interpolate(x[i:i + 1], size=(side, side), mode="bilinear", antialias=True, align_corners=False)
        out[i:i + 1] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return out.clamp(0, 1)


def _blur(x: torch.Tensor, g: torch.Generator, p=0.5, k=9) -> torch.Tensor:
    out = x.clone()
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    for i in torch.nonzero(torch.rand(x.shape[0], generator=g) < p).flatten().tolist():
        sigma = float(torch.rand(1, generator=g) * 2.0 + 0.5)
        ker = torch.exp(-(r ** 2) / (2 * sigma ** 2))
        ker = ker / ker.sum()
        xi = F.pad(x[i:i + 1], (k // 2,) * 4, mode="reflect")
        xi = F.conv2d(xi, ker.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
        out[i:i + 1] = F.conv2d(xi, ker.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    return out


_RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])


def _hed(x: torch.Tensor, g: torch.Generator, sigma=0.03, bias=0.03) -> torch.Tensor:
    """HEDJitter from src/data.py, batched: per image, scale and shift each stain channel in optical density."""
    b, dev = x.shape[0], x.device
    R = _RGB_FROM_HED.to(dev)
    M = torch.linalg.inv(R)
    od = -torch.log(x.clamp(1e-6, 1))
    hed = torch.einsum("bjhw,ji->bihw", od, M)
    alpha = (1 + (torch.rand(b, 3, generator=g) * 2 - 1) * sigma).to(dev).view(b, 3, 1, 1)
    beta = ((torch.rand(b, 3, generator=g) * 2 - 1) * bias).to(dev).view(b, 3, 1, 1)
    hed = hed * alpha + beta
    return torch.exp(-torch.einsum("bjhw,ji->bihw", hed, R)).clamp(0, 1)


class GpuAugLoader:
    """Drop-in for the training DataLoader: yields (augmented, normalised float batch on GPU, labels)."""

    def __init__(self, cfg, df, device, seed: int):
        self.x = load_uint8(cfg, df)
        self.y = torch.from_numpy(df["label"].to_numpy().astype(np.float32))
        self.bs, self.dev = cfg["train"]["batch_size"], device
        self.g = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
        self.dataset = type("D", (), {"labels": df["label"].to_numpy()})()

    def __len__(self):
        return len(self.x) // self.bs

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        g = self.g
        b = x.shape[0]
        hf = (torch.rand(b, generator=g) < 0.5).to(self.dev).view(b, 1, 1, 1)
        x = torch.where(hf, x.flip(-1), x)
        vf = (torch.rand(b, generator=g) < 0.5).to(self.dev).view(b, 1, 1, 1)
        x = torch.where(vf, x.flip(-2), x)
        k = torch.randint(0, 4, (b,), generator=g)
        x = torch.stack([torch.rot90(x[i], int(k[i]), dims=(-2, -1)) for i in range(b)])
        x = _affine(x, g)
        x = _colour(x, g)
        x = _lowres(x, g)
        x = _blur(x, g)
        x = _hed(x, g)
        return (x - self.mean) / self.std

    def __iter__(self):
        idx = torch.randperm(len(self.x), generator=self.g)
        for i in range(len(self)):
            j = idx[i * self.bs:(i + 1) * self.bs]
            xb = self.x[j].to(self.dev, non_blocking=True).float() / 255.0
            yield self.augment(xb), self.y[j]
