"""Dataset, transforms and patient-ID parsing for C-NMC 2019."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.utils import resolve

# C-NMC training filenames look like UID_11_10_1_all.bmp or UID_H10_3_2_hem.bmp
CNMC_NAME = re.compile(r"^UID_(?P<patient>H?\d+)_(?P<img>\d+)_(?P<cell>\d+)_(?P<cls>all|hem)\.bmp$", re.IGNORECASE)


def parse_cnmc_name(name: str) -> tuple[str, int]:
    """Return (patient_id, label) from a C-NMC filename. Raise on anything unexpected."""
    m = CNMC_NAME.match(Path(name).name)
    if not m:
        raise ValueError(f"Filename does not match the C-NMC pattern: {name!r}")
    return m["patient"].upper(), int(m["cls"].lower() == "all")


# ---------------------------------------------------------------- transforms
class HEDJitter(torch.nn.Module):
    """Stain augmentation in HED space (Tellez et al. 2018). Input: float tensor in [0,1], CxHxW."""

    def __init__(self, sigma: float = 0.03, bias: float = 0.03):
        super().__init__()
        self.sigma, self.bias = sigma, bias
        rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
        self.register_buffer("rgb_from_hed", rgb_from_hed)
        self.register_buffer("hed_from_rgb", torch.linalg.inv(rgb_from_hed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape
        rgb = x.clamp(1e-6, 1).reshape(3, -1).T
        od = -torch.log(rgb)
        hed = od @ self.hed_from_rgb
        alpha = 1 + (torch.rand(3) * 2 - 1) * self.sigma
        beta = (torch.rand(3) * 2 - 1) * self.bias
        hed = hed * alpha + beta
        rgb = torch.exp(-(hed @ self.rgb_from_hed))
        return rgb.T.reshape(c, h, w).clamp(0, 1)


class RandomRot90(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))


def build_transforms(cfg: dict, train: bool, augment: str | None = None) -> T.Compose:
    d = cfg["data"]
    size = d["img_size"]
    augment = augment if augment is not None else cfg["train"]["augment"]
    base = [T.ToImage(), T.Resize((size, size), antialias=True), T.ToDtype(torch.float32, scale=True)]
    aug: list = []
    if train and augment in ("geometric", "geometric_hed", "geometric_scale"):
        # geometric_scale widens zoom to +-25% (cell area x0.56 to x1.56) so absolute cell size, which the
        # model leaned on heavily in the explainability analysis, becomes a weaker cue. Normal vs leukemic area differs by about x1.37.
        scale = (0.75, 1.25) if augment == "geometric_scale" else (0.9, 1.1)
        aug += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRot90(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=scale),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    if train and augment == "geometric_hed":
        aug.append(HEDJitter())
    return T.Compose(base + aug + [T.Normalize(d["mean"], d["std"])])


# ---------------------------------------------------------------- dataset
class CellDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str | Path, transform):
        self.paths = [Path(data_root) / p for p in df["image_path"]]
        self.labels = df["label"].to_numpy().astype(np.int64)
        self.patients = df["patient_id"].astype(str).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def load_split(cfg: dict, split: str | list[str], fold: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(resolve(cfg["paths"]["splits_csv"]))
    splits = [split] if isinstance(split, str) else split
    df = df[df["split"].isin(splits)]
    if fold is not None:
        df = df[df["fold"] == fold]
    return df.reset_index(drop=True)


def make_loader(cfg: dict, df: pd.DataFrame, train: bool, augment: str | None = None,
                batch_size: int | None = None) -> DataLoader:
    ds = CellDataset(df, cfg["paths"]["data_root"], build_transforms(cfg, train, augment))
    bs = batch_size or cfg["train"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    sampler = None
    if train and cfg["train"]["imbalance"] == "weighted_sampler":
        counts = np.bincount(ds.labels, minlength=2)
        w = (1.0 / counts)[ds.labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
    return DataLoader(ds, batch_size=bs, shuffle=train and sampler is None, sampler=sampler,
                      num_workers=nw, persistent_workers=nw > 0 and cfg["data"]["persistent_workers"],
                      pin_memory=torch.cuda.is_available(), drop_last=train)
