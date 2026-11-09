"""Shared helpers: config loading, seeding, run provenance."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Keep every cache off the C drive even if the shell did not export them.
for _k, _v in {
    "HF_HOME": "E:/cache/hf",
    "TORCH_HOME": "E:/cache/torch",
    "PIP_CACHE_DIR": "E:/cache/pip",
}.items():
    os.environ.setdefault(_k, _v)


def load_config(path: str | Path = ROOT / "configs" / "config.yaml", overrides: dict | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
