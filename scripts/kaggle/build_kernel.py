"""Build the private Kaggle job that retrains the 5-seed "robust" ensemble on a Kaggle T4.

Usage
  python scripts/kaggle/build_kernel.py --owner haiclop        # writes dist/kaggle_train/
  kaggle kernels push -p dist/kaggle_train                      # (token in KAGGLE_API_TOKEN)

The job carries its own copy of the code, the audited split (data/splits.csv) and a pixel fingerprint of every
training and validation image, so it does not depend on unpushed GitHub state, and it refuses to train if the
images it downloads from Hugging Face differ from the ones used on the laptop. Outputs in /kaggle/working:
checkpoints/effb0_robust_s*.pt, reports/{runs,preds,logs}/..., kaggle_run.json.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ["src", "configs", "data/splits.csv", "data/pixel_hashes_trainval.csv"]

RUNNER = r'''
# Cell Triage: retrain the 5-seed "robust" ensemble on a Kaggle GPU. Research demo, not for clinical use.
# Built by scripts/kaggle/build_kernel.py; the project code travels inside BUNDLE below.
import base64, hashlib, io, json, os, shutil, subprocess, sys, time, zipfile

SEEDS = __SEEDS__
W, CODE, DATA = "/kaggle/working", "/tmp/code", "/tmp/cnmc"
os.environ.update(HF_HOME="/tmp/hf", TORCH_HOME="/tmp/torch", PIP_CACHE_DIR="/tmp/pip",
                  CNMC_UINT8_CACHE="/tmp/cnmc_train_224_uint8.npz", PYTHONUNBUFFERED="1")
status = {"started": time.time(), "seeds_done": [], "stage": "setup"}


def save_status(**kw):
    status.update(kw, updated=time.time())
    json.dump(status, open(f"{W}/kaggle_run.json", "w"), indent=2)


def run(cmd, log=None):
    print("$", " ".join(cmd), flush=True)
    out = open(log, "w") if log else None
    r = subprocess.run(cmd, cwd=CODE, stdout=out or None, stderr=subprocess.STDOUT if out else None)
    if out:
        out.close()
        print(open(log).read()[-3000:], flush=True)
    if r.returncode:
        save_status(stage=f"failed: {' '.join(cmd[:4])}")
        raise SystemExit(f"command failed ({r.returncode})")


BUNDLE = "__BUNDLE__"
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(CODE)
save_status(stage="checking packages and internet")
import importlib.util, socket  # noqa: E402
missing = [m for m in ("timm", "datasets", "huggingface_hub", "yaml") if importlib.util.find_spec(m) is None]
try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
    online = True
except OSError:
    online = False
save_status(online=online, missing_packages=missing)
if not online:
    raise SystemExit("No internet in this Kaggle session. Verify a phone number on the Kaggle account "
                     "(Settings > Phone verification), then run the job again.")
if missing:
    run([sys.executable, "-m", "pip", "install", "-q", *[{"yaml": "pyyaml"}.get(m, m) for m in missing]])

import torch, yaml  # noqa: E402
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print("GPU:", gpu, flush=True)
assert torch.cuda.is_available(), "no GPU in this session"

cfg_path = f"{CODE}/configs/config.yaml"
cfg = yaml.safe_load(open(cfg_path))
cfg["paths"].update(data_root=DATA, checkpoints=f"{W}/checkpoints")
cfg["train"]["amp"] = True          # T4/P100 handle fp16 (the laptop's GTX 1650 did not)
cfg["data"]["num_workers"] = 4
yaml.safe_dump(cfg, open(cfg_path, "w"))

save_status(stage="downloading C-NMC from Hugging Face", gpu=gpu)
run([sys.executable, "-m", "src.download"], log=f"{W}/download.log")

# the images must be pixel-identical to the laptop's, or the comparison with the deployed model is void
import numpy as np, pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
fp = pd.read_csv(f"{CODE}/data/pixel_hashes_trainval.csv")
bad = [p for p, h in zip(fp.image_path, fp.pixel_sha256_20)
       if not os.path.exists(f"{DATA}/{p}")
       or hashlib.sha256(np.asarray(Image.open(f"{DATA}/{p}").convert("RGB")).tobytes()).hexdigest()[:20] != h]
save_status(fingerprint_mismatches=len(bad))
assert not bad, f"{len(bad)} images differ from the laptop copy, e.g. {bad[:3]}"
print("all", len(fp), "train/val images match the laptop copy", flush=True)

os.makedirs(f"{W}/reports/logs", exist_ok=True)
for seed in SEEDS:
    name = f"effb0_robust_s{seed}"
    save_status(stage=f"training {name}")
    t0 = time.time()
    run([sys.executable, "-m", "src.train", "--name", name, "--set", f"seed={seed}", "train.augment=robust"],
        log=f"{W}/reports/logs/{name}.log")
    for sub in ("runs", "preds"):
        os.makedirs(f"{W}/reports/{sub}", exist_ok=True)
        for f in os.listdir(f"{CODE}/reports/{sub}"):
            if f.startswith(name):
                shutil.copy(f"{CODE}/reports/{sub}/{f}", f"{W}/reports/{sub}/{f}")
    status["seeds_done"].append({"seed": seed, "minutes": round((time.time() - t0) / 60, 1)})
    save_status()
save_status(stage="finished")
print(json.dumps(status, indent=2))
'''


def bundle() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
            for f in files:
                z.write(f, f.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", default="cell-triage-robust-training")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    out = ROOT / "dist" / "kaggle_train"
    out.mkdir(parents=True, exist_ok=True)
    code = RUNNER.replace("__SEEDS__", json.dumps(args.seeds)).replace("__BUNDLE__", bundle())
    (out / "train_robust_kaggle.py").write_text(code, encoding="utf-8")
    meta = {"id": f"{args.owner}/{args.slug}", "title": args.slug, "code_file": "train_robust_kaggle.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({len(code) / 1e6:.2f} MB script)")


if __name__ == "__main__":
    main()
