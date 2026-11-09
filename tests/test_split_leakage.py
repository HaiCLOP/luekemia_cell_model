from pathlib import Path

import pandas as pd
import pytest

from src.split import assert_no_leakage, make_splits
from src.utils import ROOT, load_config

SPLITS = ROOT / "data" / "splits.csv"


def _toy_manifest(n_patients=40, imgs=20):
    rows = []
    for p in range(n_patients):
        label = int(p % 3 != 0)  # about 2:1 positive, like C-NMC
        for i in range(imgs + p % 5):
            rows.append({"image_path": f"x/{p}_{i}.png", "patient_id": f"P{p}", "label": label})
    return pd.DataFrame(rows)


def test_make_splits_has_no_shared_patients():
    out = make_splits(_toy_manifest(), load_config())
    assert out.groupby("patient_id")["split"].nunique().max() == 1
    assert set(out["split"]) == {"train", "val", "test"}
    assert (out.loc[out.split == "train", "fold"] >= 0).all()


def test_leakage_detector_fires():
    df = pd.DataFrame({"image_path": ["a", "b"], "patient_id": ["P1", "P1"], "label": [1, 1], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        assert_no_leakage(df)


@pytest.mark.skipif(not SPLITS.exists(), reason="data/splits.csv not generated yet")
def test_real_splits_no_leakage_and_balanced():
    df = pd.read_csv(SPLITS)
    assert df.groupby("patient_id")["split"].nunique().max() == 1, "patient appears in more than one split"
    overall = df["label"].mean()
    for split, grp in df.groupby("split"):
        assert abs(grp["label"].mean() - overall) < 0.15, f"{split} label ratio drifts from overall"
    assert df["image_path"].is_unique
    if "group_id" in df.columns:
        assert df.groupby("group_id")["split"].nunique().max() == 1, "aliased patient group straddles splits"


MANIFEST = Path("E:/datasets/cnmc/manifest.csv")


@pytest.mark.skipif(not (SPLITS.exists() and MANIFEST.exists()), reason="data not exported yet")
def test_no_identical_image_in_two_splits():
    df = pd.read_csv(SPLITS).merge(pd.read_csv(MANIFEST)[["image_path", "sha256"]], on="image_path", how="left")
    assert df["sha256"].notna().all()
    assert df.groupby("sha256")["split"].nunique().max() == 1
