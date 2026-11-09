import numpy as np
import pytest

from src.evaluate import binary_metrics, brier, cluster_bootstrap, ece, nll, risk_coverage


def test_ece_hand_computed():
    # 15 bins. Bin of 0.9: 2 samples, 1 positive -> |0.9-0.5| * 2/4 = 0.2
    # Bin of 0.1: 2 samples, 0 positive -> |0.1-0| * 2/4 = 0.05 ; total 0.25
    p = np.array([0.9, 0.9, 0.1, 0.1])
    y = np.array([1, 0, 0, 0])
    assert ece(p, y, 15) == pytest.approx(0.25)


def test_ece_perfect_calibration_is_zero():
    assert ece(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)


def test_nll_and_brier():
    p, y = np.array([0.8, 0.3]), np.array([1, 0])
    assert nll(p, y) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    assert brier(p, y) == pytest.approx((0.04 + 0.09) / 2)


def test_binary_metrics_confusion():
    m = binary_metrics([0.9, 0.6, 0.4, 0.2], [1, 0, 1, 0])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["recall"] == 0.5 and m["specificity"] == 0.5


def test_cluster_bootstrap_resamples_patients():
    rng = np.random.default_rng(0)
    patients = np.repeat(np.arange(30), 10)
    labels = (patients % 2).astype(int)
    probs = np.clip(labels * 0.7 + rng.normal(0.15, 0.1, len(labels)), 0, 1)
    acc = lambda p, y: float(((p >= 0.5) == y).mean())
    point, lo, hi = cluster_bootstrap(acc, probs, labels, patients, iters=200)
    assert lo <= point <= hi


def test_risk_coverage_monotone_ends():
    cov, risk, aurc = risk_coverage(np.array([0.99, 0.01, 0.6, 0.45]), np.array([1, 0, 0, 1]))
    assert cov[-1] == 1.0 and risk[0] == 0.0 and risk[-1] == 0.5 and 0 <= aurc <= 1
