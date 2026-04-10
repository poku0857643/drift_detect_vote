"""Tests for StatisticalDriftDetector (KS + PSI + Wasserstein)."""
from __future__ import annotations

import numpy as np
import pytest

from app.detectors.statistical import StatisticalDriftDetector, _psi, _wasserstein


# ─────────────────────────────────────────────────────────────────── #
# Unit helpers
# ─────────────────────────────────────────────────────────────────── #

def test_psi_identical_distributions():
    rng = np.random.default_rng(0)
    col = rng.standard_normal(500).astype(np.float32)
    assert _psi(col, col) == pytest.approx(0.0, abs=1e-6)


def test_psi_large_shift():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(500).astype(np.float32)
    cur = (rng.standard_normal(500) + 10.0).astype(np.float32)
    assert _psi(ref, cur) > 0.20


def test_wasserstein_identical():
    col = np.linspace(0, 1, 200).astype(np.float32)
    assert _wasserstein(col, col) == pytest.approx(0.0, abs=1e-6)


def test_wasserstein_large_shift():
    ref = np.zeros(200, dtype=np.float32)
    cur = np.ones(200, dtype=np.float32) * 5
    # raw Wasserstein = 5, std of ref = 0 → clipped to 1; result = 5 / 1 = 5
    assert _wasserstein(ref, cur) > 1.0


# ─────────────────────────────────────────────────────────────────── #
# Detector behaviour
# ─────────────────────────────────────────────────────────────────── #

class TestStatisticalDetector:
    def test_no_drift_same_distribution(self, reference, no_drift, feature_names):
        d = StatisticalDriftDetector(threshold=0.05)
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert not result.drift_detected, "same-distribution data should not trigger drift"

    def test_drift_detected_on_shifted_data(self, reference, drifted, feature_names):
        d = StatisticalDriftDetector(threshold=0.05)
        d.fit(reference, feature_names)
        result = d.detect(drifted)
        assert result.drift_detected, "+5 shift must trigger drift"

    def test_score_bounded(self, reference, drifted, feature_names):
        d = StatisticalDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(drifted)
        assert 0.0 <= result.score <= 1.0

    def test_feature_scores_keys_match_names(self, reference, no_drift, feature_names):
        d = StatisticalDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert set(result.feature_scores.keys()) == set(feature_names)

    def test_meta_contains_per_feature(self, reference, no_drift, feature_names):
        d = StatisticalDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert "per_feature" in result.meta
        assert set(result.meta["per_feature"].keys()) == set(feature_names)

    def test_psi_threshold_respected(self, reference, feature_names):
        rng = np.random.default_rng(1)
        # Very mild shift — high PSI threshold should not flag it
        mild = (rng.standard_normal((100, 4)) + 0.1).astype(np.float32)
        d = StatisticalDriftDetector(psi_threshold=10.0, wasserstein_threshold=10.0, threshold=1e-10)
        d.fit(reference, feature_names)
        result = d.detect(mild)
        assert not result.drift_detected