"""Tests for IsolationForestDriftDetector (IF + LOF + marginal SHAP)."""
from __future__ import annotations

import numpy as np
import pytest

from app.detectors.isolation import IsolationForestDriftDetector


class TestIsolationForestDetector:
    def test_no_drift_same_distribution(self, reference, no_drift, feature_names):
        d = IsolationForestDriftDetector(threshold=0.05)
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert not result.drift_detected

    def test_drift_detected_on_shifted_data(self, reference, drifted, feature_names):
        d = IsolationForestDriftDetector(threshold=0.05)
        d.fit(reference, feature_names)
        result = d.detect(drifted)
        assert result.drift_detected

    def test_score_bounded(self, reference, drifted, feature_names):
        d = IsolationForestDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(drifted)
        assert 0.0 <= result.score <= 1.0

    def test_feature_scores_sum_to_one(self, reference, drifted, feature_names):
        d = IsolationForestDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(drifted)
        total = sum(result.feature_scores.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_meta_if_keys_present(self, reference, no_drift, feature_names):
        d = IsolationForestDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        for key in ("if_ks_statistic", "if_p_value", "reference_anomaly_rate"):
            assert key in result.meta

    def test_lof_disabled(self, reference, no_drift, feature_names):
        d = IsolationForestDriftDetector(use_lof=False)
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert result.meta["lof_enabled"] is False
        assert result.meta["lof_ks_statistic"] == 0.0

    def test_detector_name(self, reference, no_drift, feature_names):
        d = IsolationForestDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert "isolation" in result.detector_name.lower()