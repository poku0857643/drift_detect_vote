"""Tests for the numpy-based GNNDriftDetector (tabular graph detector)."""
from __future__ import annotations

import numpy as np

from app.detectors.gnn import GNNDriftDetector


class TestGNNDetector:
    def test_no_drift_same_distribution(self, reference, no_drift, feature_names):
        d = GNNDriftDetector(threshold=0.05)
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert not result.drift_detected

    def test_drift_detected_on_shifted_data(self, reference, drifted, feature_names):
        d = GNNDriftDetector(threshold=0.05)
        d.fit(reference, feature_names)
        result = d.detect(drifted)
        assert result.drift_detected

    def test_score_bounded(self, reference, drifted, feature_names):
        d = GNNDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(drifted)
        assert 0.0 <= result.score <= 1.0

    def test_feature_scores_present(self, reference, no_drift, feature_names):
        d = GNNDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert set(result.feature_scores.keys()) == set(feature_names)

    def test_detector_name(self, reference, no_drift, feature_names):
        d = GNNDriftDetector()
        d.fit(reference, feature_names)
        result = d.detect(no_drift)
        assert "gnn" in result.detector_name.lower()