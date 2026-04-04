"""
Isolation Forest Drift Detector
=================================
Fits an Isolation Forest on reference data to model the "normal" distribution.
Drift is detected when the anomaly score distribution of the current data
differs significantly from the reference anomaly score distribution
(Kolmogorov–Smirnov test on IF scores).

Per-feature importance is estimated via single-pass score perturbation:
each feature is zeroed out and the resulting score shift is measured.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .base import DetectorResult, DriftDetector


class IsolationForestDriftDetector(DriftDetector):
    """Isolation Forest anomaly-score drift detector."""

    def __init__(self, threshold: float = 0.05, n_estimators: int = 100) -> None:
        super().__init__(threshold)
        self.n_estimators = n_estimators
        self._model: IsolationForest | None = None
        self._scaler: StandardScaler | None = None
        self._ref_scores: np.ndarray | None = None
        self._ref_anomaly_rate: float = 0.0
        self._feature_names: list[str] = []

    def fit(self, reference: np.ndarray, feature_names: list[str]) -> None:
        self._feature_names = feature_names
        self._scaler = StandardScaler()
        ref_scaled = self._scaler.fit_transform(reference)

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=42,
        )
        self._model.fit(ref_scaled)
        self._ref_scores = self._model.score_samples(ref_scaled)
        self._ref_anomaly_rate = float(
            (self._model.predict(ref_scaled) == -1).mean()
        )

    def detect(self, current: np.ndarray) -> DetectorResult:
        cur_scaled = self._scaler.transform(current)
        cur_scores = self._model.score_samples(cur_scaled)

        ks_stat, p_value = scipy_stats.ks_2samp(self._ref_scores, cur_scores)
        drift_detected = p_value < self.threshold
        cur_anomaly_rate = float((self._model.predict(cur_scaled) == -1).mean())

        # Per-feature importance via score perturbation (single pass)
        baseline_mean = float(cur_scores.mean())
        feature_scores: dict[str, float] = {}
        for i, fname in enumerate(self._feature_names):
            perturbed = cur_scaled.copy()
            perturbed[:, i] = 0.0
            delta = abs(float(self._model.score_samples(perturbed).mean()) - baseline_mean)
            feature_scores[fname] = delta

        total = sum(feature_scores.values()) or 1.0
        feature_scores = {k: v / total for k, v in feature_scores.items()}

        return DetectorResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            score=float(ks_stat),
            threshold=self.threshold,
            p_value=float(p_value),
            feature_scores=feature_scores,
            meta={
                "ks_statistic": float(ks_stat),
                "reference_anomaly_rate": self._ref_anomaly_rate,
                "current_anomaly_rate": cur_anomaly_rate,
                "mean_ref_if_score": float(self._ref_scores.mean()),
                "mean_cur_if_score": float(cur_scores.mean()),
                "score_shift": float(cur_scores.mean() - self._ref_scores.mean()),
            },
        )