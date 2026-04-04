"""
Statistical Drift Detector
===========================
Kolmogorov–Smirnov two-sample test per feature with Bonferroni correction
for multiple comparisons.  Combined significance via Fisher's method.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from .base import DetectorResult, DriftDetector


class StatisticalDriftDetector(DriftDetector):
    """KS-test drift detector with Bonferroni-corrected per-feature testing."""

    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__(threshold)
        self._reference: np.ndarray | None = None
        self._feature_names: list[str] = []

    def fit(self, reference: np.ndarray, feature_names: list[str]) -> None:
        self._reference = reference.copy()
        self._feature_names = feature_names

    def detect(self, current: np.ndarray) -> DetectorResult:
        n_features = self._reference.shape[1]
        corrected_alpha = self.threshold / max(n_features, 1)

        ks_stats: list[float] = []
        p_values: list[float] = []

        for i in range(n_features):
            stat, pval = scipy_stats.ks_2samp(
                self._reference[:, i], current[:, i]
            )
            ks_stats.append(float(stat))
            p_values.append(float(pval))

        n_drifted = sum(p < corrected_alpha for p in p_values)

        # Fisher's combined p-value across all features
        safe_p = [max(p, 1e-300) for p in p_values]
        chi2 = -2.0 * sum(np.log(p) for p in safe_p)
        combined_p = float(scipy_stats.chi2.sf(chi2, df=2 * n_features))

        drift_detected = n_drifted > 0
        score = float(n_drifted / max(n_features, 1))

        feature_scores = {
            self._feature_names[i]: ks_stats[i] for i in range(n_features)
        }
        per_feature_p = {
            self._feature_names[i]: p_values[i] for i in range(n_features)
        }

        return DetectorResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            score=score,
            threshold=self.threshold,
            p_value=combined_p,
            feature_scores=feature_scores,
            meta={
                "bonferroni_alpha": corrected_alpha,
                "n_features_drifted": n_drifted,
                "fisher_combined_p": combined_p,
                "per_feature_p": per_feature_p,
            },
        )