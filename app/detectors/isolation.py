"""
Isolation Forest + LOF Drift Detector
=======================================
Two complementary ML anomaly detectors run in parallel and their verdicts
are combined.

Isolation Forest (IF)
---------------------
Fits on reference data to define "normal".  Drift detected when the anomaly
score distribution of current data differs significantly from reference
(Kolmogorov–Smirnov test on IF scores).

Local Outlier Factor (LOF)
--------------------------
Measures local density deviation of each sample compared to its neighbours.
Fitted on reference data; LOF scores on current data are compared to
reference LOF scores via KS test.  LOF is more sensitive to local cluster
shifts that IF can miss.

Feature Attribution — Marginal SHAP approximation
--------------------------------------------------
Each feature's contribution to the drift score is estimated by measuring
how much the mean anomaly score changes when that feature is replaced with
its reference mean (marginal substitution).  This is a single-pass
approximation of the SHAP marginal contribution.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .base import DetectorResult, DriftDetector


class IsolationForestDriftDetector(DriftDetector):
    """
    IF + LOF dual anomaly detector with marginal SHAP attribution.

    Parameters
    ----------
    threshold : float
        KS test significance level for both IF and LOF score comparisons.
    n_estimators : int
        Number of trees in the IsolationForest (default 100).
    lof_neighbors : int
        Number of neighbours for LocalOutlierFactor (default 20).
    use_lof : bool
        Whether to also run LOF alongside IF (default True).
    """

    def __init__(
        self,
        threshold: float = 0.05,
        n_estimators: int = 100,
        lof_neighbors: int = 20,
        use_lof: bool = True,
    ) -> None:
        super().__init__(threshold)
        self.n_estimators  = n_estimators
        self.lof_neighbors = lof_neighbors
        self.use_lof       = use_lof

        self._scaler: StandardScaler | None = None
        self._if_model: IsolationForest | None = None
        self._lof_model: LocalOutlierFactor | None = None

        self._ref_if_scores:   np.ndarray | None = None
        self._ref_lof_scores:  np.ndarray | None = None
        self._ref_means:       np.ndarray | None = None   # for SHAP substitution
        self._ref_anomaly_rate: float = 0.0
        self._feature_names:   list[str] = []

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, reference: np.ndarray, feature_names: list[str]) -> None:
        self._feature_names = feature_names
        self._scaler = StandardScaler()
        ref_scaled = self._scaler.fit_transform(reference)

        # Store reference column means for SHAP substitution
        self._ref_means = ref_scaled.mean(axis=0)

        # Isolation Forest
        self._if_model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=42,
        )
        self._if_model.fit(ref_scaled)
        self._ref_if_scores  = self._if_model.score_samples(ref_scaled)
        self._ref_anomaly_rate = float(
            (self._if_model.predict(ref_scaled) == -1).mean()
        )

        # Local Outlier Factor (novelty=True so we can score new data)
        if self.use_lof:
            n_neighbors = min(self.lof_neighbors, len(reference) - 1)
            self._lof_model = LocalOutlierFactor(
                n_neighbors=n_neighbors, novelty=True
            )
            self._lof_model.fit(ref_scaled)
            self._ref_lof_scores = self._lof_model.score_samples(ref_scaled)

    # ------------------------------------------------------------------ #
    # Feature attribution — marginal SHAP approximation
    # ------------------------------------------------------------------ #

    def _marginal_shap(self, cur_scaled: np.ndarray) -> dict[str, float]:
        """
        For each feature i, replace column i with the reference mean and
        measure the change in mean IF anomaly score.  Features that, when
        removed, cause the score to shift most are the primary drift drivers.
        """
        baseline = float(self._if_model.score_samples(cur_scaled).mean())
        scores: dict[str, float] = {}
        for i, fname in enumerate(self._feature_names):
            perturbed = cur_scaled.copy()
            perturbed[:, i] = self._ref_means[i]   # substitute with ref mean
            delta = abs(
                float(self._if_model.score_samples(perturbed).mean()) - baseline
            )
            scores[fname] = delta

        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    # ------------------------------------------------------------------ #
    # Detect
    # ------------------------------------------------------------------ #

    def detect(self, current: np.ndarray) -> DetectorResult:
        cur_scaled   = self._scaler.transform(current)
        cur_if_scores = self._if_model.score_samples(cur_scaled)

        # IF KS test
        if_ks, if_p = scipy_stats.ks_2samp(self._ref_if_scores, cur_if_scores)
        if_drift     = if_p < self.threshold

        # LOF KS test
        lof_drift = False
        lof_ks, lof_p = 0.0, 1.0
        if self.use_lof and self._lof_model is not None:
            cur_lof_scores = self._lof_model.score_samples(cur_scaled)
            lof_ks, lof_p  = scipy_stats.ks_2samp(self._ref_lof_scores, cur_lof_scores)
            lof_drift       = lof_p < self.threshold

        # Combined verdict: drift if either detector flags it
        drift_detected = if_drift or lof_drift

        # Composite score: max of both KS statistics (worst case)
        score = float(max(if_ks, lof_ks))

        # Combined p (Fisher's method on the two tests)
        p_vals = [max(if_p, 1e-300)]
        if self.use_lof:
            p_vals.append(max(lof_p, 1e-300))
        chi2 = -2.0 * sum(np.log(p) for p in p_vals)
        combined_p = float(scipy_stats.chi2.sf(chi2, df=2 * len(p_vals)))

        cur_anomaly_rate = float((self._if_model.predict(cur_scaled) == -1).mean())
        feature_scores   = self._marginal_shap(cur_scaled)

        return DetectorResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            score=score,
            threshold=self.threshold,
            p_value=combined_p,
            feature_scores=feature_scores,
            meta={
                # IF metrics
                "if_ks_statistic":        float(if_ks),
                "if_p_value":             float(if_p),
                "if_drift_detected":      if_drift,
                "reference_anomaly_rate": self._ref_anomaly_rate,
                "current_anomaly_rate":   cur_anomaly_rate,
                "mean_ref_if_score":      float(self._ref_if_scores.mean()),
                "mean_cur_if_score":      float(cur_if_scores.mean()),
                "if_score_shift":         float(cur_if_scores.mean() - self._ref_if_scores.mean()),
                # LOF metrics
                "lof_enabled":            self.use_lof,
                "lof_ks_statistic":       float(lof_ks),
                "lof_p_value":            float(lof_p),
                "lof_drift_detected":     lof_drift,
            },
        )