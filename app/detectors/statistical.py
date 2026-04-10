"""
Statistical Drift Detector
===========================
Multi-metric statistical drift detection per feature.

Tests run per feature
---------------------
1. Kolmogorov–Smirnov two-sample test — sensitive to any distributional change
2. Population Stability Index (PSI) — industry-standard credit/risk metric;
   measures how much a distribution has shifted using binned probabilities
3. Wasserstein distance (Earth Mover's Distance) — measures the minimum
   cost to transform one distribution into the other; scale-aware

Final verdict
-------------
A feature is considered drifted if **any** of the three tests flags it.
Bonferroni correction applied to KS p-values.  Fisher's method gives a
combined p-value across all features.

Score = weighted combination:
    0.4 × (fraction KS-drifted) +
    0.3 × (mean normalised PSI)  +
    0.3 × (mean normalised Wasserstein)
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from .base import DetectorResult, DriftDetector

# PSI thresholds (industry standard)
_PSI_STABLE   = 0.10   # negligible shift
_PSI_WARNING  = 0.20   # moderate shift — worth monitoring
# > 0.20 → significant shift


def _psi(reference_col: np.ndarray, current_col: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index for a single feature.

    PSI = Σ (P_ref - P_cur) * ln(P_ref / P_cur)

    Bins are defined on the reference distribution so PSI is always
    computed against the same reference bucket boundaries.
    """
    # Use reference percentiles to define bins
    bin_edges = np.percentile(reference_col, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)          # collapse duplicate edges
    if len(bin_edges) < 2:
        return 0.0

    # Extend with -inf / +inf so that values outside the reference range are
    # captured rather than silently dropped by np.histogram.
    bins_ext = np.concatenate([[-np.inf], bin_edges[1:-1], [np.inf]])
    ref_counts, _ = np.histogram(reference_col, bins=bins_ext)
    cur_counts, _ = np.histogram(current_col,   bins=bins_ext)

    # Smooth to avoid log(0)
    ref_pct = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_pct = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))

    psi = float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))
    return max(psi, 0.0)


def _wasserstein(reference_col: np.ndarray, current_col: np.ndarray) -> float:
    """Normalised Wasserstein-1 distance (normalised by reference std)."""
    raw = float(scipy_stats.wasserstein_distance(reference_col, current_col))
    ref_std = float(reference_col.std()) or 1.0
    return raw / ref_std


class StatisticalDriftDetector(DriftDetector):
    """
    KS test + PSI + Wasserstein drift detector.

    Parameters
    ----------
    threshold : float
        KS test significance level (Bonferroni-corrected per feature).
    psi_threshold : float
        PSI value above which a feature is considered drifted (default 0.20).
    wasserstein_threshold : float
        Normalised Wasserstein distance above which a feature is flagged
        (default 0.20 — roughly 20 % of the reference std).
    n_bins : int
        Number of equal-probability bins for PSI calculation (default 10).
    """

    def __init__(
        self,
        threshold: float = 0.05,
        psi_threshold: float = 0.20,
        wasserstein_threshold: float = 0.20,
        n_bins: int = 10,
    ) -> None:
        super().__init__(threshold)
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.n_bins = n_bins
        self._reference: np.ndarray | None = None
        self._feature_names: list[str] = []

    def fit(self, reference: np.ndarray, feature_names: list[str]) -> None:
        self._reference = reference.copy()
        self._feature_names = feature_names

    def detect(self, current: np.ndarray) -> DetectorResult:
        n = self._reference.shape[1]
        corrected_alpha = self.threshold / max(n, 1)

        ks_stats:    list[float] = []
        p_values:    list[float] = []
        psi_values:  list[float] = []
        wass_values: list[float] = []

        for i in range(n):
            ref_col = self._reference[:, i]
            cur_col = current[:, i]

            ks_stat, pval = scipy_stats.ks_2samp(ref_col, cur_col)
            ks_stats.append(float(ks_stat))
            p_values.append(float(pval))
            psi_values.append(_psi(ref_col, cur_col, self.n_bins))
            wass_values.append(_wasserstein(ref_col, cur_col))

        # Per-feature drift flags
        ks_drifted   = [p < corrected_alpha         for p in p_values]
        psi_drifted  = [v > self.psi_threshold       for v in psi_values]
        wass_drifted = [v > self.wasserstein_threshold for v in wass_values]
        any_drifted  = [a or b or c for a, b, c in zip(ks_drifted, psi_drifted, wass_drifted)]

        n_drifted = sum(any_drifted)
        drift_detected = n_drifted > 0

        # Fisher combined p-value (KS only — PSI/Wass have no p-value)
        safe_p   = [max(p, 1e-300) for p in p_values]
        chi2     = -2.0 * sum(np.log(p) for p in safe_p)
        combined_p = float(scipy_stats.chi2.sf(chi2, df=2 * n))

        # Composite score  (0–1)
        mean_psi_norm  = float(np.mean([min(v / max(self.psi_threshold, 1e-9), 1.0) for v in psi_values]))
        mean_wass_norm = float(np.mean([min(v / max(self.wasserstein_threshold, 1e-9), 1.0) for v in wass_values]))
        ks_frac        = float(sum(ks_drifted) / max(n, 1))
        score = float(0.4 * ks_frac + 0.3 * mean_psi_norm + 0.3 * mean_wass_norm)

        # feature_scores: primary ordering by KS statistic
        feature_scores = {self._feature_names[i]: ks_stats[i] for i in range(n)}

        per_feature_detail = {
            self._feature_names[i]: {
                "ks_stat":          ks_stats[i],
                "p_value":          p_values[i],
                "psi":              psi_values[i],
                "wasserstein":      wass_values[i],
                "ks_drifted":       ks_drifted[i],
                "psi_drifted":      psi_drifted[i],
                "wass_drifted":     wass_drifted[i],
            }
            for i in range(n)
        }

        return DetectorResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            score=min(score, 1.0),
            threshold=self.threshold,
            p_value=combined_p,
            feature_scores=feature_scores,
            meta={
                "bonferroni_alpha":      corrected_alpha,
                "n_features_drifted":    n_drifted,
                "fisher_combined_p":     combined_p,
                "mean_psi":              float(np.mean(psi_values)),
                "mean_wasserstein":      float(np.mean(wass_values)),
                "psi_threshold":         self.psi_threshold,
                "wasserstein_threshold": self.wasserstein_threshold,
                "per_feature":           per_feature_detail,
            },
        )