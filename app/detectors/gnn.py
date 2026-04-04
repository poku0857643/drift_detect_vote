"""
GNN Drift Detector
==================
Treats dataset *features* as graph nodes connected by Pearson-correlation
edges.  Statistical node features (mean, std, quantiles, skew, kurtosis)
are propagated through the reference correlation graph via multi-layer GCN
message-passing (no learned weights required).

Drift is detected when the mean per-node L2 distance between the reference
embeddings and the current embeddings exceeds a bootstrapped threshold.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from .base import DetectorResult, DriftDetector


class GNNDriftDetector(DriftDetector):
    """Graph Neural Network drift detector (numpy-only, no PyTorch)."""

    def __init__(
        self,
        threshold: float = 0.05,
        n_layers: int = 2,
        corr_edge_threshold: float = 0.3,
    ) -> None:
        super().__init__(threshold)
        self.n_layers = n_layers
        self.corr_edge_threshold = corr_edge_threshold

        self._P: np.ndarray | None = None          # reference propagation matrix
        self._X_scale: np.ndarray | None = None    # node-feature normalisation
        self._ref_emb: np.ndarray | None = None    # (n_features, n_stats) embeddings
        self._dist_threshold: float = 0.1
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    # Graph helpers
    # ------------------------------------------------------------------ #

    def _build_prop_matrix(self, data: np.ndarray) -> np.ndarray:
        """D^{-1/2}(A + I)D^{-1/2} from feature-correlation graph."""
        n = data.shape[1]
        if n == 1:
            return np.array([[1.0]])
        corr = np.abs(np.corrcoef(data.T))
        A = (corr > self.corr_edge_threshold).astype(float)
        np.fill_diagonal(A, 0.0)
        A_hat = A + np.eye(n)
        deg = A_hat.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        return d_inv_sqrt @ A_hat @ d_inv_sqrt

    # ------------------------------------------------------------------ #
    # Node feature computation  (n_features × 8 matrix)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_features(data: np.ndarray) -> np.ndarray:
        skew = np.nan_to_num(scipy_stats.skew(data, axis=0), nan=0.0)
        kurt = np.nan_to_num(scipy_stats.kurtosis(data, axis=0), nan=0.0)
        return np.stack(
            [
                data.mean(axis=0),
                data.std(axis=0) + 1e-9,
                np.percentile(data, 25, axis=0),
                np.percentile(data, 75, axis=0),
                np.percentile(data, 5, axis=0),
                np.percentile(data, 95, axis=0),
                np.clip(skew, -10.0, 10.0),
                np.clip(kurt, -10.0, 10.0),
            ],
            axis=1,
        )

    # ------------------------------------------------------------------ #
    # GCN propagation
    # ------------------------------------------------------------------ #

    def _propagate(self, X: np.ndarray) -> np.ndarray:
        H = X
        for _ in range(self.n_layers):
            H = self._P @ H
        return H

    # ------------------------------------------------------------------ #
    # fit / detect
    # ------------------------------------------------------------------ #

    def fit(self, reference: np.ndarray, feature_names: list[str]) -> None:
        self._feature_names = feature_names
        self._P = self._build_prop_matrix(reference)

        X_ref = self._node_features(reference)
        # scale by per-stat std so no stat dominates
        self._X_scale = np.maximum(X_ref.std(axis=0), 1e-9)
        self._ref_emb = self._propagate(X_ref / self._X_scale)

        # Bootstrap threshold: 3× the intra-distribution embedding variance
        half = max(len(reference) // 2, 5)
        emb1 = self._propagate(self._node_features(reference[:half]) / self._X_scale)
        emb2 = self._propagate(self._node_features(reference[half:]) / self._X_scale)
        baseline = float(np.linalg.norm(emb1 - emb2, axis=1).mean())
        self._dist_threshold = max(baseline * 3.0, 0.05)

    def detect(self, current: np.ndarray) -> DetectorResult:
        X_cur = self._node_features(current) / self._X_scale
        cur_emb = self._propagate(X_cur)

        per_node = np.linalg.norm(self._ref_emb - cur_emb, axis=1)  # (n_features,)
        mean_dist = float(per_node.mean())
        drift_detected = mean_dist > self._dist_threshold

        feature_scores = {
            self._feature_names[i]: float(per_node[i])
            for i in range(len(self._feature_names))
        }

        n_edges = int((self._P > 1e-9).sum()) - current.shape[1]

        return DetectorResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            score=float(min(mean_dist / max(self._dist_threshold, 1e-9), 1.0)),
            threshold=self._dist_threshold,
            feature_scores=feature_scores,
            meta={
                "mean_embedding_distance": mean_dist,
                "distance_threshold": self._dist_threshold,
                "n_graph_edges": max(n_edges, 0),
                "n_layers": self.n_layers,
                "corr_edge_threshold": self.corr_edge_threshold,
            },
        )
