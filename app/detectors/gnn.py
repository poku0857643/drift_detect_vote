"""
GNN Drift Detector
==================
Treats dataset *features* as graph nodes connected by Pearson-correlation
edges.  Statistical node features (mean, std, quantiles, skew, kurtosis)
are propagated through the reference correlation graph via multi-layer GCN
message-passing (no learned weights required).

Enhancements over baseline
--------------------------
- ``aggregation`` — choose 'mean' (GCN), 'sum', or 'max' pooling per layer
- ``use_edge_weights`` — use raw correlation strength as edge weight instead
  of a binary threshold, giving a richer graph signal
- ``n_node_features`` — extended node feature set (adds variance, range, CV)
- Multi-scale embedding: concatenates outputs from all layers so shallow and
  deep structural patterns both contribute to the drift score
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from .base import DetectorResult, DriftDetector


class GNNDriftDetector(DriftDetector):
    """
    Graph Neural Network drift detector (numpy-only, no PyTorch).

    Parameters
    ----------
    threshold : float
        Significance threshold (used to set bootstrap multiplier).
    n_layers : int
        Number of GCN message-passing layers (default 2).
    corr_edge_threshold : float
        Minimum |correlation| to include an edge when ``use_edge_weights``
        is False (ignored when True).
    aggregation : str
        Node aggregation rule per layer: ``'mean'`` | ``'sum'`` | ``'max'``.
    use_edge_weights : bool
        If True, edge weights equal |Pearson correlation| rather than 0/1.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        n_layers: int = 2,
        corr_edge_threshold: float = 0.3,
        aggregation: str = "mean",
        use_edge_weights: bool = True,
    ) -> None:
        super().__init__(threshold)
        if aggregation not in ("mean", "sum", "max"):
            raise ValueError("aggregation must be 'mean', 'sum', or 'max'.")
        self.n_layers = n_layers
        self.corr_edge_threshold = corr_edge_threshold
        self.aggregation = aggregation
        self.use_edge_weights = use_edge_weights

        self._P: np.ndarray | None = None
        self._X_scale: np.ndarray | None = None
        self._ref_emb: np.ndarray | None = None   # multi-scale: (n_feat, n_stats * (n_layers+1))
        self._dist_threshold: float = 0.1
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def _build_prop_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Build the normalised propagation matrix.

        When ``use_edge_weights`` is True the raw |correlation| values become
        edge weights, giving higher weight to more strongly correlated
        feature pairs.  Otherwise edges are binary (|corr| > threshold).
        """
        n = data.shape[1]
        if n == 1:
            return np.array([[1.0]])

        corr = np.abs(np.corrcoef(data.T))

        if self.use_edge_weights:
            # Soft adjacency: keep all edges but weight by correlation
            A = corr.copy()
            np.fill_diagonal(A, 0.0)
        else:
            A = (corr > self.corr_edge_threshold).astype(float)
            np.fill_diagonal(A, 0.0)

        A_hat = A + np.eye(n)   # add self-loops
        deg = A_hat.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        return d_inv_sqrt @ A_hat @ d_inv_sqrt

    # ------------------------------------------------------------------ #
    # Node feature computation  (n_features × 11)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_features(data: np.ndarray) -> np.ndarray:
        mean  = data.mean(axis=0)
        std   = data.std(axis=0) + 1e-9
        skew  = np.nan_to_num(scipy_stats.skew(data, axis=0), nan=0.0)
        kurt  = np.nan_to_num(scipy_stats.kurtosis(data, axis=0), nan=0.0)
        cv    = std / (np.abs(mean) + 1e-9)   # coefficient of variation
        rng   = np.percentile(data, 95, axis=0) - np.percentile(data, 5, axis=0)
        return np.stack(
            [
                mean,
                std,
                np.var(data, axis=0),
                np.percentile(data, 25, axis=0),
                np.percentile(data, 75, axis=0),
                np.percentile(data, 5,  axis=0),
                np.percentile(data, 95, axis=0),
                rng,
                np.clip(cv,   -10.0, 10.0),
                np.clip(skew, -10.0, 10.0),
                np.clip(kurt, -10.0, 10.0),
            ],
            axis=1,
        )   # (n_features, 11)

    # ------------------------------------------------------------------ #
    # GCN propagation with configurable aggregation
    # ------------------------------------------------------------------ #

    def _propagate_layer(self, H: np.ndarray) -> np.ndarray:
        """Single GCN layer with the configured aggregation rule."""
        if self.aggregation == "mean":
            return self._P @ H
        if self.aggregation == "sum":
            # un-normalise: multiply by degree to get raw sum
            deg = self._P.sum(axis=1, keepdims=True) + 1e-9
            return (self._P @ H) * deg
        # max aggregation: for each node take element-wise max over neighbours
        n = H.shape[0]
        out = np.zeros_like(H)
        for i in range(n):
            weights = self._P[i]          # (n,)
            neighbours = weights > 1e-9   # mask
            if neighbours.any():
                out[i] = H[neighbours].max(axis=0)
            else:
                out[i] = H[i]
        return out

    def _multi_scale_embed(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Run n_layers of GCN and concatenate all intermediate representations.
        This gives the model sensitivity to both local (shallow) and global
        (deep) graph structure simultaneously.
        """
        layers = [X_norm]
        H = X_norm
        for _ in range(self.n_layers):
            H = self._propagate_layer(H)
            layers.append(H)
        return np.concatenate(layers, axis=1)   # (n_features, n_stats*(n_layers+1))

    # ------------------------------------------------------------------ #
    # fit / detect
    # ------------------------------------------------------------------ #

    def fit(self, reference: np.ndarray, feature_names: list[str]) -> None:
        self._feature_names = feature_names
        self._P = self._build_prop_matrix(reference)

        X_ref = self._node_features(reference)
        self._X_scale = np.maximum(X_ref.std(axis=0), 1e-9)
        self._ref_emb = self._multi_scale_embed(X_ref / self._X_scale)

        # Bootstrap threshold: 3× within-distribution half-split variance
        half = max(len(reference) // 2, 5)
        emb1 = self._multi_scale_embed(
            self._node_features(reference[:half]) / self._X_scale
        )
        emb2 = self._multi_scale_embed(
            self._node_features(reference[half:]) / self._X_scale
        )
        baseline = float(np.linalg.norm(emb1 - emb2, axis=1).mean())
        self._dist_threshold = max(baseline * 3.0, 0.05)

    def detect(self, current: np.ndarray) -> DetectorResult:
        X_cur = self._node_features(current) / self._X_scale
        cur_emb = self._multi_scale_embed(X_cur)

        per_node = np.linalg.norm(self._ref_emb - cur_emb, axis=1)
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
                "aggregation": self.aggregation,
                "use_edge_weights": self.use_edge_weights,
                "embedding_dim": self._ref_emb.shape[1],
            },
        )