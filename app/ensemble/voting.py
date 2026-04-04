"""
Voting Ensemble
===============
Combines three drift detectors (GNN, Statistical, Isolation Forest) via a
configurable voting strategy.  The ensemble is the single entry-point for
drift detection across the service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.detectors.base import DetectorResult, DriftDetector
from app.detectors.gnn import GNNDriftDetector
from app.detectors.isolation import IsolationForestDriftDetector
from app.detectors.statistical import StatisticalDriftDetector


class VoteStrategy(str, Enum):
    MAJORITY  = "majority"   # >50 % detectors flag drift  (default)
    UNANIMOUS = "unanimous"  # all detectors must flag drift
    ANY       = "any"        # any single detector is enough


@dataclass
class VotingResult:
    drift_detected: bool
    strategy: VoteStrategy
    votes_for_drift: int
    total_detectors: int
    confidence: float             # votes_for_drift / total_detectors
    ensemble_score: float         # mean detector score (0–1)
    detector_results: list[DetectorResult]
    feature_scores: dict[str, float] = field(default_factory=dict)


class VotingEnsemble:
    """
    Wraps three drift detectors and resolves their votes.

    Usage
    -----
    ensemble = VotingEnsemble(strategy=VoteStrategy.MAJORITY)
    ensemble.fit(reference_array, feature_names)
    result = ensemble.detect(current_array)
    """

    def __init__(
        self,
        threshold: float = 0.05,
        strategy: VoteStrategy = VoteStrategy.MAJORITY,
    ) -> None:
        self.strategy = strategy
        self._detectors: list[DriftDetector] = [
            GNNDriftDetector(threshold=threshold),
            StatisticalDriftDetector(threshold=threshold),
            IsolationForestDriftDetector(threshold=threshold),
        ]

    def fit(self, reference: np.ndarray, feature_names: list[str]) -> None:
        for detector in self._detectors:
            detector.fit(reference, feature_names)

    def detect(self, current: np.ndarray) -> VotingResult:
        results: list[DetectorResult] = [
            d.detect(current) for d in self._detectors
        ]

        votes_for = sum(r.drift_detected for r in results)
        total = len(results)

        if self.strategy == VoteStrategy.MAJORITY:
            final = votes_for > total / 2
        elif self.strategy == VoteStrategy.UNANIMOUS:
            final = votes_for == total
        else:  # ANY
            final = votes_for >= 1

        # Aggregate per-feature scores (mean across detectors that reported them)
        all_features: set[str] = set()
        for r in results:
            all_features.update(r.feature_scores)

        agg_features: dict[str, float] = {
            feat: float(
                np.mean([r.feature_scores.get(feat, 0.0) for r in results])
            )
            for feat in all_features
        }

        return VotingResult(
            drift_detected=final,
            strategy=self.strategy,
            votes_for_drift=votes_for,
            total_detectors=total,
            confidence=votes_for / total,
            ensemble_score=float(np.mean([r.score for r in results])),
            detector_results=results,
            feature_scores=agg_features,
        )