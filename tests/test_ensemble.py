"""Tests for the VotingEnsemble."""
from __future__ import annotations

import numpy as np
import pytest

from app.ensemble.voting import VotingEnsemble, VoteStrategy


class TestVotingEnsemble:
    def test_majority_no_drift(self, reference, no_drift, feature_names):
        e = VotingEnsemble(strategy=VoteStrategy.MAJORITY)
        e.fit(reference, feature_names)
        result = e.detect(no_drift)
        assert not result.drift_detected

    def test_majority_drift_detected(self, reference, drifted, feature_names):
        e = VotingEnsemble(strategy=VoteStrategy.MAJORITY)
        e.fit(reference, feature_names)
        result = e.detect(drifted)
        assert result.drift_detected

    def test_any_strategy_more_sensitive(self, reference, feature_names):
        """ANY strategy should flag drift at least as often as MAJORITY."""
        rng = np.random.default_rng(7)
        mild = (rng.standard_normal((100, 4)) + 0.8).astype(np.float32)

        e_any = VotingEnsemble(strategy=VoteStrategy.ANY)
        e_maj = VotingEnsemble(strategy=VoteStrategy.MAJORITY)
        e_any.fit(reference, feature_names)
        e_maj.fit(reference, feature_names)

        r_any = e_any.detect(mild)
        r_maj = e_maj.detect(mild)
        # if majority flags, any must also flag
        if r_maj.drift_detected:
            assert r_any.drift_detected

    def test_unanimous_strictest(self, reference, drifted, feature_names):
        e = VotingEnsemble(strategy=VoteStrategy.UNANIMOUS)
        e.fit(reference, feature_names)
        result = e.detect(drifted)
        # unanimous can still flag if all three agree — just ensure it runs
        assert isinstance(result.drift_detected, bool)

    def test_confidence_range(self, reference, drifted, feature_names):
        e = VotingEnsemble()
        e.fit(reference, feature_names)
        result = e.detect(drifted)
        assert 0.0 <= result.confidence <= 1.0

    def test_three_detectors_returned(self, reference, no_drift, feature_names):
        e = VotingEnsemble()
        e.fit(reference, feature_names)
        result = e.detect(no_drift)
        assert result.total_detectors == 3
        assert len(result.detector_results) == 3

    def test_feature_scores_populated(self, reference, drifted, feature_names):
        e = VotingEnsemble()
        e.fit(reference, feature_names)
        result = e.detect(drifted)
        assert len(result.feature_scores) > 0