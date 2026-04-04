from __future__ import annotations

from pydantic import BaseModel

from app.ensemble.voting import VoteStrategy


class DetectorResultResponse(BaseModel):
    detector_name: str
    drift_detected: bool
    score: float
    threshold: float
    p_value: float | None = None
    feature_scores: dict[str, float]
    meta: dict


class VotingResultResponse(BaseModel):
    drift_detected: bool
    strategy: VoteStrategy
    votes_for_drift: int
    total_detectors: int
    confidence: float
    ensemble_score: float
    detector_results: list[DetectorResultResponse]
    feature_scores: dict[str, float]


class OptimizationResponse(BaseModel):
    predicted_improvement_pct: float
    confidence: float
    recommendation: str
    top_drifted_features: list[str]


class ReportSummaryResponse(BaseModel):
    run_id: str
    timestamp: str
    drift_detected: bool
    ensemble_score: float
    html_url: str
    voting_result: VotingResultResponse
    optimization: OptimizationResponse


class RunSummaryResponse(BaseModel):
    run_id: str
    timestamp: str
    drift_detected: bool
    ensemble_score: float
    votes_for_drift: int
    total_detectors: int
    reference_samples: int
    current_samples: int
    n_features: int
    predicted_improvement_pct: float


class PerformanceResponse(BaseModel):
    total_runs: int
    drift_detected_count: int
    drift_rate: float
    detector_agreement_rate: float
    avg_ensemble_score: float
    avg_predicted_improvement_pct: float
    most_drifted_features: list[str]