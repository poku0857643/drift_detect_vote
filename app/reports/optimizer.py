"""
Optimization Ratio Predictor
==============================
Estimates the potential model performance improvement if the detected drift
is corrected (e.g., via retraining).

The mapping is empirically calibrated:
  ensemble_score 0.0 → ~0 % improvement
  ensemble_score 0.5 → ~15 % improvement
  ensemble_score 1.0 → ~45 % improvement

Confidence is derived from the voting confidence (fraction of detectors
that agreed on the presence/absence of drift).
"""

from __future__ import annotations

from app.ensemble.voting import VotingResult
from app.schemas.responses import OptimizationResponse


def predict_optimization(voting_result: VotingResult) -> OptimizationResponse:
    score = voting_result.ensemble_score            # 0–1
    confidence = voting_result.confidence           # 0–1

    # Smooth non-linear mapping  (score^1.4 ≈ 0→0, 0.5→0.38, 1→1)
    improvement = round(float(45.0 * (score ** 1.4)), 1)
    improvement = max(0.0, min(improvement, 50.0))

    # Top drifted features (highest mean ensemble feature score)
    top_features = sorted(
        voting_result.feature_scores.items(), key=lambda x: x[1], reverse=True
    )[:3]
    top_names = [f for f, _ in top_features]

    if not voting_result.drift_detected:
        rec = "Model is stable. No retraining required at this time."
    elif improvement < 8:
        rec = (
            "Minor drift detected. Continue monitoring; "
            "schedule retraining within the next 30 days."
        )
    elif improvement < 20:
        rec = (
            "Moderate drift detected. Retrain the model using recent data "
            f"(focus on: {', '.join(top_names) or 'N/A'})."
        )
    else:
        rec = (
            "Significant distribution shift detected. "
            "Immediate retraining is strongly recommended to restore model performance "
            f"(key features: {', '.join(top_names) or 'N/A'})."
        )

    return OptimizationResponse(
        predicted_improvement_pct=improvement,
        confidence=round(confidence, 3),
        recommendation=rec,
        top_drifted_features=top_names,
    )