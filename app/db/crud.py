from __future__ import annotations

from collections import Counter

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import models
from app.ensemble.voting import VotingResult


def save_run(
    db: Session,
    run_id: str,
    voting_result: VotingResult,
    reference_samples: int,
    current_samples: int,
    feature_names: list[str],
    improvement_pct: float,
    confidence: float,
    recommendation: str,
    top_features: list[str],
) -> models.DriftRun:
    run = models.DriftRun(
        id=run_id,
        reference_samples=reference_samples,
        current_samples=current_samples,
        n_features=len(feature_names),
        feature_names=feature_names,
        voting_strategy=voting_result.strategy.value,
        drift_detected=voting_result.drift_detected,
        votes_for_drift=voting_result.votes_for_drift,
        total_detectors=voting_result.total_detectors,
        ensemble_score=voting_result.ensemble_score,
        confidence=voting_result.confidence,
    )
    db.add(run)

    for dr in voting_result.detector_results:
        db.add(
            models.DetectorResultDB(
                run_id=run_id,
                detector_name=dr.detector_name,
                drift_detected=dr.drift_detected,
                score=dr.score,
                threshold=dr.threshold,
                p_value=dr.p_value,
                feature_scores=dr.feature_scores,
                meta=dr.meta,
            )
        )

    mean_score = voting_result.ensemble_score
    for fname, fscore in voting_result.feature_scores.items():
        db.add(
            models.FeatureMetricDB(
                run_id=run_id,
                feature_name=fname,
                drift_score=fscore,
                drift_detected=fscore > mean_score,
            )
        )

    db.add(
        models.OptimizationPredictionDB(
            run_id=run_id,
            predicted_improvement_pct=improvement_pct,
            confidence=confidence,
            recommendation=recommendation,
            top_drifted_features=top_features,
        )
    )

    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: str) -> models.DriftRun | None:
    return db.get(models.DriftRun, run_id)


def list_runs(
    db: Session, skip: int = 0, limit: int = 50
) -> list[models.DriftRun]:
    return (
        db.query(models.DriftRun)
        .order_by(models.DriftRun.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_performance_summary(db: Session) -> dict:
    total = db.query(func.count(models.DriftRun.id)).scalar() or 0
    detected = (
        db.query(func.count(models.DriftRun.id))
        .filter(models.DriftRun.drift_detected.is_(True))
        .scalar()
        or 0
    )
    avg_score = db.query(func.avg(models.DriftRun.ensemble_score)).scalar() or 0.0

    agreed = sum(
        1
        for run in db.query(models.DriftRun).all()
        if run.votes_for_drift == run.total_detectors or run.votes_for_drift == 0
    )
    agreement_rate = agreed / max(total, 1)

    feature_counter: Counter = Counter()
    for pred in db.query(models.OptimizationPredictionDB).all():
        feature_counter.update(pred.top_drifted_features or [])
    top_features = [f for f, _ in feature_counter.most_common(5)]

    avg_improvement = (
        db.query(func.avg(models.OptimizationPredictionDB.predicted_improvement_pct)).scalar()
        or 0.0
    )

    return {
        "total_runs": total,
        "drift_detected_count": detected,
        "drift_rate": detected / max(total, 1),
        "detector_agreement_rate": round(agreement_rate, 3),
        "avg_ensemble_score": round(float(avg_score), 4),
        "avg_predicted_improvement_pct": round(float(avg_improvement), 1),
        "most_drifted_features": top_features,
    }