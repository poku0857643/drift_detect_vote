from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.db.crud import save_run
from app.db.database import get_db
from app.ensemble.voting import VotingEnsemble, VotingResult
from app.reports.optimizer import predict_optimization
from app.reports.renderer import render_report
from app.reports.store import StoredReport, report_store
from app.schemas.requests import DetectRequest, ReportRequest
from app.schemas.responses import (
    DetectorResultResponse,
    OptimizationResponse,
    ReportSummaryResponse,
    VotingResultResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Drift Detection"])


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _to_voting_response(vr: VotingResult) -> VotingResultResponse:
    return VotingResultResponse(
        drift_detected=vr.drift_detected,
        strategy=vr.strategy,
        votes_for_drift=vr.votes_for_drift,
        total_detectors=vr.total_detectors,
        confidence=vr.confidence,
        ensemble_score=vr.ensemble_score,
        feature_scores=vr.feature_scores,
        detector_results=[
            DetectorResultResponse(
                detector_name=dr.detector_name,
                drift_detected=dr.drift_detected,
                score=dr.score,
                threshold=dr.threshold,
                p_value=dr.p_value,
                feature_scores=dr.feature_scores,
                meta=dr.meta,
            )
            for dr in vr.detector_results
        ],
    )


def _run_detection(req: DetectRequest) -> tuple[VotingResult, np.ndarray, np.ndarray]:
    ref = np.array(req.reference, dtype=float)
    cur = np.array(req.current, dtype=float)
    names = req.resolved_feature_names

    ensemble = VotingEnsemble(threshold=req.threshold, strategy=req.strategy)
    ensemble.fit(ref, names)
    voting_result = ensemble.detect(cur)
    return voting_result, ref, cur


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.post(
    "/detect",
    response_model=VotingResultResponse,
    summary="Detect data drift (no persistence)",
)
async def detect(req: DetectRequest) -> VotingResultResponse:
    """
    Run all three detectors (GNN, Statistical, Isolation Forest) and return
    their combined voting result.  Nothing is stored in the database.
    """
    voting_result, _, _ = _run_detection(req)
    return _to_voting_response(voting_result)


@router.post(
    "/report",
    response_model=ReportSummaryResponse,
    summary="Detect drift, persist result, and generate an HTML report",
)
async def create_report(
    req: ReportRequest, db: Session = Depends(get_db)
) -> ReportSummaryResponse:
    """
    Runs drift detection, persists the run to SQLite, generates an HTML
    report, and returns a summary with a link to the HTML report.
    """
    voting_result, ref, cur = _run_detection(req)
    names = req.resolved_feature_names
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    optimization = predict_optimization(voting_result)

    html = render_report(
        run_id=run_id,
        timestamp=timestamp,
        voting_result=voting_result,
        optimization=optimization,
        reference_samples=len(ref),
        current_samples=len(cur),
        feature_names=names,
    )

    report_store.save(
        StoredReport(
            run_id=run_id,
            timestamp=timestamp,
            html=html,
            drift_detected=voting_result.drift_detected,
            ensemble_score=voting_result.ensemble_score,
        )
    )

    save_run(
        db=db,
        run_id=run_id,
        voting_result=voting_result,
        reference_samples=int(ref.shape[0]),
        current_samples=int(cur.shape[0]),
        feature_names=names,
        improvement_pct=optimization.predicted_improvement_pct,
        confidence=optimization.confidence,
        recommendation=optimization.recommendation,
        top_features=optimization.top_drifted_features,
    )

    return ReportSummaryResponse(
        run_id=run_id,
        timestamp=timestamp,
        drift_detected=voting_result.drift_detected,
        ensemble_score=voting_result.ensemble_score,
        html_url=f"/api/v1/report/{run_id}/html",
        voting_result=_to_voting_response(voting_result),
        optimization=optimization,
    )


@router.get(
    "/report/{run_id}/html",
    response_class=HTMLResponse,
    summary="View HTML drift report",
)
async def get_report_html(run_id: str) -> HTMLResponse:
    stored = report_store.get(run_id)
    if not stored:
        raise HTTPException(404, f"Report '{run_id}' not found (in-memory store).")
    return HTMLResponse(content=stored.html)