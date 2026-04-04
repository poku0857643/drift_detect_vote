"""
Metrics & Dashboard Router
===========================
Endpoints for viewing stored run data, detector performance,
and optimization predictions from the database.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.db.crud import get_performance_summary, get_run, list_runs
from app.db.database import get_db
from app.db.models import DriftRun
from app.schemas.responses import PerformanceResponse, RunSummaryResponse

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics & Performance"])


def _run_to_summary(run: DriftRun) -> RunSummaryResponse:
    return RunSummaryResponse(
        run_id=run.id,
        timestamp=run.timestamp.isoformat(),
        drift_detected=run.drift_detected,
        ensemble_score=run.ensemble_score,
        votes_for_drift=run.votes_for_drift,
        total_detectors=run.total_detectors,
        reference_samples=run.reference_samples,
        current_samples=run.current_samples,
        n_features=run.n_features,
        predicted_improvement_pct=(
            run.optimization.predicted_improvement_pct if run.optimization else 0.0
        ),
    )


@router.get(
    "",
    response_model=list[RunSummaryResponse],
    summary="List all drift detection runs",
)
async def list_runs_endpoint(
    skip: int = 0, limit: int = 50, db: Session = Depends(get_db)
) -> list[RunSummaryResponse]:
    """Returns a paginated list of all stored drift detection runs."""
    runs = list_runs(db, skip=skip, limit=limit)
    return [_run_to_summary(r) for r in runs]


@router.get(
    "/performance",
    response_model=PerformanceResponse,
    summary="Aggregated detector performance across all runs",
)
async def get_performance(db: Session = Depends(get_db)) -> PerformanceResponse:
    """
    Returns aggregated statistics: drift rate, detector agreement,
    average drift score, and the most frequently drifted features.
    """
    data = get_performance_summary(db)
    return PerformanceResponse(**data)


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Interactive metrics dashboard",
)
async def metrics_dashboard(db: Session = Depends(get_db)) -> HTMLResponse:
    """HTML dashboard showing historical runs and performance trends."""
    runs = list_runs(db, limit=100)
    perf = get_performance_summary(db)
    html = _render_dashboard(runs, perf)
    return HTMLResponse(content=html)


@router.get(
    "/{run_id}",
    response_model=RunSummaryResponse,
    summary="Get a single run's metrics",
)
async def get_run_metrics(
    run_id: str, db: Session = Depends(get_db)
) -> RunSummaryResponse:
    run = get_run(db, run_id)
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found.")
    return _run_to_summary(run)


# ------------------------------------------------------------------ #
# Dashboard HTML renderer
# ------------------------------------------------------------------ #

def _render_dashboard(runs: list[DriftRun], perf: dict) -> str:
    drift_color   = "#dc3545"
    stable_color  = "#28a745"
    warn_color    = "#fd7e14"

    drift_rate_pct = int(perf["drift_rate"] * 100)
    agreement_pct  = int(perf["detector_agreement_rate"] * 100)

    # Runs table rows
    table_rows = ""
    for r in runs:
        color = drift_color if r.drift_detected else stable_color
        label = "DRIFT" if r.drift_detected else "STABLE"
        imp   = r.optimization.predicted_improvement_pct if r.optimization else 0.0
        table_rows += f"""
        <tr>
          <td><a href="/api/v1/report/{r.id}/html" style="color:#007bff">{r.id}</a></td>
          <td>{r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else "—"}</td>
          <td><span style="background:{color};color:#fff;padding:2px 8px;
                           border-radius:10px;font-size:.78rem">{label}</span></td>
          <td>{r.ensemble_score:.3f}</td>
          <td>{r.votes_for_drift}/{r.total_detectors}</td>
          <td>{r.reference_samples} / {r.current_samples}</td>
          <td>{r.n_features}</td>
          <td style="color:{warn_color if imp > 10 else stable_color}">{imp:.1f}%</td>
        </tr>"""

    top_features_html = " &nbsp;·&nbsp; ".join(
        f"<strong>{f}</strong>" for f in (perf["most_drifted_features"] or ["—"])
    )

    no_data_msg = (
        '<tr><td colspan="8" style="text-align:center;padding:24px;color:#999">'
        'No runs yet. POST to <code>/api/v1/report</code> to create one.</td></tr>'
        if not runs
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>DriftDetect — Metrics Dashboard</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
          background:#f4f6f9;color:#333}}
    .wrap{{max-width:1000px;margin:36px auto;padding:0 20px}}
    h1{{font-size:1.6rem;font-weight:700;margin-bottom:6px}}
    p.sub{{color:#888;font-size:.9rem;margin-bottom:24px}}
    .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px}}
    .kpi{{background:#fff;padding:18px;border-radius:10px;text-align:center;
          box-shadow:0 1px 4px rgba(0,0,0,.08)}}
    .kpi .val{{font-size:1.8rem;font-weight:700}}
    .kpi .lbl{{font-size:.78rem;color:#888;margin-top:4px}}
    .card{{background:#fff;border-radius:10px;padding:20px;
           box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px;overflow-x:auto}}
    h2{{font-size:1rem;font-weight:600;color:#444;margin-bottom:14px}}
    table{{width:100%;border-collapse:collapse;font-size:.87rem}}
    th{{background:#f8f9fa;padding:8px 10px;border-bottom:2px solid #dee2e6;
        font-weight:600;text-align:left;white-space:nowrap}}
    td{{padding:8px 10px;border-bottom:1px solid #eee;white-space:nowrap}}
    tr:hover td{{background:#fafbfc}}
    .feat{{background:#e8f4fd;border-left:4px solid #007bff;
           padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:20px}}
    .footer{{text-align:center;padding:24px 0 8px;color:#bbb;font-size:.8rem}}
  </style>
</head>
<body>
<div class="wrap">
  <h1>DriftDetect — Metrics Dashboard</h1>
  <p class="sub">Real-time view of stored detection runs, model performance, and optimization signals.</p>

  <div class="grid">
    <div class="kpi">
      <div class="val" style="color:#333">{perf["total_runs"]}</div>
      <div class="lbl">Total Runs</div>
    </div>
    <div class="kpi">
      <div class="val" style="color:{drift_color if drift_rate_pct > 50 else stable_color}">
        {drift_rate_pct}%
      </div>
      <div class="lbl">Drift Rate</div>
    </div>
    <div class="kpi">
      <div class="val" style="color:#6610f2">{agreement_pct}%</div>
      <div class="lbl">Detector Agreement</div>
    </div>
    <div class="kpi">
      <div class="val" style="color:{warn_color}">{perf["avg_predicted_improvement_pct"]}%</div>
      <div class="lbl">Avg. Optimization Potential</div>
    </div>
  </div>

  <div class="feat">
    <strong>Most Frequently Drifted Features:</strong> &nbsp; {top_features_html}
  </div>

  <div class="card">
    <h2>Detection Runs (latest first)</h2>
    <table>
      <thead>
        <tr>
          <th>Run ID</th><th>Timestamp</th><th>Status</th>
          <th>Score</th><th>Votes</th><th>Samples (Ref/Cur)</th>
          <th>Features</th><th>Opt. Potential</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
        {no_data_msg}
      </tbody>
    </table>
  </div>

  <div class="footer">
    DriftDetect API &nbsp;·&nbsp; Avg ensemble score: {perf["avg_ensemble_score"]:.4f}
  </div>
</div>
</body>
</html>"""