"""
HTML Report Renderer
=====================
Generates a self-contained HTML report (inline CSS, no external deps).
"""

from __future__ import annotations

from app.ensemble.voting import VotingResult
from app.schemas.responses import OptimizationResponse


# ------------------------------------------------------------------ #
# Colour palette
# ------------------------------------------------------------------ #
_DRIFT_COLOR   = "#dc3545"
_SAFE_COLOR    = "#28a745"
_WARN_COLOR    = "#fd7e14"
_NEUTRAL_COLOR = "#6c757d"


def _bar(value: float, max_val: float, color: str, height: int = 12) -> str:
    pct = int(min(value / max(max_val, 1e-9), 1.0) * 100)
    return (
        f'<div style="background:#e9ecef;border-radius:4px;height:{height}px;overflow:hidden">'
        f'<div style="width:{pct}%;height:{height}px;background:{color};border-radius:4px"></div>'
        f"</div>"
    )


def _detector_card(dr, accent: str) -> str:
    icon = "▲" if dr.drift_detected else "●"
    p_row = (
        f"<tr><td>P-Value</td><td>{dr.p_value:.4f}</td></tr>"
        if dr.p_value is not None
        else ""
    )
    meta_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f if isinstance(v, float) else v}</td></tr>"
        for k, v in dr.meta.items()
        if not isinstance(v, dict) and k not in ("per_feature_p",)
    )
    return f"""
    <div style="border:1px solid #dee2e6;border-left:4px solid {accent};
                border-radius:8px;padding:18px;margin-bottom:14px;background:#fff">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h3 style="margin:0;font-size:1rem">{icon} {dr.detector_name}</h3>
        <span style="background:{accent};color:#fff;padding:3px 10px;
                     border-radius:12px;font-size:.8rem;font-weight:600">
          {"DRIFT" if dr.drift_detected else "STABLE"}
        </span>
      </div>
      <table style="width:100%;margin-top:12px;font-size:.88rem;border-collapse:collapse">
        <tr><td style="padding:3px 6px;color:#666">Score</td>
            <td style="padding:3px 6px">{dr.score:.4f} &nbsp;
              {_bar(dr.score, 1.0, accent, 8)}</td></tr>
        <tr><td style="padding:3px 6px;color:#666">Threshold</td>
            <td style="padding:3px 6px">{dr.threshold:.4f}</td></tr>
        {p_row}
        {meta_rows}
      </table>
    </div>"""


def render_report(
    run_id: str,
    timestamp: str,
    voting_result: VotingResult,
    optimization: OptimizationResponse,
    reference_samples: int,
    current_samples: int,
    feature_names: list[str],
) -> str:
    has_drift = voting_result.drift_detected
    banner_color = _DRIFT_COLOR if has_drift else _SAFE_COLOR
    verdict = "DRIFT DETECTED" if has_drift else "NO DRIFT DETECTED"
    confidence_pct = int(voting_result.confidence * 100)

    # Detector cards
    accents = [_DRIFT_COLOR, _WARN_COLOR, "#007bff"]
    detector_cards = "".join(
        _detector_card(dr, accents[i % len(accents)])
        for i, dr in enumerate(voting_result.detector_results)
    )

    # Feature drift table (top 10 by score)
    sorted_features = sorted(
        voting_result.feature_scores.items(), key=lambda x: x[1], reverse=True
    )[:10]
    max_fscore = max((s for _, s in sorted_features), default=1.0)
    feat_rows = "".join(
        f"""<tr>
          <td style="padding:6px 10px">{fname}</td>
          <td style="padding:6px 10px">{score:.4f}</td>
          <td style="padding:6px 10px;width:40%">{_bar(score, max_fscore, banner_color)}</td>
        </tr>"""
        for fname, score in sorted_features
    )

    # Vote indicators
    vote_circles = "".join(
        f'<span style="display:inline-block;width:28px;height:28px;border-radius:50%;'
        f'background:{"#dc3545" if dr.drift_detected else "#28a745"};'
        f'margin:0 4px;line-height:28px;text-align:center;color:#fff;font-size:.75rem">'
        f'{"D" if dr.drift_detected else "S"}</span>'
        for dr in voting_result.detector_results
    )

    opt_color = (
        _DRIFT_COLOR if optimization.predicted_improvement_pct > 20
        else _WARN_COLOR if optimization.predicted_improvement_pct > 8
        else _SAFE_COLOR
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Drift Report — {run_id}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
          background:#f4f6f9;color:#333;line-height:1.5}}
    .wrap{{max-width:860px;margin:36px auto;padding:0 20px}}
    .banner{{background:{banner_color};color:#fff;padding:28px 32px;border-radius:12px;margin-bottom:24px}}
    .banner h1{{font-size:1.9rem;font-weight:700}}
    .banner p{{margin-top:6px;opacity:.9}}
    .grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}}
    .kpi{{background:#fff;padding:18px;border-radius:10px;text-align:center;
          box-shadow:0 1px 4px rgba(0,0,0,.08)}}
    .kpi .val{{font-size:2rem;font-weight:700;color:{banner_color}}}
    .kpi .lbl{{font-size:.8rem;color:#888;margin-top:4px}}
    h2{{font-size:1.1rem;font-weight:600;color:#444;margin:24px 0 12px}}
    .card{{background:#fff;border-radius:10px;padding:20px;
           box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:16px}}
    table{{width:100%;border-collapse:collapse}}
    th,td{{padding:8px 10px;border-bottom:1px solid #eee;font-size:.88rem}}
    th{{background:#f8f9fa;font-weight:600;text-align:left}}
    .opt-box{{background:#f0fff4;border-left:4px solid {opt_color};
              padding:16px 20px;border-radius:0 8px 8px 0;margin-top:24px}}
    .footer{{text-align:center;padding:28px 0 10px;color:#aaa;font-size:.8rem}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="banner">
    <h1>{verdict}</h1>
    <p>Run&nbsp;<strong>{run_id}</strong> &nbsp;·&nbsp; {timestamp[:19].replace("T"," ")} UTC</p>
  </div>

  <div class="grid3">
    <div class="kpi">
      <div class="val">{voting_result.votes_for_drift}/{voting_result.total_detectors}</div>
      <div class="lbl">Detectors flagged drift</div>
    </div>
    <div class="kpi">
      <div class="val">{voting_result.ensemble_score:.3f}</div>
      <div class="lbl">Ensemble drift score</div>
    </div>
    <div class="kpi">
      <div class="val" style="color:{opt_color}">{optimization.predicted_improvement_pct}%</div>
      <div class="lbl">Est. improvement if retrained</div>
    </div>
  </div>

  <h2>Voting Summary</h2>
  <div class="card">
    <p style="margin-bottom:10px">
      Strategy: <strong>{voting_result.strategy.value.capitalize()}</strong> &nbsp;·&nbsp;
      Confidence: <strong>{confidence_pct}%</strong>
    </p>
    {vote_circles}
    <div style="margin-top:12px">
      {_bar(voting_result.confidence, 1.0, banner_color, 18)}
    </div>
    <p style="margin-top:6px;font-size:.8rem;color:#888">D = Drift &nbsp; S = Stable</p>
  </div>

  <h2>Detector Results</h2>
  {detector_cards}

  <h2>Feature Drift Scores (top {len(sorted_features)})</h2>
  <div class="card">
    <table>
      <thead><tr><th>Feature</th><th>Score</th><th>Intensity</th></tr></thead>
      <tbody>{feat_rows}</tbody>
    </table>
  </div>

  <h2>Dataset Overview</h2>
  <div class="card">
    <table>
      <thead><tr><th>Property</th><th>Reference</th><th>Current</th></tr></thead>
      <tbody>
        <tr><td>Samples</td><td>{reference_samples}</td><td>{current_samples}</td></tr>
        <tr><td>Features</td><td colspan="2">{len(feature_names)}</td></tr>
      </tbody>
    </table>
  </div>

  <div class="opt-box">
    <strong>Optimization Recommendation</strong>
    <p style="margin-top:6px">{optimization.recommendation}</p>
    <p style="margin-top:8px;font-size:.85rem;color:#555">
      Predicted performance improvement if corrected:
      <strong style="color:{opt_color}">{optimization.predicted_improvement_pct}%</strong>
      &nbsp;(confidence: {int(optimization.confidence * 100)}%)
    </p>
  </div>

  <div class="footer">DriftDetect API &nbsp;·&nbsp; run {run_id}</div>
</div>
</body>
</html>"""