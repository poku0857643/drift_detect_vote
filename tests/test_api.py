"""Integration tests for the FastAPI endpoints.

Uses httpx AsyncClient so tests run without a real server process.
GATConv detector is intentionally left uninitialised (app.state.gat_detector = None)
to avoid checkpoint dependency — the /drift/gnn 503 path is tested separately.
"""
from __future__ import annotations

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from main import app

# ─────────────────────────────────────────────────────────────────── #
# Payloads — use random data that triggers clear majority drift
# ─────────────────────────────────────────────────────────────────── #

_rng = np.random.default_rng(99)
_REF   = _rng.standard_normal((200, 4)).tolist()
_CUR   = (_rng.standard_normal((200, 4)) + 5.0).tolist()   # strongly shifted → drift

DETECT_PAYLOAD = {
    "reference": _REF,
    "current": _CUR,
    "feature_names": ["a", "b", "c", "d"],
    "threshold": 0.05,
    "strategy": "majority",
}

REPORT_PAYLOAD = {**DETECT_PAYLOAD}

GNN_PAYLOAD = {
    "node_features": [[1.0, 2.0, 3.0, 4.0]] * 20,
    "edge_index": [[i for i in range(19)], [(i + 1) for i in range(19)]],
}


# ─────────────────────────────────────────────────────────────────── #
# Health
# ─────────────────────────────────────────────────────────────────── #

@pytest.mark.anyio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────── #
# /api/v1/detect
# ─────────────────────────────────────────────────────────────────── #

@pytest.mark.anyio
async def test_detect_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/detect", json=DETECT_PAYLOAD)
    assert r.status_code == 200


@pytest.mark.anyio
async def test_detect_response_schema():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/detect", json=DETECT_PAYLOAD)
    body = r.json()
    assert "drift_detected" in body
    assert "votes_for_drift" in body
    assert "detector_results" in body
    assert len(body["detector_results"]) == 3


@pytest.mark.anyio
async def test_detect_drift_on_shifted_data():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/detect", json=DETECT_PAYLOAD)
    assert r.json()["drift_detected"] is True


# ─────────────────────────────────────────────────────────────────── #
# /api/v1/report
# ─────────────────────────────────────────────────────────────────── #

@pytest.mark.anyio
async def test_report_creates_html_link():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/report", json=REPORT_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "run_id" in body
    assert "html_url" in body


@pytest.mark.anyio
async def test_report_html_retrieval():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r_report = await c.post("/api/v1/report", json=REPORT_PAYLOAD)
        html_url = r_report.json()["html_url"]
        r_html = await c.get(html_url)
    assert r_html.status_code == 200
    assert "text/html" in r_html.headers["content-type"]


@pytest.mark.anyio
async def test_report_html_404_unknown_run_id():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/report/nonexistent/html")
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────── #
# /drift/gnn  (GATConv endpoint)
# ─────────────────────────────────────────────────────────────────── #

@pytest.mark.anyio
async def test_gnn_503_when_no_checkpoint():
    """Without a checkpoint loaded the endpoint must return 503."""
    # Override state to simulate missing checkpoint
    app.state.gat_detector = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/drift/gnn", json=GNN_PAYLOAD)
    assert r.status_code == 503


@pytest.mark.anyio
async def test_gnn_400_bad_node_features():
    app.state.gat_detector = None  # ensure 400 path is tested independently
    # Inject a mock detector so we can reach the validation code
    from unittest.mock import MagicMock
    mock_detector = MagicMock()
    mock_detector.detect.return_value = {
        "drift_detected": False,
        "signals": {
            "mmd2": 0.0, "attention_ks_stat": 0.0, "attention_ks_p": 1.0,
            "cosine_distance": 0.0, "gae_reconstruction_loss": 0.5,
            "gae_reconstruction_delta": 0.0,
        },
        "thresholds": {"mmd2": 0.05, "ks_p": 0.05},
        "ref_recon_loss": 0.5,
    }
    app.state.gat_detector = mock_detector

    bad_payload = {"node_features": [1.0, 2.0, 3.0], "edge_index": [[0], [1]]}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/drift/gnn", json=bad_payload)
    assert r.status_code == 422   # Pydantic validation error for wrong shape

    # clean up
    app.state.gat_detector = None