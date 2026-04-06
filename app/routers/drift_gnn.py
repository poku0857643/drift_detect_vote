"""
POST /drift/gnn
================
GATConv drift detection endpoint. Accepts a production graph (node features
+ edge index), runs the four-signal GATDriftDetector, and writes metrics to
GCP Cloud Monitoring asynchronously.

The detector and GCP monitor are loaded once at app lifespan startup and
stored in app.state — never re-initialised per request.
"""

from __future__ import annotations

import uuid
import logging

import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift", tags=["GNN Drift (GATConv)"])


# ─────────────────────────────────────────────────────────────────── #
# Schemas
# ─────────────────────────────────────────────────────────────────── #

class GNNDetectRequest(BaseModel):
    node_features: list[list[float]] = Field(
        ..., description="Node feature matrix — list of N rows, each with F floats"
    )
    edge_index: list[list[int]] = Field(
        ..., description="Edge index — [[src, ...], [dst, ...]], shape (2, E)"
    )


class GNNDriftSignals(BaseModel):
    mmd2: float
    attention_ks_stat: float
    attention_ks_p: float
    cosine_distance: float
    gae_reconstruction_loss: float
    gae_reconstruction_delta: float


class GNNDriftResponse(BaseModel):
    run_id: str
    drift_detected: bool
    signals: GNNDriftSignals
    thresholds: dict[str, float]
    ref_recon_loss: float
    gcp_logged: bool


# ─────────────────────────────────────────────────────────────────── #
# Endpoint
# ─────────────────────────────────────────────────────────────────── #

@router.post("/gnn", response_model=GNNDriftResponse, summary="GATConv four-signal drift detection")
async def detect_gnn(
    req: GNNDetectRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> GNNDriftResponse:
    """
    Run the GATConv detector on a production graph.

    - Returns all four drift signals (MMD², attention KS, cosine, GAE loss).
    - Writes metrics to GCP Cloud Monitoring asynchronously.
    - Returns 503 if the detector was not initialised at startup
      (GCS checkpoint unavailable).
    """
    detector = getattr(request.app.state, "gat_detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "GATConv detector not initialised. "
                "Ensure GCS_REFERENCE_GRAPH_PATH is set and the checkpoint is reachable."
            ),
        )

    # Build production graph tensor
    try:
        x = torch.tensor(req.node_features, dtype=torch.float)
        ei = torch.tensor(req.edge_index, dtype=torch.long)
    except Exception as exc:
        raise HTTPException(400, f"Invalid graph data: {exc}") from exc

    if x.dim() != 2:
        raise HTTPException(400, "node_features must be a 2-D matrix (N × F).")
    if ei.shape[0] != 2:
        raise HTTPException(400, "edge_index must have shape (2, E).")

    prod_graph = Data(x=x, edge_index=ei)

    # Run detection
    result = detector.detect(prod_graph)
    run_id = uuid.uuid4().hex[:8]

    # Write to GCP in background — never blocks the response
    gcp_monitor = getattr(request.app.state, "gcp_monitor", None)
    if gcp_monitor:
        metrics_to_log = {
            **result["signals"],
            "drift_detected": float(result["drift_detected"]),
        }
        background_tasks.add_task(gcp_monitor.write, metrics_to_log, run_id)
        gcp_logged = True
    else:
        gcp_logged = False

    return GNNDriftResponse(
        run_id=run_id,
        drift_detected=result["drift_detected"],
        signals=GNNDriftSignals(**result["signals"]),
        thresholds=result["thresholds"],
        ref_recon_loss=result["ref_recon_loss"],
        gcp_logged=gcp_logged,
    )