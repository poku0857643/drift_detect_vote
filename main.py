from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.db.database import Base, engine
from app.gcp.monitoring import GCPDriftMonitor
from app.routers import drift, metrics
from app.routers import drift_gnn

logger = logging.getLogger(__name__)


def _load_gat_detector(checkpoint_path: str | None):
    """Load GATDriftDetector from a local path or GCS URI. Returns None on failure."""
    if not checkpoint_path:
        logger.warning("GCS_REFERENCE_GRAPH_PATH not set — /drift/gnn will return 503.")
        return None
    try:
        from app.detectors.gnn_gat import GATCheckpoint, GATDriftDetector
        if checkpoint_path.startswith("gs://"):
            from google.cloud import storage
            import io
            bucket, blob_path = checkpoint_path[5:].split("/", 1)
            buf = io.BytesIO(storage.Client().bucket(bucket).blob(blob_path).download_as_bytes())
            ckpt = GATCheckpoint.load(buf)
        else:
            ckpt = GATCheckpoint.load(checkpoint_path)
        logger.info("GATDriftDetector loaded from %s", checkpoint_path)
        return GATDriftDetector(ckpt)
    except Exception as exc:
        logger.error("Failed to load GATDriftDetector: %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. SQLite schema
    Base.metadata.create_all(bind=engine)

    # 2. GATConv detector (loaded once — never per-request)
    app.state.gat_detector = _load_gat_detector(
        os.getenv("GCS_REFERENCE_GRAPH_PATH", "checkpoints/gat_checkpoint.pt")
    )

    # 3. GCP Cloud Monitoring (disabled gracefully when env var absent)
    app.state.gcp_monitor = GCPDriftMonitor(os.getenv("GCP_PROJECT_ID"))

    yield


app = FastAPI(
    title="DriftDetect API",
    description=(
        "Three-model voting ensemble for ML drift detection — "
        "**GNN** (GATConv + numpy GCN), **Statistical** (KS + PSI + Wasserstein), "
        "**Isolation Forest** (+ LOF + SHAP). "
        "Graph drift via `/drift/gnn` with GCP Cloud Monitoring. "
        "Tabular drift via `/api/v1/report` with SQLite persistence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(drift.router)
app.include_router(metrics.router)
app.include_router(drift_gnn.router)


@app.get("/", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "service": "drift-detect",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/api/v1/metrics/dashboard",
        "gnn_drift": "/drift/gnn",
    }