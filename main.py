from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.database import Base, engine
from app.routers import drift, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise SQLite schema on startup
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="DriftDetect API",
    description=(
        "Detect data drift using a three-model voting ensemble: "
        "**GNN** (graph neural network), **Statistical** (KS test + Bonferroni), "
        "and **Isolation Forest** (ML anomaly scoring). "
        "Results are stored in SQLite and viewable via the metrics dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(drift.router)
app.include_router(metrics.router)


@app.get("/", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "service": "drift-detect",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/api/v1/metrics/dashboard",
    }