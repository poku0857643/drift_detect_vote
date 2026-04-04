"""
SQLAlchemy ORM models.

Tables
------
drift_runs            — one row per detection run
detector_results      — one row per detector per run  (3 per run)
feature_metrics       — one row per feature per run
optimization_predictions — one row per run: predicted improvement ratio
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class DriftRun(Base):
    __tablename__ = "drift_runs"

    id:                Mapped[str]      = mapped_column(String, primary_key=True)
    timestamp:         Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reference_samples: Mapped[int]      = mapped_column(Integer)
    current_samples:   Mapped[int]      = mapped_column(Integer)
    n_features:        Mapped[int]      = mapped_column(Integer)
    feature_names:     Mapped[list]     = mapped_column(JSON)
    voting_strategy:   Mapped[str]      = mapped_column(String)
    drift_detected:    Mapped[bool]     = mapped_column(Boolean)
    votes_for_drift:   Mapped[int]      = mapped_column(Integer)
    total_detectors:   Mapped[int]      = mapped_column(Integer)
    ensemble_score:    Mapped[float]    = mapped_column(Float)
    confidence:        Mapped[float]    = mapped_column(Float)

    detector_results: Mapped[list[DetectorResultDB]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    feature_metrics: Mapped[list[FeatureMetricDB]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    optimization: Mapped[OptimizationPredictionDB | None] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )


class DetectorResultDB(Base):
    __tablename__ = "detector_results"

    id:             Mapped[int]        = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id:         Mapped[str]        = mapped_column(String, ForeignKey("drift_runs.id"))
    detector_name:  Mapped[str]        = mapped_column(String)
    drift_detected: Mapped[bool]       = mapped_column(Boolean)
    score:          Mapped[float]      = mapped_column(Float)
    threshold:      Mapped[float]      = mapped_column(Float)
    p_value:        Mapped[float|None] = mapped_column(Float, nullable=True)
    feature_scores: Mapped[dict]       = mapped_column(JSON)
    meta:           Mapped[dict]       = mapped_column(JSON)

    run: Mapped[DriftRun] = relationship(back_populates="detector_results")


class FeatureMetricDB(Base):
    __tablename__ = "feature_metrics"

    id:             Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id:         Mapped[str]   = mapped_column(String, ForeignKey("drift_runs.id"))
    feature_name:   Mapped[str]   = mapped_column(String)
    drift_score:    Mapped[float] = mapped_column(Float)
    drift_detected: Mapped[bool]  = mapped_column(Boolean)

    run: Mapped[DriftRun] = relationship(back_populates="feature_metrics")


class OptimizationPredictionDB(Base):
    __tablename__ = "optimization_predictions"

    id:                        Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id:                    Mapped[str]   = mapped_column(String, ForeignKey("drift_runs.id"))
    predicted_improvement_pct: Mapped[float] = mapped_column(Float)
    confidence:                Mapped[float] = mapped_column(Float)
    recommendation:            Mapped[str]   = mapped_column(String)
    top_drifted_features:      Mapped[list]  = mapped_column(JSON)

    run: Mapped[DriftRun] = relationship(back_populates="optimization")