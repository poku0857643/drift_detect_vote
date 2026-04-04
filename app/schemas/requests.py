from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.ensemble.voting import VoteStrategy


class DetectRequest(BaseModel):
    reference: list[list[float]] = Field(
        ..., description="Reference (baseline) dataset — list of samples, each a list of feature values"
    )
    current: list[list[float]] = Field(
        ..., description="Current dataset to test for drift"
    )
    feature_names: list[str] | None = Field(
        None, description="Optional feature names; auto-generated if omitted"
    )
    strategy: VoteStrategy = Field(
        VoteStrategy.MAJORITY,
        description="Voting strategy: majority (≥2/3), unanimous (3/3), any (≥1/3)",
    )
    threshold: float = Field(
        0.05, ge=0.0, le=1.0, description="Statistical significance threshold"
    )

    @model_validator(mode="after")
    def _validate(self) -> "DetectRequest":
        if not self.reference or not self.current:
            raise ValueError("reference and current must be non-empty.")
        n_ref = len(self.reference[0])
        n_cur = len(self.current[0])
        if n_ref != n_cur:
            raise ValueError(
                f"Feature mismatch: reference has {n_ref} features, current has {n_cur}."
            )
        if len(self.reference) < 5 or len(self.current) < 5:
            raise ValueError("At least 5 samples required in each dataset.")
        if self.feature_names and len(self.feature_names) != n_ref:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} != n_features {n_ref}."
            )
        return self

    @property
    def resolved_feature_names(self) -> list[str]:
        n = len(self.reference[0])
        return self.feature_names or [f"feature_{i}" for i in range(n)]


class ReportRequest(DetectRequest):
    """Same as DetectRequest; POST /report also persists the result."""
    pass