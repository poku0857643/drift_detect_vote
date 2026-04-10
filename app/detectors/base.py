from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _to_python(obj: Any) -> Any:
    """Recursively coerce numpy scalars/booleans to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class DetectorResult:
    detector_name: str
    drift_detected: bool
    score: float                          # 0–1, higher = more drift
    threshold: float
    p_value: float | None = None
    feature_scores: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Coerce numpy scalars to plain Python types so Pydantic never
        # encounters numpy.bool_ / numpy.float64 during serialisation.
        self.drift_detected = bool(self.drift_detected)
        self.score = float(self.score)
        self.threshold = float(self.threshold)
        if self.p_value is not None:
            self.p_value = float(self.p_value)
        self.feature_scores = {k: float(v) for k, v in self.feature_scores.items()}
        self.meta = _to_python(self.meta)


class DriftDetector(ABC):
    """Abstract base class for all drift detectors."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fit(self, reference: np.ndarray, feature_names: list[str]) -> None:
        """Fit on reference (baseline) data."""

    @abstractmethod
    def detect(self, current: np.ndarray) -> DetectorResult:
        """Return drift result comparing current to fitted reference."""