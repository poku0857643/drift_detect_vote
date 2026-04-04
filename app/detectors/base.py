from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectorResult:
    detector_name: str
    drift_detected: bool
    score: float                          # 0–1, higher = more drift
    threshold: float
    p_value: float | None = None
    feature_scores: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


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