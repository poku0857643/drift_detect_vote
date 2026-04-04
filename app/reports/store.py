from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StoredReport:
    run_id: str
    timestamp: str
    html: str
    drift_detected: bool
    ensemble_score: float


class ReportStore:
    """In-memory HTML report store (singleton via module-level instance)."""

    def __init__(self) -> None:
        self._data: dict[str, StoredReport] = {}

    def save(self, report: StoredReport) -> None:
        self._data[report.run_id] = report

    def get(self, run_id: str) -> StoredReport | None:
        return self._data.get(run_id)

    def ids(self) -> list[str]:
        return list(self._data.keys())


# Module-level singleton — imported by routers
report_store = ReportStore()