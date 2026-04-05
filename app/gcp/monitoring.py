"""
GCP Cloud Monitoring — Drift Metric Writer
==========================================
Writes drift signals as custom time-series metrics to GCP Cloud Monitoring.

Metric namespace: custom.googleapis.com/drift/{signal_name}

IAM requirement: roles/monitoring.metricWriter on the service account.

All writes are intended to be called via FastAPI BackgroundTasks so they
never block the HTTP response.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from google.cloud import monitoring_v3
    from google.protobuf import timestamp_pb2
    _GCP_AVAILABLE = True
except ImportError:
    _GCP_AVAILABLE = False
    logger.warning(
        "google-cloud-monitoring not installed — GCP metric writes will be no-ops. "
        "Install with: pip install google-cloud-monitoring"
    )


METRIC_PREFIX = "custom.googleapis.com/drift"


class GCPDriftMonitor:
    """
    Writes drift signal values as GCP custom metrics.

    Parameters
    ----------
    project_id : str
        GCP project ID.  When None the monitor is disabled (local dev).
    """

    def __init__(self, project_id: str | None) -> None:
        self.project_id = project_id
        self._enabled = bool(project_id and _GCP_AVAILABLE)

        if self._enabled:
            self._client = monitoring_v3.MetricServiceClient()
            self._project_name = f"projects/{project_id}"
            logger.info("GCPDriftMonitor initialised — project=%s", project_id)
        else:
            self._client = None
            logger.info(
                "GCPDriftMonitor disabled — "
                "set GCP_PROJECT_ID env var and install google-cloud-monitoring to enable."
            )

    # ------------------------------------------------------------------ #

    def write(self, metrics: dict[str, float], run_id: str = "") -> bool:
        """
        Write a dict of {signal_name: float_value} as custom time-series.

        Returns True on success, False on failure (never raises — safe for
        BackgroundTasks use).
        """
        if not self._enabled:
            logger.debug("GCP monitoring disabled — skipping write for run_id=%s", run_id)
            return False

        now_sec = int(time.time())
        series = []

        for name, value in metrics.items():
            ts = monitoring_v3.TimeSeries(
                metric=monitoring_v3.types.Metric(
                    type=f"{METRIC_PREFIX}/{name}",
                    labels={"run_id": run_id} if run_id else {},
                ),
                resource=monitoring_v3.types.MonitoredResource(
                    type="global",
                    labels={"project_id": self.project_id},
                ),
                points=[
                    monitoring_v3.types.Point(
                        interval=monitoring_v3.types.TimeInterval(
                            end_time=timestamp_pb2.Timestamp(seconds=now_sec)
                        ),
                        value=monitoring_v3.types.TypedValue(double_value=float(value)),
                    )
                ],
            )
            series.append(ts)

        try:
            self._client.create_time_series(
                name=self._project_name, time_series=series
            )
            logger.info(
                "GCP metrics written — run_id=%s  signals=%s", run_id, list(metrics.keys())
            )
            return True
        except Exception as exc:
            logger.error("GCP write failed — %s", exc)
            return False