"""
Monitoring and observability module for the ML system design framework.

Covers:
- Performance metric tracking
- Data drift detection (population stability index, KS test)
- Model drift detection
- Alert mechanisms
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import MonitoringConfig
from utils import ensure_directory, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Metric store
# ---------------------------------------------------------------------------


class MetricStore:
    """
    Lightweight append-only metrics store backed by a CSV file.

    Parameters
    ----------
    config:
        Monitoring configuration.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        ensure_directory(config.metrics_path)
        self._metrics_file = os.path.join(config.metrics_path, "metrics.csv")
        self._in_memory: List[Dict[str, Any]] = []
        self._log = logging.getLogger(__name__)

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Append a metric data point."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "name": name,
            "value": value,
            "tags": json.dumps(tags or {}),
        }
        self._in_memory.append(entry)
        self._flush_one(entry)
        self._log.debug("Metric %s=%.4f tags=%s", name, value, tags)

    def query(self, name: str) -> List[Dict[str, Any]]:
        """Return all recorded data points for *name*."""
        return [m for m in self._in_memory if m["name"] == name]

    def latest(self, name: str) -> Optional[float]:
        """Return the most recent value for *name*, or *None*."""
        matches = self.query(name)
        return matches[-1]["value"] if matches else None

    # ------------------------------------------------------------------

    def _flush_one(self, entry: Dict[str, Any]) -> None:
        file_exists = os.path.isfile(self._metrics_file)
        with open(self._metrics_file, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)


# ---------------------------------------------------------------------------
# Drift detectors
# ---------------------------------------------------------------------------


class DataDriftDetector:
    """
    Detects distributional shift between a reference and a production dataset.

    Uses the two-sample Kolmogorov-Smirnov test for continuous features and
    the population stability index (PSI) for a quick summary.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self._reference: Optional[pd.DataFrame] = None
        self._log = logging.getLogger(__name__)

    def set_reference(self, df: pd.DataFrame) -> None:
        """Store *df* as the reference (baseline) distribution."""
        self._reference = df.copy()
        self._log.info(
            "Reference dataset set (%d rows, %d columns).", *df.shape
        )

    def detect(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare *production_df* against the reference dataset.

        Returns
        -------
        dict
            Per-column drift results and an overall ``drift_detected`` flag.
        """
        if self._reference is None:
            raise RuntimeError("Call set_reference() before detect().")

        numeric_cols = [
            c
            for c in self._reference.select_dtypes(include=[np.number]).columns
            if c in production_df.columns
        ]

        results: Dict[str, Dict[str, Any]] = {}
        for col in numeric_cols:
            ref_vals = self._reference[col].dropna().values
            prod_vals = production_df[col].dropna().values
            ks_stat, p_value = stats.ks_2samp(ref_vals, prod_vals)
            drifted = p_value < self.config.drift_significance_level
            results[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drifted": drifted,
            }

        n_drifted = sum(1 for r in results.values() if r["drifted"])
        overall = n_drifted > 0
        self._log.info(
            "Drift detection: %d/%d columns drifted.", n_drifted, len(results)
        )
        return {"drift_detected": overall, "columns": results}

    @staticmethod
    def psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
        """
        Compute the Population Stability Index (PSI).

        PSI < 0.1  → no significant shift
        PSI < 0.25 → moderate shift
        PSI >= 0.25 → significant shift
        """
        eps = 1e-8
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        ref_counts = np.histogram(reference, bins=breakpoints)[0] + eps
        prod_counts = np.histogram(production, bins=breakpoints)[0] + eps

        ref_pct = ref_counts / ref_counts.sum()
        prod_pct = prod_counts / prod_counts.sum()

        psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return psi


class ModelDriftDetector:
    """
    Detects model performance degradation over time.

    Compares a rolling window of recent performance against a baseline.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self._baseline_metric: Optional[float] = None
        self._history: List[float] = []
        self._log = logging.getLogger(__name__)

    def set_baseline(self, metric_value: float) -> None:
        """Record the baseline (post-training) performance metric."""
        self._baseline_metric = metric_value
        self._log.info("Baseline metric set to %.4f.", metric_value)

    def update(self, metric_value: float) -> Dict[str, Any]:
        """
        Record a new performance observation and check for drift.

        Returns
        -------
        dict
            ``{"drift_detected": bool, "relative_drop": float}``
        """
        self._history.append(metric_value)

        if self._baseline_metric is None or self._baseline_metric == 0:
            return {"drift_detected": False, "relative_drop": 0.0}

        window = self._history[-self.config.drift_detection_window:]
        avg = np.mean(window)
        relative_drop = (self._baseline_metric - avg) / self._baseline_metric
        drifted = bool(relative_drop > self.config.model_retraining_trigger_threshold)

        if drifted:
            self._log.warning(
                "Model drift detected! Relative drop: %.2f%% (threshold: %.2f%%)",
                relative_drop * 100,
                self.config.model_retraining_trigger_threshold * 100,
            )
        return {"drift_detected": drifted, "relative_drop": float(relative_drop)}


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------


class AlertManager:
    """
    Routes alert notifications to configured channels.

    Currently supports callback-based alerting; extend for email / Slack.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self._handlers: List[Callable[[str, str, Dict[str, Any]], None]] = []
        self._alerts: List[Dict[str, Any]] = []
        self._log = logging.getLogger(__name__)

    def register_handler(
        self, handler: Callable[[str, str, Dict[str, Any]], None]
    ) -> None:
        """Register a callable ``handler(level, message, metadata)``."""
        self._handlers.append(handler)

    def alert(
        self,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Dispatch an alert to all registered handlers.

        Parameters
        ----------
        level:
            Severity — ``"info"``, ``"warning"``, or ``"critical"``.
        message:
            Human-readable description.
        metadata:
            Optional structured data attached to the alert.
        """
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "metadata": metadata or {},
        }
        self._alerts.append(record)
        self._log.log(
            getattr(logging, level.upper(), logging.INFO),
            "[ALERT] %s",
            message,
        )
        for handler in self._handlers:
            try:
                handler(level, message, record["metadata"])
            except Exception:
                self._log.error(
                    "Alert handler failed.", exc_info=True
                )

    @property
    def alerts(self) -> List[Dict[str, Any]]:
        return list(self._alerts)


# ---------------------------------------------------------------------------
# Monitoring orchestrator
# ---------------------------------------------------------------------------


class MonitoringSystem:
    """
    High-level entry point that wires together the metric store,
    drift detectors, and alert manager.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or MonitoringConfig()
        self.metrics = MetricStore(self.config)
        self.data_drift = DataDriftDetector(self.config)
        self.model_drift = ModelDriftDetector(self.config)
        self.alerts = AlertManager(self.config)
        self._log = logging.getLogger(__name__)

    def record_prediction(
        self,
        features: Dict[str, Any],
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log a single prediction event and optionally record accuracy."""
        if latency_ms is not None:
            self.metrics.record("prediction_latency_ms", latency_ms)
        if ground_truth is not None:
            correct = int(prediction == ground_truth)
            self.metrics.record("prediction_correct", float(correct))

    def check_performance(self, metric_name: str, current_value: float) -> None:
        """
        Record a performance metric and trigger a drift check.

        Fires a critical alert if model drift is detected.
        """
        self.metrics.record(metric_name, current_value)
        result = self.model_drift.update(current_value)
        if result["drift_detected"]:
            self.alerts.alert(
                "critical",
                f"Model performance degraded by "
                f"{result['relative_drop']:.1%} on '{metric_name}'.",
                metadata=result,
            )

    def check_data_drift(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """Run data drift detection and fire a warning alert if drift is found."""
        result = self.data_drift.detect(production_df)
        if result["drift_detected"]:
            n_drifted = sum(
                1 for v in result["columns"].values() if v["drifted"]
            )
            self.alerts.alert(
                "warning",
                f"Data drift detected in {n_drifted} feature(s).",
                metadata=result,
            )
        return result
