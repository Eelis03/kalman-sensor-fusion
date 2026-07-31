"""Consistency testing, accuracy metrics, and figures.

This layer reads traces and never writes to them. It is the only layer that
imports matplotlib.
"""

from __future__ import annotations

from sensor_fusion.analysis.consistency import (
    ConsistencyReport,
    Verdict,
    chi2_interval,
    consistency_report,
)
from sensor_fusion.analysis.metrics import ErrorSummary, rmse, summarize
from sensor_fusion.analysis.report import FilterAssessment, assess

__all__ = [
    "ConsistencyReport",
    "ErrorSummary",
    "FilterAssessment",
    "Verdict",
    "assess",
    "chi2_interval",
    "consistency_report",
    "rmse",
    "summarize",
]
