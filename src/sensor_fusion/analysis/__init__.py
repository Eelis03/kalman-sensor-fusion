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
from sensor_fusion.analysis.whiteness import (
    DEFAULT_LAGS,
    Whiteness,
    WhitenessReport,
    whiteness_report,
)

__all__ = [
    "DEFAULT_LAGS",
    "ConsistencyReport",
    "ErrorSummary",
    "FilterAssessment",
    "Verdict",
    "Whiteness",
    "WhitenessReport",
    "assess",
    "chi2_interval",
    "consistency_report",
    "rmse",
    "summarize",
    "whiteness_report",
]
