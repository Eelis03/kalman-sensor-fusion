"""Accuracy metrics computed from a filter trace."""

from __future__ import annotations

from dataclasses import dataclass

from sensor_fusion._math import rmse
from sensor_fusion.pipeline.trace import FilterTrace

__all__ = ["ErrorSummary", "rmse", "summarize"]


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    """Root mean square position and velocity error over one run."""

    label: str
    updates: int
    position_rmse: float
    velocity_rmse: float

    def summary(self) -> str:
        """Return a single line with the label and both errors."""
        return (
            f"{self.label}: position RMSE {self.position_rmse:.4f} m, "
            f"velocity RMSE {self.velocity_rmse:.4f} m/s over {self.updates} updates"
        )


def summarize(trace: FilterTrace, label: str | None = None) -> ErrorSummary:
    """Return the error summary for one trace."""
    return ErrorSummary(
        label=label if label is not None else f"{trace.filter_name}/{trace.motion_name}",
        updates=len(trace.records),
        position_rmse=rmse(trace.position_error),
        velocity_rmse=rmse(trace.velocity_error),
    )
