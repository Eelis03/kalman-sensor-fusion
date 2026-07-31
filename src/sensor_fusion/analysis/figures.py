"""Figures.

The non-interactive Agg backend is selected before pyplot is imported, so these
functions work under continuous integration with no display attached. Every
function returns a Figure and writes nothing; saving is the caller's decision.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sensor_fusion._types import FloatArray
from sensor_fusion.analysis.consistency import ConsistencyReport
from sensor_fusion.pipeline.montecarlo import MonteCarloResult
from sensor_fusion.pipeline.simulator import Scenario
from sensor_fusion.pipeline.trace import FilterTrace

__all__ = ["consistency_figure", "error_figure", "trajectory_figure"]


def trajectory_figure(
    scenario: Scenario, traces: Sequence[FilterTrace], labels: Sequence[str]
) -> Figure:
    """Plot ground truth against each filter's estimated track."""
    figure, axes = plt.subplots(figsize=(7.0, 6.0))
    axes.plot(
        scenario.truth_cartesian[:, 0],
        scenario.truth_cartesian[:, 1],
        color="0.25",
        linewidth=2.0,
        label="ground truth",
    )
    for trace, label in zip(traces, labels, strict=True):
        estimates = np.array(
            [record.estimate_cartesian[:2] for record in trace.records], dtype=np.float64
        )
        axes.plot(estimates[:, 0], estimates[:, 1], linewidth=1.2, label=label)
    axes.set_xlabel("x position (m)")
    axes.set_ylabel("y position (m)")
    axes.set_title("Estimated track against ground truth")
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(alpha=0.3)
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def error_figure(traces: Sequence[FilterTrace], labels: Sequence[str]) -> Figure:
    """Plot position and velocity error magnitude against time."""
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for trace, label in zip(traces, labels, strict=True):
        times = trace.times
        position = np.linalg.norm(trace.position_error, axis=1)
        velocity = np.linalg.norm(trace.velocity_error, axis=1)
        axes[0].plot(times, position, linewidth=1.0, label=label)
        axes[1].plot(times, velocity, linewidth=1.0, label=label)
    axes[0].set_ylabel("position error (m)")
    axes[1].set_ylabel("velocity error (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title("Error magnitude against time")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def consistency_figure(
    result: MonteCarloResult, reports: Sequence[ConsistencyReport]
) -> Figure:
    """Plot the across-run average of each consistency statistic with its bounds."""
    panels = 1 + len(result.nis)
    figure, axes = plt.subplots(panels, 1, figsize=(8.0, 2.4 * panels), squeeze=False)
    column = axes[:, 0]

    lookup = {report.statistic: report for report in reports}
    nees_report = next((report for report in reports if report.statistic.startswith("NEES")), None)
    samples = result.nees_state if result.native_nees_available else result.nees_cartesian
    _plot_statistic(column[0], result.times, samples, nees_report, "NEES")

    for index, (name, values) in enumerate(sorted(result.nis.items()), start=1):
        _plot_statistic(
            column[index],
            result.nis_times[name],
            np.asarray(values),
            lookup.get(f"NIS ({name})"),
            f"NIS {name}",
        )

    column[-1].set_xlabel("time (s)")
    figure.suptitle(f"Consistency over {result.runs} Monte Carlo runs")
    figure.tight_layout()
    return figure


def _plot_statistic(
    axis: Axes,
    times: FloatArray,
    samples: FloatArray,
    report: ConsistencyReport | None,
    label: str,
) -> None:
    averaged = np.mean(np.asarray(samples, dtype=np.float64), axis=0)
    axis.plot(times, averaged, linewidth=1.0, color="C0", label=f"{label} run average")
    if report is not None:
        axis.axhline(report.lower, color="C3", linestyle="--", linewidth=1.0, label="95% bounds")
        axis.axhline(report.upper, color="C3", linestyle="--", linewidth=1.0)
        axis.axhline(
            float(report.dof), color="0.4", linestyle=":", linewidth=1.0, label="expected value"
        )
        axis.set_title(report.summary(), fontsize=7)
    axis.set_ylabel(label)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right", fontsize=7)
