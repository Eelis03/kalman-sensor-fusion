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

__all__ = [
    "consistency_figure",
    "error_figure",
    "nees_comparison_figure",
    "nees_panels_figure",
    "trajectory_figure",
]


def _nees_samples(result: MonteCarloResult) -> FloatArray:
    """Return the NEES array the verdict was taken from, native where it exists."""
    return result.nees_state if result.native_nees_available else result.nees_cartesian


def trajectory_figure(
    scenario: Scenario,
    traces: Sequence[FilterTrace],
    labels: Sequence[str],
    figsize: tuple[float, float] = (7.0, 6.0),
) -> Figure:
    """Plot ground truth against each filter's estimated track.

    The sensor sits at the origin and is drawn, because the distance between it
    and the path is a property of the scenario that the accuracy tables cannot
    show. Range, bearing, and range rate all degenerate at that point, so a
    reader is entitled to see how far the target stays from it.
    """
    figure, axes = plt.subplots(figsize=figsize)
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
    truth = np.asarray(scenario.truth_cartesian, dtype=np.float64)
    ranges = np.hypot(truth[:, 0], truth[:, 1])
    nearest = int(np.argmin(ranges))
    axes.plot(
        [0.0, truth[nearest, 0]],
        [0.0, truth[nearest, 1]],
        color="0.15",
        linestyle="--",
        linewidth=0.9,
        label=f"closest approach, {ranges[nearest]:.0f} m",
    )
    axes.plot(
        [0.0], [0.0], marker="x", color="0.15", markersize=9, linestyle="none", label="sensor"
    )
    axes.set_xlabel("x position (m)")
    axes.set_ylabel("y position (m)")
    axes.set_title("Estimated track against ground truth")
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(alpha=0.3)
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def _draw_nees(
    axis: Axes,
    times: FloatArray,
    samples: FloatArray,
    report: ConsistencyReport,
    label: str,
    color: str,
) -> None:
    """Draw one across-run NEES trace onto ``axis``."""
    averaged = np.mean(np.asarray(samples, dtype=np.float64), axis=0)
    axis.plot(times, averaged, linewidth=1.1, color=color, label=label)


def _draw_bounds(axis: Axes, report: ConsistencyReport) -> None:
    """Shade the chi-square interval and mark the expected value."""
    axis.axhspan(report.lower, report.upper, color="0.55", alpha=0.22, linewidth=0.0)
    axis.axhline(float(report.dof), color="0.3", linestyle=":", linewidth=1.0)


def nees_panels_figure(
    results: Sequence[MonteCarloResult],
    reports: Sequence[ConsistencyReport],
    labels: Sequence[str],
    figsize: tuple[float, float] = (7.2, 6.2),
) -> Figure:
    """Stack one NEES panel per campaign, each against its own chi-square band.

    One panel per campaign rather than one shared axis, because the degrees of
    freedom differ between a five-state CTRV filter and a four-state constant
    velocity one, so the bands are not the same band and drawing them as one
    would be a lie about which interval each trace is being judged against.
    """
    panels = len(results)
    figure, axes = plt.subplots(panels, 1, figsize=figsize, squeeze=False, sharex=True)
    column = axes[:, 0]
    for index, (result, report, label) in enumerate(zip(results, reports, labels, strict=True)):
        axis = column[index]
        _draw_bounds(axis, report)
        _draw_nees(axis, result.times, _nees_samples(result), report, label, f"C{index}")
        axis.set_ylabel("NEES")
        axis.grid(alpha=0.25)
        axis.set_title(
            f"{label}: mean {report.mean:.3f} against expected {report.dof}, "
            f"{report.verdict.value}",
            fontsize=8,
        )
        upper = max(float(np.max(np.mean(_nees_samples(result), axis=0))), report.upper)
        axis.set_ylim(0.0, min(upper, 4.0 * report.upper) * 1.08)
    column[-1].set_xlabel("time (s)")
    figure.suptitle(
        "Across-run mean NEES with the 95 percent chi-square interval shaded", fontsize=10
    )
    figure.tight_layout()
    return figure


def nees_comparison_figure(
    results: Sequence[MonteCarloResult],
    reports: Sequence[ConsistencyReport],
    labels: Sequence[str],
    title: str,
    figsize: tuple[float, float] = (7.2, 3.6),
    log_scale: bool = False,
) -> Figure:
    """Overlay several NEES traces that share one chi-square band on one axis.

    Only valid when every campaign has the same degrees of freedom and the same
    run count, which is checked, since otherwise the single shaded band would
    not apply to every trace drawn on it.

    ``log_scale`` is for the case where one filter leaves the interval by an
    order of magnitude and another never leaves it at all. On a linear axis the
    excursion flattens the interval into a line at the bottom of the plot and the
    reader can no longer see what either trace is being judged against.
    """
    dofs = {report.dof for report in reports}
    runs = {report.runs for report in reports}
    if len(dofs) != 1 or len(runs) != 1:
        raise ValueError("one shared band requires one shared dof and one shared run count")

    figure, axes = plt.subplots(figsize=figsize)
    _draw_bounds(axes, reports[0])
    for index, (result, report, label) in enumerate(zip(results, reports, labels, strict=True)):
        _draw_nees(
            axes,
            result.times,
            _nees_samples(result),
            report,
            f"{label}, mean {report.mean:.2f}, {report.verdict.value}",
            f"C{index}",
        )
    axes.set_xlabel("time (s)")
    axes.set_ylabel("NEES")
    if log_scale:
        axes.set_yscale("log")
        axes.set_ylabel("NEES (log scale)")
    axes.set_title(title, fontsize=10)
    axes.grid(alpha=0.25)
    axes.legend(loc="upper right", fontsize=8)
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


def consistency_figure(result: MonteCarloResult, reports: Sequence[ConsistencyReport]) -> Figure:
    """Plot the across-run average of each consistency statistic with its bounds."""
    panels = 1 + len(result.nis)
    figure, axes = plt.subplots(panels, 1, figsize=(8.0, 2.4 * panels), squeeze=False)
    column = axes[:, 0]

    lookup = {report.statistic: report for report in reports}
    nees_report = next((report for report in reports if report.statistic.startswith("NEES")), None)
    _plot_statistic(column[0], result.times, _nees_samples(result), nees_report, "NEES")

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
