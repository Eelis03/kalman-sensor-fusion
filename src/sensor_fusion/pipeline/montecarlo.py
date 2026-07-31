"""Monte Carlo harness.

A single run tells you almost nothing about consistency: one draw of a
chi-square variable with four degrees of freedom lands outside its own
95 percent interval one time in twenty by construction. The harness repeats the
same scenario under independent noise realisations and stacks the statistics so
that the across-run average at each time step has a distribution tight enough to
test against.

This module produces raw stacked statistics only. Turning them into a verdict is
the analysis layer's job, which keeps the dependency pointing one way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sensor_fusion._math import rmse
from sensor_fusion._types import FloatArray
from sensor_fusion.algorithm.base import GaussianState, StateEstimator
from sensor_fusion.pipeline.fusion import FusionSettings, run_filter
from sensor_fusion.pipeline.simulator import Scenario, ScenarioConfig, simulate
from sensor_fusion.pipeline.trace import FilterTrace

__all__ = ["MonteCarloResult", "RunBuilder", "run_monte_carlo"]

RunBuilder = Callable[[Scenario], tuple[StateEstimator, GaussianState]]


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Statistics stacked across independent runs of one scenario."""

    filter_name: str
    motion_name: str
    runs: int
    state_dim: int
    times: FloatArray
    nees_cartesian: FloatArray
    nees_state: FloatArray
    nis: dict[str, FloatArray]
    nis_times: dict[str, FloatArray]
    nis_dof: dict[str, int]
    position_rmse: FloatArray
    velocity_rmse: FloatArray

    @property
    def native_nees_available(self) -> bool:
        """True when the filter's motion model generated the truth."""
        return bool(np.all(np.isfinite(self.nees_state)))

    @property
    def mean_position_rmse(self) -> float:
        """Position RMSE averaged over runs."""
        return float(np.mean(self.position_rmse))

    @property
    def mean_velocity_rmse(self) -> float:
        """Velocity RMSE averaged over runs."""
        return float(np.mean(self.velocity_rmse))


def _check_alignment(reference: FloatArray, candidate: FloatArray, label: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference, candidate):
        raise ValueError(
            f"runs produced different {label} grids; Monte Carlo averaging requires a "
            "deterministic measurement schedule, so use zero latency jitter or a "
            "latency budget that covers every arrival"
        )


def run_monte_carlo(
    config: ScenarioConfig,
    build: RunBuilder,
    *,
    runs: int = 50,
    seed: int = 0,
    settings: FusionSettings | None = None,
) -> MonteCarloResult:
    """Run ``runs`` independent realisations of ``config`` and stack the statistics."""
    if runs < 1:
        raise ValueError("runs must be at least one")

    traces: list[FilterTrace] = []
    for index in range(runs):
        scenario = simulate(config, seed=seed + index)
        estimator, initial = build(scenario)
        traces.append(run_filter(estimator, scenario, initial, settings))

    first = traces[0]
    times = first.times
    sensor_names = first.sensor_names
    for trace in traces[1:]:
        _check_alignment(times, trace.times, "measurement time")

    nis: dict[str, FloatArray] = {}
    nis_times: dict[str, FloatArray] = {}
    nis_dof: dict[str, int] = {}
    for name in sensor_names:
        reference_times = first.nis_times(name)
        for trace in traces[1:]:
            _check_alignment(reference_times, trace.nis_times(name), f"{name} measurement time")
        nis[name] = np.stack([trace.nis(name) for trace in traces])
        nis_times[name] = reference_times
        nis_dof[name] = next(
            record.nis_dof for record in first.records if record.sensor_name == name
        )

    return MonteCarloResult(
        filter_name=first.filter_name,
        motion_name=first.motion_name,
        runs=runs,
        state_dim=int(first.records[0].mean.size),
        times=times,
        nees_cartesian=np.stack([trace.nees_cartesian for trace in traces]),
        nees_state=np.stack([trace.nees_state for trace in traces]),
        nis=nis,
        nis_times=nis_times,
        nis_dof=nis_dof,
        position_rmse=np.array([rmse(trace.position_error) for trace in traces]),
        velocity_rmse=np.array([rmse(trace.velocity_error) for trace in traces]),
    )
