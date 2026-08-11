"""Driving one estimator over an asynchronous multi-sensor measurement stream.

The runner owns three responsibilities that do not belong in a filter: deciding
what to do with a report that arrives out of order, propagating the belief over
the irregular gap between consecutive reports, and recording the statistics the
analysis layer needs.

Asynchronous fusion here is the sequential-update form: sensors are not assumed
to be synchronised, each report is applied on its own as soon as it is released,
and the prediction step covers whatever interval separates it from the previous
one. Because the sensors are conditionally independent given the state, applying
two reports with the same timestamp one after the other gives the same posterior
as stacking them into one taller measurement.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.algorithm.base import GaussianState, StateEstimator
from sensor_fusion.model.angles import wrap_to_pi
from sensor_fusion.model.motion import (
    ConstantAcceleration,
    ConstantTurnRate,
    ConstantVelocity,
    MotionModel,
)
from sensor_fusion.pipeline.simulator import Scenario
from sensor_fusion.pipeline.trace import (
    FilterTrace,
    Measurement,
    OutOfOrderPolicy,
    StepRecord,
)

__all__ = ["FusionSettings", "InitialUncertainty", "matched_belief", "run_filter"]

# Timestamps come off a uniform grid, so an exact equality test would be enough,
# but a tolerance keeps the comparison honest if a schedule is ever changed to
# something that does not land on the grid exactly.
_TIME_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class FusionSettings:
    """Policy choices for one run over a measurement stream."""

    policy: OutOfOrderPolicy = OutOfOrderPolicy.BUFFER
    latency_budget: float = 0.15
    sensors: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.latency_budget < 0.0:
            raise ValueError("latency_budget must be non-negative")


@dataclass(frozen=True, slots=True)
class InitialUncertainty:
    """Standard deviations used to seed a filter whose model is not the truth model."""

    position_std: float = 1.0
    velocity_std: float = 3.0
    speed_std: float = 3.0
    heading_std: float = 0.6
    yaw_rate_std: float = 0.6
    accel_std: float = 3.0

    def build(self, model: MotionModel, cartesian: FloatArray) -> GaussianState:
        """Return the initial belief for ``model`` seeded from a Cartesian state."""
        mean = model.from_cartesian(np.asarray(cartesian, dtype=np.float64))
        if isinstance(model, ConstantVelocity):
            deviations = [
                self.position_std,
                self.position_std,
                self.velocity_std,
                self.velocity_std,
            ]
        elif isinstance(model, ConstantAcceleration):
            deviations = [
                self.position_std,
                self.position_std,
                self.velocity_std,
                self.velocity_std,
                self.accel_std,
                self.accel_std,
            ]
        elif isinstance(model, ConstantTurnRate):
            deviations = [
                self.position_std,
                self.position_std,
                self.speed_std,
                self.heading_std,
                self.yaw_rate_std,
            ]
        else:  # pragma: no cover - defensive, no other model exists
            raise TypeError(f"no initial uncertainty defined for {model.name}")
        return GaussianState(mean=mean, cov=np.diag(np.asarray(deviations, dtype=np.float64) ** 2))


def matched_belief(scenario: Scenario) -> GaussianState:
    """Return the belief that makes a filter correctly specified for ``scenario``.

    A filter started here, using the scenario's own truth model, satisfies every
    assumption behind the chi-square consistency bounds: the initial error is
    drawn from the initial covariance and the process noise is the assumed one.
    """
    return GaussianState(
        mean=np.asarray(scenario.config.initial_mean, dtype=np.float64),
        cov=np.asarray(scenario.config.initial_cov, dtype=np.float64),
    )


def _release_order(
    measurements: tuple[Measurement, ...], settings: FusionSettings
) -> tuple[tuple[Measurement, ...], int]:
    """Return the processing order and the number of out-of-order arrivals.

    Under ``BUFFER`` a report is held until the newest arrival is at least
    ``latency_budget`` seconds past its timestamp, at which point every buffered
    report older than that deadline is released in timestamp order. Under
    ``DISCARD`` the arrival order is kept as it is.
    """
    inversions = 0
    newest_timestamp = -np.inf
    for measurement in measurements:
        if measurement.time < newest_timestamp - _TIME_TOLERANCE:
            inversions += 1
        newest_timestamp = max(newest_timestamp, measurement.time)

    if settings.policy is OutOfOrderPolicy.DISCARD:
        return measurements, inversions

    pending: list[tuple[float, int, Measurement]] = []
    released: list[Measurement] = []
    for sequence, measurement in enumerate(measurements):
        heapq.heappush(pending, (measurement.time, sequence, measurement))
        deadline = measurement.arrival_time - settings.latency_budget
        while pending and pending[0][0] <= deadline:
            released.append(heapq.heappop(pending)[2])
    while pending:
        released.append(heapq.heappop(pending)[2])
    return tuple(released), inversions


def _native_nees(model: MotionModel, mean: FloatArray, cov: FloatArray, truth: FloatArray) -> float:
    error = np.asarray(truth, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    for index in model.angle_indices:
        error[index] = float(wrap_to_pi(np.asarray([error[index]]))[0])
    return float(error @ np.linalg.solve(cov, error))


def _cartesian_nees(
    model: MotionModel, mean: FloatArray, cov: FloatArray, truth_cartesian: FloatArray
) -> tuple[FloatArray, float]:
    """Return the Cartesian estimate and its normalised estimation error squared.

    The second moment comes from ``model.cartesian_moment``, which is the exact
    covariance of ``truth_cartesian - estimate`` under the belief rather than a
    Jacobian projection of the state covariance, so the statistic has expectation
    four whatever the curvature of the Cartesian view.
    """
    estimate = model.to_cartesian(mean)
    projected_cov = model.cartesian_moment(mean, cov)
    error = np.asarray(truth_cartesian, dtype=np.float64) - estimate
    return estimate, float(error @ np.linalg.solve(projected_cov, error))


def run_filter(
    estimator: StateEstimator,
    scenario: Scenario,
    initial: GaussianState,
    settings: FusionSettings | None = None,
) -> FilterTrace:
    """Run ``estimator`` over ``scenario`` and return the recorded trace."""
    settings = settings or FusionSettings()
    model = estimator.motion
    truth_model = scenario.config.truth_model
    # Value equality, not just a matching name. Every motion model is a frozen
    # dataclass, so this compares the process noise parameters too. A constant
    # velocity filter with a different spectral density from the constant
    # velocity model that generated the truth is mis-specified, and reporting its
    # native NEES as the exact statistic would hide exactly the mis-specification
    # the statistic exists to detect.
    native_available = truth_model == model

    selected = scenario.measurements
    if settings.sensors is not None:
        allowed = set(settings.sensors)
        selected = tuple(item for item in selected if item.sensor_name in allowed)

    ordered, inversions = _release_order(selected, settings)

    state = initial
    clock = 0.0
    discarded = 0
    records: list[StepRecord] = []

    for measurement in ordered:
        gap = measurement.time - clock
        if gap < -_TIME_TOLERANCE:
            discarded += 1
            continue
        if gap > _TIME_TOLERANCE:
            state = estimator.predict(state, gap)
        result = estimator.update(state, measurement.value, measurement.sensor)
        state = result.state
        clock = measurement.time

        truth_cartesian = scenario.truth_cartesian[measurement.truth_index]
        estimate, nees_cartesian = _cartesian_nees(model, state.mean, state.cov, truth_cartesian)
        nees_state = (
            _native_nees(
                model,
                state.mean,
                state.cov,
                scenario.truth_states[measurement.truth_index],
            )
            if native_available
            else float("nan")
        )
        records.append(
            StepRecord(
                time=measurement.time,
                sensor_name=measurement.sensor_name,
                mean=state.mean,
                cov=state.cov,
                innovation=result.innovation,
                nis=result.nis,
                nis_dof=measurement.sensor.dim,
                truth_cartesian=np.asarray(truth_cartesian, dtype=np.float64),
                estimate_cartesian=estimate,
                nees_cartesian=nees_cartesian,
                nees_state=nees_state,
            )
        )

    return FilterTrace(
        filter_name=estimator.name,
        motion_name=model.name,
        policy=settings.policy,
        records=tuple(records),
        processed=len(records),
        discarded=discarded,
        out_of_order_arrivals=inversions,
    )
