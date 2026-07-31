"""Ground truth and measurement generation.

The simulator draws the initial truth from the same Gaussian the filter is
initialised with, and drives the truth with the same process noise the filter
assumes. A filter run on such a scenario is correctly specified by construction,
so its normalised innovation squared and normalised estimation error squared are
chi-square distributed and any departure from the chi-square bounds is a defect
in the implementation rather than in the model. Mis-specification is then
introduced deliberately, by giving the filter a different motion model from the
one that generated the truth, and the same statistics measure it.

Truth is generated on a fine base grid and measurements are placed on grid
points, so no interpolation is ever needed to compare an estimate against the
truth. Sensor periods are given in whole base steps, which is what makes that
possible.

Reproducibility
---------------
Every random draw goes through a closed-form matrix factor supplied by the model
layer. No eigendecomposition is used anywhere in the noise path, because the
sign of an eigenvector is not determined by the problem and two LAPACK builds
can legitimately return factors differing by a reflection, which would make the
same seed produce different data on a different machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.model.angles import wrap_to_pi
from sensor_fusion.model.measurement import MeasurementModel
from sensor_fusion.model.motion import MotionModel
from sensor_fusion.pipeline.trace import Measurement

__all__ = ["Scenario", "ScenarioConfig", "SensorSchedule", "simulate"]


@dataclass(frozen=True, slots=True)
class SensorSchedule:
    """When one sensor reports and how late its reports arrive.

    ``period_steps`` and ``offset_steps`` are counted in base grid steps, so two
    sensors with coprime periods interleave irregularly, which is what an
    asynchronous multi-sensor system actually looks like.
    """

    sensor: MeasurementModel
    period_steps: int
    offset_steps: int = 0
    latency_mean: float = 0.0
    latency_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.period_steps < 1:
            raise ValueError("period_steps must be at least one")
        if self.offset_steps < 0:
            raise ValueError("offset_steps must be non-negative")
        if self.latency_mean < 0.0 or self.latency_jitter < 0.0:
            raise ValueError("latency parameters must be non-negative")


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Everything needed to generate one scenario."""

    truth_model: MotionModel
    initial_mean: FloatArray
    initial_cov: FloatArray
    schedules: tuple[SensorSchedule, ...]
    base_dt: float = 0.005
    steps: int = 2000
    process_noise: bool = True

    def __post_init__(self) -> None:
        if self.base_dt <= 0.0:
            raise ValueError("base_dt must be positive")
        if self.steps < 1:
            raise ValueError("steps must be at least one")
        if not self.schedules:
            raise ValueError("at least one sensor schedule is required")

    @property
    def duration(self) -> float:
        """Total simulated time in seconds."""
        return self.base_dt * self.steps


@dataclass(frozen=True, slots=True)
class Scenario:
    """Ground truth on a fine grid plus the measurements sampled from it."""

    config: ScenarioConfig
    times: FloatArray
    truth_states: FloatArray
    truth_cartesian: FloatArray
    measurements: tuple[Measurement, ...] = field(default=())

    @property
    def sensors(self) -> tuple[MeasurementModel, ...]:
        """The sensor models used in this scenario."""
        return tuple(schedule.sensor for schedule in self.config.schedules)


def _draw(factor: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Return ``factor @ standard_normal``, the noise for one step."""
    return np.asarray(factor @ rng.standard_normal(factor.shape[1]), dtype=np.float64)


def _initial_factor(cov: FloatArray) -> FloatArray:
    """Return the Cholesky factor of the initial covariance.

    An all-zero covariance is accepted and returns a zero factor, which is how a
    scenario asks for a deterministic initial state. Any other matrix must be
    positive definite, and the error says so rather than surfacing as a bare
    LAPACK failure several frames deeper.
    """
    array = np.asarray(cov, dtype=np.float64)
    if not np.any(array):
        return np.zeros_like(array)
    try:
        return np.asarray(np.linalg.cholesky(array), dtype=np.float64)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "initial_cov must be positive definite, or exactly zero for a "
            "deterministic initial state"
        ) from error


def _generate_truth(config: ScenarioConfig, rng: np.random.Generator) -> FloatArray:
    model = config.truth_model
    mean = np.asarray(config.initial_mean, dtype=np.float64)
    cov = np.asarray(config.initial_cov, dtype=np.float64)
    states = np.empty((config.steps + 1, model.dim), dtype=np.float64)
    states[0] = model.normalize(mean + _draw(_initial_factor(cov), rng))
    for step in range(config.steps):
        propagated = model.predict(states[step], config.base_dt)
        if config.process_noise:
            propagated = propagated + _draw(
                model.process_noise_factor(states[step], config.base_dt), rng
            )
        states[step + 1] = model.normalize(propagated)
    return states


def _generate_measurements(
    config: ScenarioConfig,
    truth_cartesian: FloatArray,
    rng: np.random.Generator,
    latency_rng: np.random.Generator,
) -> tuple[Measurement, ...]:
    collected: list[Measurement] = []
    for schedule in config.schedules:
        sensor = schedule.sensor
        factor = sensor.noise_factor
        angle_index = np.asarray(sensor.angle_indices, dtype=np.intp)
        for index in range(schedule.offset_steps, config.steps + 1, schedule.period_steps):
            time = index * config.base_dt
            value = sensor.predict(truth_cartesian[index]) + _draw(factor, rng)
            if angle_index.size:
                value[angle_index] = wrap_to_pi(value[angle_index])
            latency = schedule.latency_mean
            if schedule.latency_jitter > 0.0:
                latency = max(
                    0.0,
                    latency + schedule.latency_jitter * float(latency_rng.standard_normal()),
                )
            collected.append(
                Measurement(
                    time=time,
                    arrival_time=time + latency,
                    sensor=sensor,
                    value=value,
                    truth_index=index,
                )
            )
    # Sort by arrival time. The secondary keys make the order total and therefore
    # reproducible when two reports arrive in the same floating point instant.
    collected.sort(key=lambda item: (item.arrival_time, item.sensor_name, item.time))
    return tuple(collected)


def simulate(config: ScenarioConfig, seed: int) -> Scenario:
    """Generate one scenario from ``config`` under the given seed.

    Three independent streams are spawned from the seed, one for the truth, one
    for the measurement noise, and one for the transport latency. Keeping them
    separate means that attaching latency to a scenario changes only when
    reports arrive, not what they say, so a comparison of out-of-order policies
    is not confounded by a different noise realisation.
    """
    truth_seed, measurement_seed, latency_seed = np.random.SeedSequence(seed).spawn(3)
    truth_states = _generate_truth(config, np.random.default_rng(truth_seed))
    truth_cartesian = np.stack(
        [config.truth_model.to_cartesian(state) for state in truth_states]
    )
    measurements = _generate_measurements(
        config,
        truth_cartesian,
        np.random.default_rng(measurement_seed),
        np.random.default_rng(latency_seed),
    )
    times = np.arange(config.steps + 1, dtype=np.float64) * config.base_dt
    return Scenario(
        config=config,
        times=times,
        truth_states=truth_states,
        truth_cartesian=truth_cartesian,
        measurements=measurements,
    )
