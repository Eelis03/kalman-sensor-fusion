"""Named scenarios shared by the examples, the tests, and the documentation.

Keeping the configurations here rather than in each example script means the
numbers quoted in the README, the numbers a test pins, and the numbers a reader
reproduces all come from one definition.

Sensor rates are deliberately not harmonically related. Lidar reports every 20
base steps and radar every 15, offset by 7, so on a 200 Hz base grid the two
sensors run at 10 Hz and 13.33 Hz and never report in the same instant. The gap
the filter must predict over therefore changes at almost every step, which is
what an asynchronous system looks like and what a fixed-step implementation
would quietly get wrong.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from sensor_fusion.model.measurement import Lidar, Radar
from sensor_fusion.model.motion import ConstantTurnRate, ConstantVelocity
from sensor_fusion.pipeline.simulator import ScenarioConfig, SensorSchedule

__all__ = [
    "BASE_DT",
    "LIDAR_PERIOD_STEPS",
    "MINIMUM_SAFE_RANGE",
    "RADAR_OFFSET_STEPS",
    "RADAR_PERIOD_STEPS",
    "boundary_crossing_target",
    "straight_target",
    "turning_target",
    "with_latency",
]

BASE_DT: Final[float] = 0.005
LIDAR_PERIOD_STEPS: Final[int] = 20
RADAR_PERIOD_STEPS: Final[int] = 15
RADAR_OFFSET_STEPS: Final[int] = 7

# Every scenario here is placed so that the target stays at least this far from
# the sensor across a wide range of seeds. Range, bearing, and range rate all
# degenerate as the target approaches the origin: the bearing is undefined there
# and its Jacobian is unbounded, so a scenario whose target passes close to the
# sensor measures the behaviour of a singularity as much as the behaviour of a
# filter. A test asserts this bound over 200 seeds per scenario, so that a future
# change to a starting state cannot quietly reintroduce the problem.
MINIMUM_SAFE_RANGE: Final[float] = 15.0


def _schedules() -> tuple[SensorSchedule, ...]:
    return (
        SensorSchedule(sensor=Lidar(), period_steps=LIDAR_PERIOD_STEPS, offset_steps=0),
        SensorSchedule(
            sensor=Radar(),
            period_steps=RADAR_PERIOD_STEPS,
            offset_steps=RADAR_OFFSET_STEPS,
        ),
    )


def straight_target(steps: int = 2000) -> ScenarioConfig:
    """A constant velocity target, the case where the exact answer is known.

    The truth model is linear and the initial truth is drawn from the initial
    covariance, so a linear Kalman filter using lidar alone is exactly optimal
    here and the extended and unscented filters must reproduce it.

    The target starts 28 m from the sensor and moves away from it. See
    :data:`MINIMUM_SAFE_RANGE` for why every scenario is placed this way.
    """
    return ScenarioConfig(
        truth_model=ConstantVelocity(spectral_density=2.0),
        initial_mean=np.array([25.0, 12.0, 6.0, 3.0]),
        initial_cov=np.diag(np.array([1.0, 1.0, 4.0, 4.0])),
        schedules=_schedules(),
        base_dt=BASE_DT,
        steps=steps,
    )


def turning_target(steps: int = 2000) -> ScenarioConfig:
    """A target with a random-walking speed and yaw rate.

    This is the scenario the extended and unscented filters are compared on. The
    truth turns, so a constant velocity filter is mis-specified against it and a
    CTRV filter is correctly specified, which lets the same consistency
    machinery show both outcomes.

    The nominal path is a circle of radius 12.5 m centred 64 m from the sensor,
    so the target completes rather more than one full revolution in 10 s while
    staying far outside the range at which the polar measurement degenerates.
    """
    return ScenarioConfig(
        truth_model=ConstantTurnRate(spectral_density_accel=0.5, spectral_density_yaw=0.004),
        initial_mean=np.array([50.0, 27.5, 10.0, 0.0, 0.8]),
        initial_cov=np.diag(np.array([1.0, 1.0, 1.0, 0.09, 0.02])),
        schedules=_schedules(),
        base_dt=BASE_DT,
        steps=steps,
    )


def boundary_crossing_target(steps: int = 2000, process_noise: bool = True) -> ScenarioConfig:
    """A target whose bearing and heading both cross the plus or minus pi cut.

    The circular path is centred on the negative x axis at a distance of about
    52 m with a radius of about 22 m, so the bearing from the sensor sweeps
    through ``pi`` twice per revolution, and the yaw rate of 0.55 rad/s carries
    the CTRV heading through the same cut. A filter that subtracts angles
    without wrapping sees an innovation close to ``2 * pi`` at each crossing and
    is thrown off the track; one that wraps correctly does not notice.

    Closest approach to the sensor is about 30 m, so the radar geometry never
    becomes degenerate and any divergence seen here is the wrap, not the range.
    """
    return ScenarioConfig(
        truth_model=ConstantTurnRate(spectral_density_accel=0.1, spectral_density_yaw=0.002),
        initial_mean=np.array([-30.0, 0.0, 12.0, 0.5 * math.pi, 0.55]),
        initial_cov=np.diag(np.array([0.25, 0.25, 0.25, 0.01, 0.0025])),
        schedules=_schedules(),
        base_dt=BASE_DT,
        steps=steps,
        process_noise=process_noise,
    )


def with_latency(
    config: ScenarioConfig,
    *,
    lidar_latency: float = 0.02,
    radar_latency: float = 0.09,
    jitter: float = 0.01,
) -> ScenarioConfig:
    """Return ``config`` with transport latency attached to each sensor.

    The radar is given the longer mean latency, which is the realistic case: a
    radar return goes through more processing before it reaches the fusion node
    than a lidar point cloud centroid does. With the defaults, a radar report
    stamped at ``t`` typically arrives after a lidar report stamped at
    ``t + 0.05``, so the stream reaching the filter is genuinely out of order.
    """
    latencies = {"lidar": lidar_latency, "radar": radar_latency}
    schedules = tuple(
        SensorSchedule(
            sensor=schedule.sensor,
            period_steps=schedule.period_steps,
            offset_steps=schedule.offset_steps,
            latency_mean=latencies.get(schedule.sensor.name, 0.0),
            latency_jitter=jitter,
        )
        for schedule in config.schedules
    )
    return ScenarioConfig(
        truth_model=config.truth_model,
        initial_mean=config.initial_mean,
        initial_cov=config.initial_cov,
        schedules=schedules,
        base_dt=config.base_dt,
        steps=config.steps,
        process_noise=config.process_noise,
    )
