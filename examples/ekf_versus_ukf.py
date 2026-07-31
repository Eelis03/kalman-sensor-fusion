"""When the unscented filter is worth its extra cost.

The two filters differ only in how they push a Gaussian through a nonlinearity.
Where the nonlinearity is mild over the width of the current covariance, the
Jacobian linearisation of the extended filter is a good approximation and the
two agree. Where it is not, the extended filter propagates a covariance that is
wrong, and being wrong about the covariance is worse than being inaccurate.

This script sweeps the strength of the nonlinearity by removing the lidar, so
the only information is the polar radar measurement, and then degrading the
radar. It reports accuracy, the consistency verdict, and wall clock time for
both filters, so the trade can be read off measured numbers rather than folklore.

    uv run python examples/ekf_versus_ukf.py
    uv run python examples/ekf_versus_ukf.py --quick
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sensor_fusion.algorithm import (
    ExtendedKalmanFilter,
    GaussianState,
    StateEstimator,
    UnscentedKalmanFilter,
)
from sensor_fusion.analysis.report import assess
from sensor_fusion.model.measurement import Lidar, Radar
from sensor_fusion.model.motion import ConstantTurnRate
from sensor_fusion.pipeline.fusion import matched_belief
from sensor_fusion.pipeline.montecarlo import run_monte_carlo
from sensor_fusion.pipeline.scenarios import BASE_DT
from sensor_fusion.pipeline.simulator import Scenario, ScenarioConfig, SensorSchedule

# A target far enough from the sensor for a bearing error to translate into a
# large cross-range position error, which is where a Jacobian linearisation of
# the polar measurement starts to cost something.
INITIAL_MEAN = np.array([60.0, -20.0, 14.0, 1.4, 0.35])
INITIAL_COV = np.diag(np.array([4.0, 4.0, 4.0, 0.25, 0.09]))


def _config(schedules: tuple[SensorSchedule, ...], steps: int) -> ScenarioConfig:
    return ScenarioConfig(
        truth_model=ConstantTurnRate(spectral_density_accel=0.5, spectral_density_yaw=0.05),
        initial_mean=INITIAL_MEAN,
        initial_cov=INITIAL_COV,
        schedules=schedules,
        base_dt=BASE_DT,
        steps=steps,
    )


def _regimes(steps: int) -> tuple[tuple[str, ScenarioConfig], ...]:
    return (
        (
            "lidar and radar, 10 Hz and 13.3 Hz",
            _config(
                (
                    SensorSchedule(sensor=Lidar(), period_steps=20),
                    SensorSchedule(sensor=Radar(), period_steps=15, offset_steps=7),
                ),
                steps,
            ),
        ),
        (
            "radar only, 13.3 Hz, bearing sigma 0.03 rad",
            _config((SensorSchedule(sensor=Radar(), period_steps=15),), steps),
        ),
        (
            "radar only, 5 Hz, bearing sigma 0.10 rad",
            _config(
                (
                    SensorSchedule(
                        sensor=Radar(sigma_range=1.0, sigma_bearing=0.10, sigma_range_rate=1.0),
                        period_steps=40,
                    ),
                ),
                steps,
            ),
        ),
        (
            "radar only, 2.5 Hz, bearing sigma 0.20 rad",
            _config(
                (
                    SensorSchedule(
                        sensor=Radar(sigma_range=2.0, sigma_bearing=0.20, sigma_range_rate=2.0),
                        period_steps=80,
                    ),
                ),
                steps,
            ),
        ),
    )


def _build_ekf(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return ExtendedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


def _build_ukf(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return UnscentedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


def main(argv: list[str] | None = None) -> int:
    """Run the regime sweep and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    args = parser.parse_args(argv)

    runs = 4 if args.quick else args.runs
    steps = 200 if args.quick else args.steps

    print(f"{runs} runs of {steps} base steps each, base seed {args.seed}")
    for label, config in _regimes(steps):
        print(f"\n{label}")
        for name, build in (("EKF", _build_ekf), ("UKF", _build_ukf)):
            start = time.perf_counter()
            result = run_monte_carlo(config, build, runs=runs, seed=args.seed)
            elapsed = time.perf_counter() - start
            assessment = assess(result)
            print(
                f"  {name}: position RMSE {assessment.position_rmse:8.4f} m, "
                f"velocity RMSE {assessment.velocity_rmse:8.4f} m/s, "
                f"NEES mean {assessment.nees.mean:9.3f} "
                f"median {assessment.median_run_nees:6.3f} "
                f"({assessment.nees.verdict.value}), "
                f"lost {assessment.diverged_runs}/{runs}, "
                f"{elapsed:.2f} s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
