"""Compare the extended and unscented filters against single-sensor baselines.

Accuracy is reported as the mean over independent Monte Carlo runs, with the
standard deviation across runs, because the root mean square error of a single
run of a couple of hundred updates carries enough sampling noise to reverse the
ordering of two filters that are genuinely within a percent of each other.

The script also checks the two nonlinear filters against the exact linear Kalman
filter on a linear problem, where any disagreement beyond rounding is a defect.

    uv run python examples/compare_filters.py
    uv run python examples/compare_filters.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sensor_fusion.algorithm import (
    ExtendedKalmanFilter,
    GaussianState,
    KalmanFilter,
    StateEstimator,
    UnscentedKalmanFilter,
)
from sensor_fusion.analysis.figures import error_figure, trajectory_figure
from sensor_fusion.model.motion import ConstantAcceleration, ConstantVelocity, MotionModel
from sensor_fusion.pipeline.fusion import (
    FusionSettings,
    InitialUncertainty,
    matched_belief,
    run_filter,
)
from sensor_fusion.pipeline.montecarlo import run_monte_carlo
from sensor_fusion.pipeline.scenarios import straight_target, turning_target
from sensor_fusion.pipeline.simulator import Scenario, simulate

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "outputs"

BOTH_SENSORS = FusionSettings()
LIDAR_ONLY = FusionSettings(sensors=("lidar",))
RADAR_ONLY = FusionSettings(sensors=("radar",))
SEEDER = InitialUncertainty()


def _ekf(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return ExtendedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


def _ukf(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return UnscentedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


def _seeded(scenario: Scenario, model: MotionModel) -> tuple[StateEstimator, GaussianState]:
    """Build a UKF on ``model``, seeded from the truth at time zero."""
    if model is scenario.config.truth_model:
        return UnscentedKalmanFilter(model), matched_belief(scenario)
    return UnscentedKalmanFilter(model), SEEDER.build(model, scenario.truth_cartesian[0])


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=2000, help="base grid steps at 200 Hz")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    args = parser.parse_args(argv)

    runs = 4 if args.quick else args.runs
    steps = 200 if args.quick else args.steps
    config = turning_target(steps=steps)

    configurations = (
        ("EKF, lidar and radar", _ekf, BOTH_SENSORS),
        ("UKF, lidar and radar", _ukf, BOTH_SENSORS),
        ("EKF, lidar only", _ekf, LIDAR_ONLY),
        ("EKF, radar only", _ekf, RADAR_ONLY),
    )

    print(f"turning target, {runs} runs of {steps} base steps, base seed {args.seed}")
    print(f"{'configuration':24s} {'position RMSE (m)':>22s} {'velocity RMSE (m/s)':>24s}")
    for label, build, settings in configurations:
        result = run_monte_carlo(config, build, runs=runs, seed=args.seed, settings=settings)
        print(
            f"{label:24s} "
            f"{result.mean_position_rmse:12.4f} +/- {np.std(result.position_rmse):<7.4f} "
            f"{result.mean_velocity_rmse:14.4f} +/- {np.std(result.velocity_rmse):<7.4f}"
        )

    print("\nsame scenario, same UKF, three motion models")
    print(f"{'motion model':24s} {'position RMSE (m)':>22s} {'velocity RMSE (m/s)':>24s}")
    for label, model in (
        ("constant velocity", ConstantVelocity(spectral_density=20.0)),
        ("constant acceleration", ConstantAcceleration(spectral_density=200.0)),
        ("CTRV (matches truth)", config.truth_model),
    ):
        result = run_monte_carlo(
            config,
            lambda scenario, model=model: _seeded(scenario, model),
            runs=runs,
            seed=args.seed,
            settings=BOTH_SENSORS,
        )
        print(
            f"{label:24s} "
            f"{result.mean_position_rmse:12.4f} +/- {np.std(result.position_rmse):<7.4f} "
            f"{result.mean_velocity_rmse:14.4f} +/- {np.std(result.velocity_rmse):<7.4f}"
        )

    linear = simulate(straight_target(steps=steps), seed=args.seed)
    linear_model = linear.config.truth_model
    exact = run_filter(KalmanFilter(linear_model), linear, matched_belief(linear), LIDAR_ONLY)
    print("\nagreement with the exact Kalman filter on a linear problem, lidar only")
    for label, estimator in (
        ("EKF", ExtendedKalmanFilter(linear_model)),
        ("UKF", UnscentedKalmanFilter(linear_model)),
    ):
        trace = run_filter(estimator, linear, matched_belief(linear), LIDAR_ONLY)
        mean_gap = max(
            float(np.max(np.abs(left.mean - right.mean)))
            for left, right in zip(exact.records, trace.records, strict=True)
        )
        cov_gap = max(
            float(np.max(np.abs(left.cov - right.cov)))
            for left, right in zip(exact.records, trace.records, strict=True)
        )
        print(f"  {label}: largest mean difference {mean_gap:.3e}, "
              f"largest covariance difference {cov_gap:.3e}")

    scenario = simulate(config, seed=args.seed)
    traces = []
    labels = []
    for label, build, settings in configurations:
        estimator, initial = build(scenario)
        traces.append(run_filter(estimator, scenario, initial, settings))
        labels.append(label)
    args.outdir.mkdir(parents=True, exist_ok=True)
    track = args.outdir / "tracks.png"
    errors = args.outdir / "errors.png"
    trajectory_figure(scenario, traces, labels).savefig(track, dpi=140)
    error_figure(traces, labels).savefig(errors, dpi=140)
    print(f"\nwrote {track}")
    print(f"wrote {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
