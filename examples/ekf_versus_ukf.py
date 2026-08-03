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

The same four-rung sensor ladder is run against two targets, because the answer
is a property of the pair rather than of the sensor. On the base scenario the
two filters are interchangeable on every rung. On a target three times further
out, moving away and turning, the extended filter's covariance is already wrong
with both sensors present.

    uv run python examples/ekf_versus_ukf.py
    uv run python examples/ekf_versus_ukf.py --quick
"""

from __future__ import annotations

import argparse
import time

from sensor_fusion.algorithm import (
    ExtendedKalmanFilter,
    GaussianState,
    StateEstimator,
    UnscentedKalmanFilter,
)
from sensor_fusion.analysis.report import assess
from sensor_fusion.pipeline.fusion import matched_belief
from sensor_fusion.pipeline.montecarlo import run_monte_carlo
from sensor_fusion.pipeline.scenarios import distant_target, sensor_regimes, turning_target
from sensor_fusion.pipeline.simulator import Scenario


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
    for target, config in (
        ("base scenario, target circling 64 m from the sensor", turning_target(steps)),
        ("distant scenario, target 100 m out and leaving", distant_target(steps)),
    ):
        print(f"\n=== {target} ===")
        for label, regime in sensor_regimes(config):
            print(f"\n{label}")
            for name, build in (("EKF", _build_ekf), ("UKF", _build_ukf)):
                start = time.perf_counter()
                result = run_monte_carlo(regime, build, runs=runs, seed=args.seed)
                elapsed = time.perf_counter() - start
                assessment = assess(result)
                print(
                    f"  {name}: position RMSE {assessment.position_rmse:8.4f} m, "
                    f"velocity RMSE {assessment.velocity_rmse:8.4f} m/s, "
                    f"NEES mean {assessment.nees.mean:9.3f} "
                    f"median {assessment.median_run_nees:6.3f} "
                    f"({assessment.nees.verdict.value}), "
                    f"{assessment.nees.above_fraction * 100:5.1f} percent of steps above, "
                    f"lost {assessment.diverged_runs}/{runs}, "
                    f"{elapsed:.2f} s"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
