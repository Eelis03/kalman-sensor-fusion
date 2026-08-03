"""Regenerate the three figures tracked under ``docs/figures``.

Every figure here is drawn from the same scenario, seed, and run count as the
table it sits beside in the README, so a reader who reruns the corresponding
example script gets the numbers the picture is made of.

The figures are committed as snapshots rather than rebuilt in continuous
integration. Matplotlib output is not byte reproducible across platforms or
across its own releases, so a byte comparison would fail for reasons that have
nothing to do with this package. What continuous integration does check is that
the code paths that draw them still run.

    uv run python examples/make_figures.py
    uv run python examples/make_figures.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensor_fusion.algorithm import (
    ExtendedKalmanFilter,
    GaussianState,
    StateEstimator,
    UnscentedKalmanFilter,
)
from sensor_fusion.analysis.figures import (
    nees_comparison_figure,
    nees_panels_figure,
    trajectory_figure,
)
from sensor_fusion.analysis.report import assess
from sensor_fusion.model.motion import ConstantVelocity
from sensor_fusion.pipeline.fusion import (
    FusionSettings,
    InitialUncertainty,
    matched_belief,
    run_filter,
)
from sensor_fusion.pipeline.montecarlo import run_monte_carlo
from sensor_fusion.pipeline.scenarios import distant_target, sensor_regimes, turning_target
from sensor_fusion.pipeline.simulator import Scenario, simulate

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

# Chosen so that three figures of line plots stay well inside the 250 KB the
# repository budgets for tracked figures, while still being legible at the width
# a README renders at. Raising it buys nothing a reader can see.
DPI = 110

# Seeds and run counts match the example scripts whose tables the figures
# illustrate: consistency_study.py for the specification panels, ekf_versus_ukf.py
# for the radar-only comparison, compare_filters.py for the track geometry.
CONSISTENCY_SEED = 2026
SWEEP_SEED = 7
TRACK_SEED = 11

SEEDER = InitialUncertainty()


def _matched(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return UnscentedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


def _mismatched(scenario: Scenario, density: float) -> tuple[StateEstimator, GaussianState]:
    model = ConstantVelocity(spectral_density=density)
    return UnscentedKalmanFilter(model), SEEDER.build(model, scenario.truth_cartesian[0])


def specification_figure(runs: int, steps: int, outdir: Path) -> Path:
    """Draw the NEES trace under a correct, a starved, and a flooded filter."""
    config = turning_target(steps=steps)
    campaigns = (
        ("CTRV, correctly specified", _matched),
        (
            "constant velocity, spectral density 2, under-noised",
            lambda scenario: _mismatched(scenario, 2.0),
        ),
        (
            "constant velocity, spectral density 4000, over-noised",
            lambda scenario: _mismatched(scenario, 4000.0),
        ),
    )
    results = []
    reports = []
    labels = []
    for label, build in campaigns:
        result = run_monte_carlo(config, build, runs=runs, seed=CONSISTENCY_SEED)
        results.append(result)
        reports.append(assess(result).nees)
        labels.append(label)

    target = outdir / "nees-specification.png"
    nees_panels_figure(results, reports, labels).savefig(target, dpi=DPI)
    return target


def filter_comparison_figure(runs: int, steps: int, outdir: Path) -> Path:
    """Draw the extended and unscented filters against one shared band.

    The regime is the first row of the distant target in
    ``examples/ekf_versus_ukf.py``: both sensors present, no track lost by either
    filter, identical data. It is chosen over a radar-only rung because there is
    nothing to argue about in it. Neither filter is struggling, the accuracy gap
    is 13 percent, and the covariance gap is still the whole width of the
    interval.
    """
    _, config = sensor_regimes(distant_target(steps))[0]
    results = []
    reports = []
    labels = []
    for label, estimator in (
        ("extended Kalman filter", ExtendedKalmanFilter),
        ("unscented Kalman filter", UnscentedKalmanFilter),
    ):

        def build(
            scenario: Scenario, estimator: type[StateEstimator] = estimator
        ) -> tuple[StateEstimator, GaussianState]:
            return estimator(scenario.config.truth_model), matched_belief(scenario)

        result = run_monte_carlo(config, build, runs=runs, seed=SWEEP_SEED)
        results.append(result)
        reports.append(assess(result).nees)
        labels.append(label)

    target = outdir / "nees-ekf-ukf.png"
    nees_comparison_figure(
        results,
        reports,
        labels,
        title="Distant target, both sensors: two ways of propagating one covariance",
        log_scale=True,
    ).savefig(target, dpi=DPI)
    return target


def geometry_figure(steps: int, outdir: Path) -> Path:
    """Draw the tracks against the truth, with the sensor at the origin."""
    scenario = simulate(turning_target(steps=steps), seed=TRACK_SEED)
    model = scenario.config.truth_model
    configurations = (
        ("UKF, lidar and radar", FusionSettings()),
        ("UKF, lidar only", FusionSettings(sensors=("lidar",))),
        ("UKF, radar only", FusionSettings(sensors=("radar",))),
    )
    traces = [
        run_filter(
            UnscentedKalmanFilter(model), scenario, matched_belief(scenario), settings
        )
        for _, settings in configurations
    ]
    target = outdir / "tracks.png"
    trajectory_figure(
        scenario, traces, [label for label, _ in configurations], figsize=(6.4, 5.4)
    ).savefig(target, dpi=DPI)
    return target


def main(argv: list[str] | None = None) -> int:
    """Regenerate every tracked figure and report where each was written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60, help="runs for the specification panels")
    parser.add_argument("--sweep-runs", type=int, default=40, help="runs for the radar comparison")
    parser.add_argument("--steps", type=int, default=2000, help="base grid steps at 200 Hz")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    args = parser.parse_args(argv)

    runs = 4 if args.quick else args.runs
    sweep_runs = 4 if args.quick else args.sweep_runs
    steps = 200 if args.quick else args.steps

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = (
        specification_figure(runs, steps, args.outdir),
        filter_comparison_figure(sweep_runs, steps, args.outdir),
        geometry_figure(steps, args.outdir),
    )
    total = 0
    for path in written:
        size = path.stat().st_size
        total += size
        print(f"wrote {path} ({size / 1024:.1f} KB)")
    print(f"total {total / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
