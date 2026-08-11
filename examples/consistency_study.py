"""Monte Carlo consistency study.

Runs the same scenario many times under independent noise and classifies each
filter as consistent, optimistic, or conservative from its normalised estimation
error squared and its normalised innovation squared, with the interval and the
per-step coverage that support the verdict.

Four configurations are run: a correctly specified CTRV filter, which should come
out consistent, a constant velocity filter given a little too little process
noise for a turning target, which should come out optimistic while still holding
the track, the same filter starved outright, which should lose the track
entirely, and the same filter given far too much, which should come out
conservative.

The mildly starved case is the one worth looking at. Its position error is not
alarming and its track looks reasonable, but its covariance is wrong by roughly a
factor of two in NEES. That is the failure a root mean square error comparison
cannot see and a consistency test can.

Each campaign also reports the whiteness of its innovations, which asks a
question no magnitude statistic answers: whether one innovation predicts the
next. The flooded filter is the case to read there, because every magnitude
statistic it produces says conservative and only the sign of its correlation says
that it is overcorrecting.

    uv run python examples/consistency_study.py
    uv run python examples/consistency_study.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensor_fusion.algorithm import GaussianState, StateEstimator, UnscentedKalmanFilter
from sensor_fusion.analysis.consistency import ConsistencyReport
from sensor_fusion.analysis.figures import consistency_figure
from sensor_fusion.analysis.report import assess
from sensor_fusion.analysis.whiteness import whiteness_report
from sensor_fusion.model.motion import ConstantVelocity
from sensor_fusion.pipeline.fusion import InitialUncertainty, matched_belief
from sensor_fusion.pipeline.montecarlo import MonteCarloResult, run_monte_carlo
from sensor_fusion.pipeline.scenarios import turning_target
from sensor_fusion.pipeline.simulator import Scenario

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "outputs"

# Process noise spectral densities for the deliberately mis-specified constant
# velocity filters, in m**2 / s**3.
SLIGHTLY_TOO_LITTLE = 2.0
FAR_TOO_LITTLE = 0.05
FAR_TOO_MUCH = 4000.0

SEEDER = InitialUncertainty()


def matched(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    """Build a filter that uses the model the truth was generated from."""
    return UnscentedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


def mismatched(scenario: Scenario, density: float) -> tuple[StateEstimator, GaussianState]:
    """Build a constant velocity filter for a target that is in fact turning."""
    model = ConstantVelocity(spectral_density=density)
    return UnscentedKalmanFilter(model), SEEDER.build(model, scenario.truth_cartesian[0])


def main(argv: list[str] | None = None) -> int:
    """Run the study and write the consistency figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    args = parser.parse_args(argv)

    runs = 5 if args.quick else args.runs
    steps = 200 if args.quick else args.steps
    config = turning_target(steps=steps)

    campaigns = (
        ("correctly specified CTRV", matched),
        (
            f"constant velocity, spectral density {SLIGHTLY_TOO_LITTLE}",
            lambda scenario: mismatched(scenario, SLIGHTLY_TOO_LITTLE),
        ),
        (
            f"constant velocity, spectral density {FAR_TOO_LITTLE}",
            lambda scenario: mismatched(scenario, FAR_TOO_LITTLE),
        ),
        (
            f"constant velocity, spectral density {FAR_TOO_MUCH}",
            lambda scenario: mismatched(scenario, FAR_TOO_MUCH),
        ),
    )

    print(f"turning target, {runs} runs of {steps} base steps, base seed {args.seed}")
    figures: list[tuple[MonteCarloResult, tuple[ConsistencyReport, ...]]] = []
    for title, build in campaigns:
        result = run_monte_carlo(config, build, runs=runs, seed=args.seed)
        assessment = assess(result)
        print(f"\n{title}")
        for line in assessment.lines():
            print("  " + line)
        for name in sorted(result.normalized_innovations):
            print("  " + whiteness_report(name, result.normalized_innovations[name]).summary())
        figures.append((result, (assessment.nees, *assessment.nis)))

    args.outdir.mkdir(parents=True, exist_ok=True)
    result, reports = figures[0]
    target = args.outdir / "consistency.png"
    consistency_figure(result, reports).savefig(target, dpi=140)
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
