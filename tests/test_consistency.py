"""Tier one: the consistency machinery and the verdicts it produces.

The key test here is that a filter built to be correctly specified, with the
truth drawn from the filter's own initial covariance and driven by the filter's
own process noise, is classified as consistent under a fixed seed. A filter that
fails this test is wrong somewhere in the covariance path even if its estimates
look reasonable.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from sensor_fusion.algorithm.base import GaussianState, StateEstimator
from sensor_fusion.algorithm.ekf import ExtendedKalmanFilter
from sensor_fusion.algorithm.ukf import UnscentedKalmanFilter
from sensor_fusion.analysis.consistency import Verdict, chi2_interval, consistency_report
from sensor_fusion.analysis.metrics import rmse, summarize
from sensor_fusion.analysis.report import FilterAssessment, assess
from sensor_fusion.model.motion import ConstantVelocity
from sensor_fusion.pipeline.fusion import (
    FusionSettings,
    InitialUncertainty,
    matched_belief,
    run_filter,
)
from sensor_fusion.pipeline.montecarlo import RunBuilder, run_monte_carlo
from sensor_fusion.pipeline.scenarios import straight_target, turning_target, with_latency
from sensor_fusion.pipeline.simulator import Scenario, simulate
from sensor_fusion.pipeline.trace import OutOfOrderPolicy

RUNS = 24
STEPS = 800
SEED = 909


def _matched_ukf(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return UnscentedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


def _matched_ekf(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return ExtendedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


class TestChiSquareInterval:
    """The interval must match the chi-square distribution it claims."""

    @pytest.mark.parametrize("dof", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("runs", [1, 10, 60])
    def test_interval_matches_the_quantiles(self, dof: int, runs: int) -> None:
        """Bounds are the chi-square quantiles of the sum, divided by the count."""
        lower, upper = chi2_interval(dof, runs, confidence=0.95)
        assert lower == pytest.approx(float(chi2.ppf(0.025, dof * runs)) / runs, rel=1e-12)
        assert upper == pytest.approx(float(chi2.ppf(0.975, dof * runs)) / runs, rel=1e-12)

    def test_interval_brackets_the_expected_value(self) -> None:
        """The mean of a chi-square variable is its degrees of freedom."""
        for dof in (1, 2, 3, 4, 5, 6):
            lower, upper = chi2_interval(dof, 40)
            assert lower < dof < upper

    def test_interval_narrows_with_more_runs(self) -> None:
        """Averaging more independent samples tightens the interval."""
        narrow = chi2_interval(4, 200)
        wide = chi2_interval(4, 5)
        assert narrow[1] - narrow[0] < wide[1] - wide[0]

    def test_invalid_arguments_are_rejected(self) -> None:
        """Degrees of freedom, run count, and confidence are validated."""
        with pytest.raises(ValueError, match="at least one"):
            chi2_interval(0, 10)
        with pytest.raises(ValueError, match="confidence"):
            chi2_interval(4, 10, confidence=1.5)


class TestVerdicts:
    """Classification of a statistic against its interval."""

    def test_samples_at_the_expected_value_are_consistent(self) -> None:
        """A statistic sitting on its degrees of freedom is consistent."""
        samples = np.full((30, 50), 4.0)
        report = consistency_report("synthetic", samples, dof=4)
        assert report.verdict is Verdict.CONSISTENT
        assert report.inside_fraction == pytest.approx(1.0)

    def test_inflated_samples_are_optimistic(self) -> None:
        """A statistic far above its interval means the covariance is too small."""
        report = consistency_report("synthetic", np.full((30, 50), 40.0), dof=4)
        assert report.verdict is Verdict.OPTIMISTIC
        assert report.above_fraction == pytest.approx(1.0)

    def test_deflated_samples_are_conservative(self) -> None:
        """A statistic far below its interval means the covariance is too large."""
        report = consistency_report("synthetic", np.full((30, 50), 0.2), dof=4)
        assert report.verdict is Verdict.CONSERVATIVE
        assert report.below_fraction == pytest.approx(1.0)

    def test_actual_chi_square_draws_are_consistent(self) -> None:
        """Sampling the reference distribution must pass the test it defines."""
        rng = np.random.default_rng(31)
        samples = rng.chisquare(df=4, size=(60, 200)).astype(np.float64)
        report = consistency_report("synthetic", samples, dof=4)
        assert report.verdict is Verdict.CONSISTENT
        assert report.inside_fraction > 0.85

    def test_shape_is_validated(self) -> None:
        """A one-dimensional input is a caller error, not a single run."""
        with pytest.raises(ValueError, match="runs, steps"):
            consistency_report("synthetic", np.zeros(10), dof=4)


class TestTransientAgainstPersistentError:
    """Separating a filter that converges badly from one that is simply wrong.

    Two filters can share a grand mean and mean opposite things by it: one whose
    covariance is wrong for the first fraction of a second while it converges
    from a loose prior, and one whose covariance is mildly wrong throughout. The
    fraction of steps above the upper bound tells them apart, and nothing is
    trimmed from the mean, because a filter that needs a second before its
    covariance can be trusted has a property worth reporting.
    """

    def test_a_short_transient_leaves_most_steps_inside(self) -> None:
        """A filter wrong only while converging is above the bound only then."""
        samples = np.full((30, 50), 4.0)
        samples[:, :6] = 40.0
        report = consistency_report("synthetic", samples, dof=4)
        assert report.above_fraction == pytest.approx(6 / 50)
        assert report.inside_fraction == pytest.approx(44 / 50)
        assert report.verdict is Verdict.OPTIMISTIC, "the transient still counts in the mean"

    def test_a_persistent_error_is_above_the_bound_throughout(self) -> None:
        """The same verdict, reached a completely different way."""
        report = consistency_report("synthetic", np.full((30, 50), 7.0), dof=4)
        assert report.above_fraction == pytest.approx(1.0)
        assert report.verdict is Verdict.OPTIMISTIC

    def test_the_two_can_share_a_grand_mean(self) -> None:
        """Guard the guard: the mean alone genuinely cannot separate these cases."""
        transient = np.full((30, 50), 4.0)
        transient[:, :6] = 29.0
        persistent = np.full((30, 50), 7.0)
        left = consistency_report("transient", transient, dof=4)
        right = consistency_report("persistent", persistent, dof=4)
        assert left.mean == pytest.approx(right.mean, rel=1e-9)
        assert left.above_fraction < 0.2 < right.above_fraction


class TestCorrectlySpecifiedFilters:
    """A correctly specified filter must land inside the chi-square bounds."""

    @pytest.mark.parametrize("build", [_matched_ekf, _matched_ukf], ids=["ekf", "ukf"])
    def test_linear_scenario_is_consistent(self, build: RunBuilder) -> None:
        """On the linear scenario every statistic is inside its interval."""
        result = run_monte_carlo(straight_target(steps=STEPS), build, runs=RUNS, seed=SEED)
        assessment = assess(result)
        assert assessment.nees.verdict is Verdict.CONSISTENT
        assert all(report.verdict is Verdict.CONSISTENT for report in assessment.nis)
        assert assessment.diverged_runs == 0

    @pytest.mark.parametrize("build", [_matched_ekf, _matched_ukf], ids=["ekf", "ukf"])
    def test_turning_scenario_is_consistent(self, build: RunBuilder) -> None:
        """On the nonlinear scenario every statistic is inside its interval."""
        result = run_monte_carlo(turning_target(steps=STEPS), build, runs=RUNS, seed=SEED)
        assessment = assess(result)
        assert assessment.nees.verdict is Verdict.CONSISTENT
        assert all(report.verdict is Verdict.CONSISTENT for report in assessment.nis)
        assert assessment.diverged_runs == 0

    def test_native_nees_is_reported_when_the_models_match(self) -> None:
        """The exact state-space statistic is preferred over the projection."""
        result = run_monte_carlo(
            turning_target(steps=400), _matched_ukf, runs=6, seed=SEED
        )
        assert result.native_nees_available
        assert assess(result).nees.statistic.startswith("NEES (state")


class TestMisspecifiedFilters:
    """Deliberate mis-specification must be detected in the right direction."""

    def _campaign(self, density: float, runs: int = 20) -> FilterAssessment:
        seeder = InitialUncertainty()

        def build(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
            model = ConstantVelocity(spectral_density=density)
            return UnscentedKalmanFilter(model), seeder.build(
                model, scenario.truth_cartesian[0]
            )

        return assess(
            run_monte_carlo(turning_target(steps=STEPS), build, runs=runs, seed=SEED)
        )

    def test_too_little_process_noise_is_optimistic(self) -> None:
        """A filter that ignores the turn claims an accuracy it does not have."""
        assessment = self._campaign(0.05)
        assert assessment.nees.verdict is Verdict.OPTIMISTIC
        assert all(report.verdict is Verdict.OPTIMISTIC for report in assessment.nis)

    def test_too_much_process_noise_is_conservative(self) -> None:
        """A filter drowning in assumed process noise wastes information."""
        assessment = self._campaign(4000.0)
        assert assessment.nees.verdict is Verdict.CONSERVATIVE
        assert all(report.verdict is Verdict.CONSERVATIVE for report in assessment.nis)

    def test_the_cartesian_projection_is_used_when_the_models_differ(self) -> None:
        """Mis-specified runs are labelled as using the projected statistic."""
        assessment = self._campaign(0.05, runs=6)
        assert "Cartesian projection" in assessment.nees.statistic


class TestMonteCarloHarness:
    """The harness must refuse to average runs that are not comparable."""

    def test_alignment_is_checked(self) -> None:
        """Traces with different measurement grids cannot be stacked."""
        config = with_latency(turning_target(steps=600), jitter=0.05)
        with pytest.raises(ValueError, match="deterministic measurement schedule"):
            run_monte_carlo(
                config,
                _matched_ukf,
                runs=6,
                seed=SEED,
                settings=FusionSettings(OutOfOrderPolicy.DISCARD),
            )

    def test_runs_must_be_positive(self) -> None:
        """A campaign of zero runs is a caller error."""
        with pytest.raises(ValueError, match="at least one"):
            run_monte_carlo(turning_target(steps=100), _matched_ukf, runs=0)


class TestMetrics:
    """Root mean square error helpers."""

    def test_rmse_of_a_known_stack(self) -> None:
        """RMSE is the root mean square of the error magnitude, not per axis."""
        errors = np.array([[3.0, 4.0], [0.0, 0.0]])
        assert rmse(errors) == pytest.approx(np.sqrt((25.0 + 0.0) / 2.0))

    def test_rmse_of_an_empty_stack_is_not_a_number(self) -> None:
        """An empty run has no error, and saying zero would be a lie."""
        assert np.isnan(rmse(np.zeros((0, 2))))

    def test_summary_labels_default_to_the_filter_and_model(self) -> None:
        """A trace summarises itself without needing a caller-supplied label."""
        scenario = simulate(straight_target(steps=200), seed=1)
        trace = run_filter(
            UnscentedKalmanFilter(scenario.config.truth_model),
            scenario,
            matched_belief(scenario),
        )
        assert summarize(trace).label == "ukf/constant-velocity"
