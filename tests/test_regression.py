"""Tier two: a recorded reference run pinned with numeric tolerance.

What is pinned here, and why
----------------------------
Every value below is either an integer count, a classification, a closed-form
quantity, or an aggregate metric of a numerically stable recursion. Nothing here
is the internal state of an iterative solve that did not converge, and nothing
here is a derived scalar of such a state.

The aggregate metrics are pinned at a relative tolerance of ``1e-6``. The chain
that produces them is: a seed sequence, which NumPy defines to give identical
bit streams on every platform; a Cholesky or closed-form matrix factor, which
has no sign ambiguity, unlike an eigendecomposition, whose eigenvector signs are
not determined by the problem and can differ between LAPACK builds; and then a
fixed number of Kalman recursion steps. The recursion is contracting under the
observability these scenarios have, so a difference of the order of the machine
epsilon in a BLAS reduction order does not grow. The observed cross-run spread
is of the order of ``1e-14`` relative, which leaves eight orders of magnitude of
margin against the tolerance.

Two things are deliberately not pinned by value. The statistics of the
deliberately mis-specified filters are asserted only by their classification and
by a wide qualitative bound, because a filter running far outside its
assumptions has large gains and is not the well-conditioned system the argument
above relies on. The relative accuracy of the two out-of-order policies on a
single short run is likewise not asserted, because at this sample size the
ordering is within noise; the claim that reordering helps belongs to the
forty-run comparison in ``examples/asynchronous_fusion.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_fusion.algorithm.base import GaussianState, StateEstimator
from sensor_fusion.algorithm.ekf import ExtendedKalmanFilter
from sensor_fusion.algorithm.sigma import ScaledUnscentedSpec
from sensor_fusion.algorithm.ukf import UnscentedKalmanFilter
from sensor_fusion.analysis.consistency import Verdict, chi2_interval
from sensor_fusion.analysis.report import FilterAssessment, assess
from sensor_fusion.model.motion import ConstantVelocity
from sensor_fusion.pipeline.fusion import (
    FusionSettings,
    InitialUncertainty,
    matched_belief,
    run_filter,
)
from sensor_fusion.pipeline.montecarlo import MonteCarloResult, run_monte_carlo
from sensor_fusion.pipeline.scenarios import turning_target, with_latency
from sensor_fusion.pipeline.simulator import Scenario, simulate
from sensor_fusion.pipeline.trace import OutOfOrderPolicy

STEPS = 400
RUNS = 12
SEED = 4242
RTOL = 1e-6

BOTH = FusionSettings()
LIDAR = FusionSettings(sensors=("lidar",))
RADAR = FusionSettings(sensors=("radar",))

# Recorded aggregate metrics: (position RMSE, velocity RMSE, NEES grand mean)
# and the NEES verdict that went with them.
REFERENCE: dict[tuple[str, str], tuple[float, float, float, Verdict]] = {
    ("ekf", "both"): (
        0.14086537215290348,
        0.770231735427901,
        5.201665429780526,
        Verdict.CONSISTENT,
    ),
    ("ekf", "lidar"): (
        0.16045615795385781,
        0.9838885966634342,
        5.526967401330779,
        Verdict.CONSISTENT,
    ),
    # Radar alone is where the extended filter's linearisation starts to cost
    # something. It is recorded as optimistic here, at 7.211 against an upper
    # bound of 6.941, and the unscented filter on the same data is not.
    ("ekf", "radar"): (
        0.6747160551537382,
        1.1779102489084432,
        7.210937536058946,
        Verdict.OPTIMISTIC,
    ),
    ("ukf", "both"): (
        0.14303011354295112,
        0.7662808407122598,
        5.004358236828753,
        Verdict.CONSISTENT,
    ),
    ("ukf", "lidar"): (
        0.1602348223192503,
        0.9946345944154458,
        5.422900771313384,
        Verdict.CONSISTENT,
    ),
    ("ukf", "radar"): (
        0.6804722075855371,
        1.1988559118723072,
        5.461289509879485,
        Verdict.CONSISTENT,
    ),
}

# Recorded normalised innovation squared grand means, per filter and sensor.
REFERENCE_NIS: dict[tuple[str, str], float] = {
    ("ekf", "lidar"): 2.0398308324227643,
    ("ekf", "radar"): 2.948644555797801,
    ("ukf", "lidar"): 2.007157673863794,
    ("ukf", "radar"): 2.9327254497459307,
}

SETTINGS = {"both": BOTH, "lidar": LIDAR, "radar": RADAR}


def _ekf(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return ExtendedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


def _ukf(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
    return UnscentedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


BUILDERS = {"ekf": _ekf, "ukf": _ukf}


def _campaign(filter_name: str, sensors: str) -> MonteCarloResult:
    return run_monte_carlo(
        turning_target(steps=STEPS),
        BUILDERS[filter_name],
        runs=RUNS,
        seed=SEED,
        settings=SETTINGS[sensors],
    )


class TestRecordedCampaigns:
    """Aggregate metrics of six recorded Monte Carlo campaigns."""

    @pytest.mark.parametrize(("filter_name", "sensors"), sorted(REFERENCE))
    def test_aggregate_metrics_match_the_record(self, filter_name: str, sensors: str) -> None:
        """Position RMSE, velocity RMSE, and mean NEES reproduce the baseline."""
        expected_position, expected_velocity, expected_nees, _ = REFERENCE[filter_name, sensors]
        result = _campaign(filter_name, sensors)
        assessment = assess(result)
        assert result.mean_position_rmse == pytest.approx(expected_position, rel=RTOL)
        assert result.mean_velocity_rmse == pytest.approx(expected_velocity, rel=RTOL)
        assert assessment.nees.mean == pytest.approx(expected_nees, rel=RTOL)

    @pytest.mark.parametrize(("filter_name", "sensors"), sorted(REFERENCE))
    def test_recorded_verdicts(self, filter_name: str, sensors: str) -> None:
        """Each campaign reproduces its recorded NEES classification.

        Five of the six are consistent. The extended filter on radar alone is
        recorded as optimistic, which is the finding rather than a defect: with
        the linear sensor removed, the linearisation of the polar measurement is
        the only thing standing between the filter and the truth, and it is not
        good enough to keep the covariance honest.
        """
        assessment = assess(_campaign(filter_name, sensors))
        assert assessment.nees.verdict is REFERENCE[filter_name, sensors][3]
        assert all(report.verdict is Verdict.CONSISTENT for report in assessment.nis)
        assert assessment.diverged_runs == 0

    @pytest.mark.parametrize("filter_name", ["ekf", "ukf"])
    def test_recorded_innovation_statistics(self, filter_name: str) -> None:
        """Mean NIS for each sensor reproduces the baseline."""
        assessment = assess(_campaign(filter_name, "both"))
        for report in assessment.nis:
            sensor = report.statistic.removeprefix("NIS (").removesuffix(")")
            assert report.mean == pytest.approx(REFERENCE_NIS[filter_name, sensor], rel=RTOL)

    def test_fusing_beats_either_sensor_alone(self) -> None:
        """The recorded ordering of the accuracy figures is a real property."""
        for filter_name in ("ekf", "ukf"):
            fused = REFERENCE[filter_name, "both"][0]
            assert fused < REFERENCE[filter_name, "lidar"][0]
            assert fused < REFERENCE[filter_name, "radar"][0]


class TestRecordedCounts:
    """Integer counts, which are exact and cannot drift."""

    def test_measurement_counts(self) -> None:
        """The schedule produces exactly these many reports."""
        scenario = simulate(turning_target(steps=STEPS), seed=SEED)
        lidar = sum(1 for m in scenario.measurements if m.sensor_name == "lidar")
        radar = sum(1 for m in scenario.measurements if m.sensor_name == "radar")
        assert (lidar, radar) == (21, 27)

    @pytest.mark.parametrize(
        ("policy", "budget", "processed", "discarded"),
        [
            (OutOfOrderPolicy.BUFFER, 0.15, 48, 0),
            (OutOfOrderPolicy.BUFFER, 0.0, 28, 20),
            (OutOfOrderPolicy.DISCARD, 0.0, 28, 20),
        ],
    )
    def test_policy_bookkeeping(
        self, policy: OutOfOrderPolicy, budget: float, processed: int, discarded: int
    ) -> None:
        """Each out-of-order policy processes and drops exactly these many reports."""
        config = with_latency(turning_target(steps=STEPS))
        scenario = simulate(config, seed=SEED)
        trace = run_filter(
            UnscentedKalmanFilter(config.truth_model),
            scenario,
            matched_belief(scenario),
            FusionSettings(policy, budget),
        )
        assert trace.out_of_order_arrivals == 20
        assert trace.processed == processed
        assert trace.discarded == discarded


class TestClosedFormQuantities:
    """Values that follow from a formula and are reproducible by construction."""

    @pytest.mark.parametrize(
        ("dof", "lower", "upper"),
        [
            (2, 1.0334291847870365, 3.2803397522169924),
            (4, 2.5628754757810768, 5.751882149138839),
            (5, 3.3734790035701523, 6.941472906431099),
        ],
    )
    def test_chi_square_intervals(self, dof: int, lower: float, upper: float) -> None:
        """The twelve-run intervals are the recorded chi-square quantiles."""
        computed = chi2_interval(dof, RUNS)
        assert computed[0] == pytest.approx(lower, rel=1e-12)
        assert computed[1] == pytest.approx(upper, rel=1e-12)

    def test_default_sigma_point_weights(self) -> None:
        """With alpha one and kappa zero the weights take these exact values."""
        mean_weights, cov_weights = ScaledUnscentedSpec().weights(5)
        assert mean_weights.shape == (11,)
        assert float(mean_weights[0]) == 0.0
        assert np.allclose(mean_weights[1:], 0.1, atol=1e-15)
        assert float(cov_weights[0]) == 2.0
        assert np.allclose(cov_weights[1:], 0.1, atol=1e-15)


class TestMisspecifiedClassification:
    """Mis-specified filters are pinned by classification, not by value.

    A filter running far outside its assumptions carries large gains and is not
    the well-conditioned recursion the tolerance argument in the module
    docstring depends on, so its statistics are asserted qualitatively.
    """

    @staticmethod
    def _assess(density: float) -> FilterAssessment:
        seeder = InitialUncertainty()

        def build(scenario: Scenario) -> tuple[StateEstimator, GaussianState]:
            model = ConstantVelocity(spectral_density=density)
            return UnscentedKalmanFilter(model), seeder.build(model, scenario.truth_cartesian[0])

        return assess(run_monte_carlo(turning_target(steps=STEPS), build, runs=RUNS, seed=SEED))

    def test_starved_filter_is_optimistic_by_a_wide_margin(self) -> None:
        """Too little process noise inflates NEES far above its four bound."""
        assessment = self._assess(0.05)
        assert assessment.nees.verdict is Verdict.OPTIMISTIC
        assert assessment.nees.mean > 20.0

    def test_flooded_filter_is_conservative_by_a_wide_margin(self) -> None:
        """Too much process noise pushes NEES well below its lower bound."""
        assessment = self._assess(4000.0)
        assert assessment.nees.verdict is Verdict.CONSERVATIVE
        assert assessment.nees.mean < 3.0
