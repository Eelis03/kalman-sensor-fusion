"""Tier one: properties of the three filters and the unscented transform.

The strongest correctness check available for a nonlinear filter is that it
reduces to the exact answer when the problem is linear. On a linear Gaussian
problem the Kalman filter is optimal and closed form, so any disagreement from
the extended or unscented filter beyond floating point rounding is a defect in
the implementation and not a modelling choice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sensor_fusion.algorithm.base import (
    GaussianState,
    StateEstimator,
    is_positive_semidefinite,
    safe_cholesky,
    symmetrize,
)
from sensor_fusion.algorithm.ekf import ExtendedKalmanFilter
from sensor_fusion.algorithm.kf import KalmanFilter
from sensor_fusion.algorithm.sigma import (
    ScaledUnscentedSpec,
    sigma_points,
    unscented_covariance,
    unscented_mean,
    unscented_residuals,
)
from sensor_fusion.algorithm.ukf import UnscentedKalmanFilter
from sensor_fusion.model.measurement import Lidar, MeasurementModel, Radar
from sensor_fusion.model.motion import ConstantTurnRate, ConstantVelocity
from sensor_fusion.pipeline.fusion import FusionSettings, matched_belief, run_filter
from sensor_fusion.pipeline.scenarios import boundary_crossing_target, straight_target
from sensor_fusion.pipeline.simulator import Scenario, simulate
from sensor_fusion.pipeline.trace import Measurement

SPECS = (
    ScaledUnscentedSpec(),
    ScaledUnscentedSpec(alpha=0.6, beta=2.0, kappa=1.0),
    ScaledUnscentedSpec(alpha=1.0, beta=0.0, kappa=3.0),
)


def _random_covariance(dim: int, rng: np.random.Generator) -> np.ndarray:
    root = rng.standard_normal((dim, dim))
    return symmetrize(root @ root.T + dim * np.eye(dim))


class TestUnscentedTransform:
    """The transform must be exact wherever exactness is achievable."""

    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"alpha{s.alpha}-kappa{s.kappa}")
    @pytest.mark.parametrize("dim", [2, 4, 5, 6])
    def test_mean_weights_sum_to_one(self, spec: ScaledUnscentedSpec, dim: int) -> None:
        """The mean weights are an affine combination and must sum to one."""
        mean_weights, _ = spec.weights(dim)
        assert float(np.sum(mean_weights)) == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"alpha{s.alpha}-kappa{s.kappa}")
    @pytest.mark.parametrize("dim", [2, 4, 5, 6])
    def test_covariance_weights_sum_to_the_stated_value(
        self, spec: ScaledUnscentedSpec, dim: int
    ) -> None:
        """Covariance weights sum to 1 + (1 - alpha**2 + beta), not to one."""
        _, cov_weights = spec.weights(dim)
        expected = 1.0 + (1.0 - spec.alpha**2 + spec.beta)
        assert float(np.sum(cov_weights)) == pytest.approx(expected, abs=1e-13)

    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"alpha{s.alpha}-kappa{s.kappa}")
    def test_sigma_points_recover_the_mean_and_covariance(self, spec: ScaledUnscentedSpec) -> None:
        """The transform of the identity map returns the input moments exactly."""
        rng = np.random.default_rng(3)
        for dim in (2, 4, 5):
            mean = rng.standard_normal(dim)
            cov = _random_covariance(dim, rng)
            mean_weights, cov_weights = spec.weights(dim)
            points = sigma_points(mean, cov, spec)
            recovered = unscented_mean(points, mean_weights)
            residuals = unscented_residuals(points, recovered)
            assert np.allclose(recovered, mean, atol=1e-10)
            assert np.allclose(
                unscented_covariance(residuals, residuals, cov_weights), cov, atol=1e-9
            )

    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"alpha{s.alpha}-kappa{s.kappa}")
    def test_linear_function_is_reproduced_exactly(self, spec: ScaledUnscentedSpec) -> None:
        """For f(x) = A x + b the transform gives A mu + b and A P A.T exactly."""
        rng = np.random.default_rng(11)
        dim, out = 5, 3
        mean = rng.standard_normal(dim)
        cov = _random_covariance(dim, rng)
        matrix = rng.standard_normal((out, dim))
        offset = rng.standard_normal(out)

        mean_weights, cov_weights = spec.weights(dim)
        points = sigma_points(mean, cov, spec)
        mapped = points @ matrix.T + offset

        recovered = unscented_mean(mapped, mean_weights)
        residuals = unscented_residuals(mapped, recovered)
        recovered_cov = unscented_covariance(residuals, residuals, cov_weights)

        assert np.allclose(recovered, matrix @ mean + offset, atol=1e-10)
        assert np.allclose(recovered_cov, matrix @ cov @ matrix.T, atol=1e-9)

    def test_quadratic_mean_is_exact_where_linearisation_is_not(self) -> None:
        """The transform captures a second-order term that a Jacobian cannot.

        For a quadratic form the exact mean is ``mu' Q mu + trace(Q P)``. A
        first-order linearisation drops the trace term entirely. This is the
        concrete content of the claim that the unscented transform is accurate
        beyond first order, and it is the reason the unscented filter keeps a
        usable covariance where the extended filter does not.
        """
        rng = np.random.default_rng(8)
        spec = ScaledUnscentedSpec()
        dim = 4
        mean = rng.standard_normal(dim)
        root = rng.standard_normal((dim, dim))
        cov = symmetrize(root @ root.T + dim * np.eye(dim))
        form = rng.standard_normal((dim, dim))
        form = 0.5 * (form + form.T)

        exact = float(mean @ form @ mean + np.trace(form @ cov))
        linearised = float(mean @ form @ mean)
        mean_weights, _ = spec.weights(dim)
        points = sigma_points(mean, cov, spec)
        transformed = float(mean_weights @ np.array([point @ form @ point for point in points]))

        assert transformed == pytest.approx(exact, abs=1e-10)
        assert abs(linearised - exact) > 1.0

    def test_default_weights_are_non_negative(self) -> None:
        """The default parameters keep every weight non-negative by design.

        That is what guarantees the recovered covariance is a non-negative
        combination of outer products, hence positive semi-definite whatever the
        nonlinearity does.
        """
        for dim in (2, 4, 5, 6):
            mean_weights, cov_weights = ScaledUnscentedSpec().weights(dim)
            assert float(np.min(mean_weights)) >= 0.0
            assert float(np.min(cov_weights)) >= 0.0

    def test_angular_mean_crosses_the_branch_cut(self) -> None:
        """Averaging angles either side of pi must not return zero."""
        points = np.array([[math.pi - 0.05], [math.pi - 0.01], [-math.pi + 0.03]])
        weights = np.full(3, 1.0 / 3.0)
        plain = float(np.sum(weights * points[:, 0]))
        aware = float(unscented_mean(points, weights, angle_indices=(0,))[0])
        assert abs(plain) < 2.1
        assert abs(abs(aware) - math.pi) < 0.05


class TestLinearEquivalence:
    """On a linear problem all three filters must agree with the exact answer."""

    def test_filters_agree_with_the_exact_kalman_filter(self) -> None:
        """EKF and UKF reproduce the linear Kalman filter to near machine precision.

        Lidar only, because the linear Kalman filter is defined only for a linear
        sensor and the radar is the nonlinear one. With the constant velocity
        motion model this is a fully linear Gaussian problem, where the Kalman
        filter is the exact minimum mean square error estimator.
        """
        scenario = simulate(straight_target(steps=600), seed=4)
        model = scenario.config.truth_model
        settings = FusionSettings(sensors=("lidar",))
        exact = run_filter(KalmanFilter(model), scenario, matched_belief(scenario), settings)
        for estimator in (ExtendedKalmanFilter(model), UnscentedKalmanFilter(model)):
            trace = run_filter(estimator, scenario, matched_belief(scenario), settings)
            assert len(trace.records) == len(exact.records)
            for left, right in zip(exact.records, trace.records, strict=True):
                assert np.allclose(left.mean, right.mean, atol=1e-9, rtol=1e-9)
                assert np.allclose(left.cov, right.cov, atol=1e-9, rtol=1e-9)
                assert left.nis == pytest.approx(right.nis, abs=1e-9, rel=1e-9)

    def test_linear_filter_rejects_a_nonlinear_model(self) -> None:
        """The exact filter refuses work it cannot do exactly."""
        with pytest.raises(ValueError, match="nonlinear"):
            KalmanFilter(ConstantTurnRate())

    def test_linear_filter_rejects_a_nonlinear_sensor(self) -> None:
        """The exact filter refuses a polar measurement."""
        estimator = KalmanFilter(ConstantVelocity())
        state = GaussianState(mean=np.array([1.0, 2.0, 3.0, 4.0]), cov=np.eye(4))
        with pytest.raises(ValueError, match="nonlinear"):
            estimator.update(state, np.array([1.0, 0.1, 0.5]), Radar())


class TestCovarianceInvariants:
    """The covariance must stay a covariance through every step."""

    @pytest.mark.parametrize(
        "estimator_factory",
        [ExtendedKalmanFilter, UnscentedKalmanFilter],
        ids=["ekf", "ukf"],
    )
    def test_covariance_stays_symmetric_and_positive_semidefinite(
        self, estimator_factory: type[ExtendedKalmanFilter] | type[UnscentedKalmanFilter]
    ) -> None:
        """Every predicted and updated covariance passes the symmetry and eigenvalue test."""
        scenario = simulate(boundary_crossing_target(steps=800), seed=9)
        model = scenario.config.truth_model
        estimator: StateEstimator = estimator_factory(model)
        state = matched_belief(scenario)
        clock = 0.0
        checked = 0
        for measurement in scenario.measurements:
            gap = measurement.time - clock
            if gap > 0.0:
                state = estimator.predict(state, gap)
                assert is_positive_semidefinite(state.cov), "prediction broke the covariance"
            result = estimator.update(state, measurement.value, measurement.sensor)
            assert is_positive_semidefinite(result.state.cov), "update broke the covariance"
            assert is_positive_semidefinite(result.innovation_cov)
            state = result.state
            clock = measurement.time
            checked += 1
        assert checked >= 90

    def test_safe_cholesky_recovers_from_a_marginal_matrix(self) -> None:
        """A covariance that is barely indefinite is still factorable."""
        matrix = np.diag(np.array([1.0, 1.0, -1e-18]))
        factor = safe_cholesky(matrix)
        assert factor.shape == (3, 3)
        assert np.allclose(factor @ factor.T, matrix, atol=1e-9)

    def test_safe_cholesky_rejects_a_genuinely_indefinite_matrix(self) -> None:
        """A modelling failure is raised rather than papered over."""
        with pytest.raises(np.linalg.LinAlgError):
            safe_cholesky(np.diag(np.array([1.0, -5.0])))


class _UnwrappedRadar:
    """A radar whose innovation subtracts angles without wrapping.

    This exists only as a negative control: it is the defect the wrapping in
    :class:`~sensor_fusion.model.measurement.Radar` prevents, and the test below
    shows that the defect is not cosmetic.
    """

    def __init__(self, inner: Radar) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        """Name of the wrapped sensor."""
        return self._inner.name

    @property
    def dim(self) -> int:
        """Dimension of the wrapped sensor."""
        return self._inner.dim

    @property
    def is_linear(self) -> bool:
        """The wrapped sensor is nonlinear."""
        return self._inner.is_linear

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """Deliberately empty, so no caller wraps anything."""
        return ()

    @property
    def noise_cov(self) -> np.ndarray:
        """Noise covariance of the wrapped sensor."""
        return self._inner.noise_cov

    @property
    def noise_factor(self) -> np.ndarray:
        """Noise factor of the wrapped sensor."""
        return self._inner.noise_factor

    def predict(self, cartesian: np.ndarray) -> np.ndarray:
        """Delegate to the wrapped sensor."""
        return self._inner.predict(cartesian)

    def jacobian(self, cartesian: np.ndarray) -> np.ndarray:
        """Delegate to the wrapped sensor."""
        return self._inner.jacobian(cartesian)

    def residual(self, observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Subtract without wrapping, which is the defect under test."""
        return np.asarray(observed - predicted, dtype=np.float64)


class TestAngleBoundary:
    """A target crossing the plus or minus pi boundary must not break the filter."""

    def test_scenario_actually_crosses_the_boundary(self) -> None:
        """Guard the guard: the scenario must exercise what the test claims."""
        scenario = simulate(boundary_crossing_target(steps=2000), seed=3)
        bearing = np.arctan2(scenario.truth_cartesian[:, 1], scenario.truth_cartesian[:, 0])
        heading = scenario.truth_states[:, 3]
        assert int(np.count_nonzero(np.abs(np.diff(bearing)) > 3.0)) >= 1
        assert int(np.count_nonzero(np.abs(np.diff(heading)) > 3.0)) >= 1
        ranges = np.hypot(scenario.truth_cartesian[:, 0], scenario.truth_cartesian[:, 1])
        assert float(np.min(ranges)) > 10.0, "range must stay away from the singularity"

    @pytest.mark.parametrize(
        "estimator_factory",
        [ExtendedKalmanFilter, UnscentedKalmanFilter],
        ids=["ekf", "ukf"],
    )
    def test_filter_does_not_diverge_across_the_boundary(
        self, estimator_factory: type[ExtendedKalmanFilter] | type[UnscentedKalmanFilter]
    ) -> None:
        """Position error stays small through every branch cut crossing."""
        scenario = simulate(boundary_crossing_target(steps=2000), seed=3)
        trace = run_filter(
            estimator_factory(scenario.config.truth_model), scenario, matched_belief(scenario)
        )
        errors = np.linalg.norm(trace.position_error, axis=1)
        assert float(np.max(errors)) < 2.0
        assert float(np.mean(trace.nees_state)) < 15.0

    def test_omitting_the_wrap_visibly_breaks_the_filter(self) -> None:
        """The negative control: without wrapping the same run goes badly wrong."""
        scenario = simulate(boundary_crossing_target(steps=2000), seed=3)
        model = scenario.config.truth_model
        replaced: list[Measurement] = []
        for measurement in scenario.measurements:
            sensor: MeasurementModel = measurement.sensor
            if isinstance(sensor, Radar):
                sensor = _UnwrappedRadar(sensor)
            replaced.append(
                Measurement(
                    time=measurement.time,
                    arrival_time=measurement.arrival_time,
                    sensor=sensor,
                    value=measurement.value,
                    truth_index=measurement.truth_index,
                )
            )
        broken = Scenario(
            config=scenario.config,
            times=scenario.times,
            truth_states=scenario.truth_states,
            truth_cartesian=scenario.truth_cartesian,
            measurements=tuple(replaced),
        )

        good = run_filter(ExtendedKalmanFilter(model), scenario, matched_belief(scenario))
        bad = run_filter(ExtendedKalmanFilter(model), broken, matched_belief(broken))
        good_error = float(np.max(np.linalg.norm(good.position_error, axis=1)))
        bad_error = float(np.max(np.linalg.norm(bad.position_error, axis=1)))
        assert bad_error > 20.0 * good_error


class TestSensorIndependence:
    """Two reports at the same instant may be applied one after the other.

    This is what makes the sequential update form of asynchronous fusion
    legitimate: conditional independence of the sensors given the state means
    the joint update factors, so a taller stacked measurement is unnecessary.
    """

    _PRIOR = GaussianState(
        mean=np.array([9.0, -3.0, 2.0, 1.0]), cov=np.diag(np.array([2.0, 3.0, 4.0, 5.0]))
    )

    def test_linear_sensors_commute_exactly(self) -> None:
        """With two linear sensors the order of same-time updates cannot matter."""
        estimator = ExtendedKalmanFilter(ConstantVelocity(spectral_density=1.0))
        coarse, fine = Lidar(sigma_x=0.4, sigma_y=0.5), Lidar(sigma_x=0.1, sigma_y=0.12)
        coarse_value = np.array([9.4, -3.2])
        fine_value = np.array([9.2, -3.05])

        forward = estimator.update(self._PRIOR, coarse_value, coarse).state
        forward = estimator.update(forward, fine_value, fine).state
        backward = estimator.update(self._PRIOR, fine_value, fine).state
        backward = estimator.update(backward, coarse_value, coarse).state

        assert np.allclose(forward.mean, backward.mean, atol=1e-12)
        assert np.allclose(forward.cov, backward.cov, atol=1e-12)

    def test_nonlinear_sensors_commute_only_approximately(self) -> None:
        """With a polar sensor the two orders linearise at different points.

        They must still agree to within the linearisation error, which for this
        prior is of the order of a millimetre. Exact commutation is not claimed
        because it is not true, and asserting it at machine precision would be
        asserting something false.
        """
        estimator = ExtendedKalmanFilter(ConstantVelocity(spectral_density=1.0))
        lidar, radar = Lidar(), Radar()
        lidar_value = np.array([9.4, -3.2])
        radar_value = np.array([9.9, -0.31, 1.4])

        forward = estimator.update(self._PRIOR, lidar_value, lidar).state
        forward = estimator.update(forward, radar_value, radar).state
        backward = estimator.update(self._PRIOR, radar_value, radar).state
        backward = estimator.update(backward, lidar_value, lidar).state

        assert np.allclose(forward.mean, backward.mean, atol=5e-3)
        assert np.allclose(forward.cov, backward.cov, atol=5e-2)
