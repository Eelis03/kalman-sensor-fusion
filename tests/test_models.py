"""Tier one: properties of the motion and measurement models.

The central check here is that every analytic Jacobian agrees with a central
finite difference of the function it claims to differentiate. A Jacobian with a
sign error or a missing term does not raise; the filter absorbs it into the gain
and tracks acceptably for a while before diverging, so nothing short of this
comparison catches it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from sensor_fusion.model.angles import wrap_scalar_to_pi, wrap_to_pi
from sensor_fusion.model.measurement import Lidar, MeasurementModel, Radar
from sensor_fusion.model.motion import (
    CTRV_HEADING,
    ConstantAcceleration,
    ConstantTurnRate,
    ConstantVelocity,
    MotionModel,
)

FD_STEP = 1e-6

MOTION_MODELS: tuple[MotionModel, ...] = (
    ConstantVelocity(spectral_density=2.0),
    ConstantAcceleration(spectral_density=1.5),
    ConstantTurnRate(spectral_density_accel=0.5, spectral_density_yaw=0.02),
)

MOTION_STATES: dict[str, tuple[np.ndarray, ...]] = {
    "constant-velocity": (
        np.array([3.0, -4.0, 2.0, 5.0]),
        np.array([-11.0, 0.5, -3.0, 0.25]),
    ),
    "constant-acceleration": (
        np.array([3.0, -4.0, 2.0, 5.0, 0.7, -0.3]),
        np.array([-11.0, 0.5, -3.0, 0.25, -1.1, 2.0]),
    ),
    "ctrv": (
        np.array([3.0, -4.0, 9.0, 0.7, 0.4]),
        np.array([-11.0, 0.5, 12.0, -2.9, -0.85]),
        # A yaw rate at the cardinal sine series boundary and one exactly at zero,
        # which is where a branching implementation would disagree with itself.
        np.array([5.0, 5.0, 7.0, 3.0, 1e-9]),
        np.array([5.0, 5.0, 7.0, 3.0, 0.0]),
    ),
}

CARTESIAN_STATES: tuple[np.ndarray, ...] = (
    np.array([12.0, 5.0, 3.0, -2.0]),
    np.array([-8.0, -30.0, -6.0, 1.5]),
    np.array([0.5, -40.0, 9.0, 9.0]),
)


def _central_difference(
    function: Callable[[np.ndarray], np.ndarray], point: np.ndarray, columns: int
) -> np.ndarray:
    """Return the central finite difference Jacobian of ``function`` at ``point``."""
    reference = np.asarray(function(point))
    jacobian = np.zeros((reference.size, columns), dtype=np.float64)
    for column in range(columns):
        step = np.zeros(point.size, dtype=np.float64)
        step[column] = FD_STEP
        forward = np.asarray(function(point + step))
        backward = np.asarray(function(point - step))
        jacobian[:, column] = (forward - backward) / (2.0 * FD_STEP)
    return jacobian


class TestAngles:
    """Wrapping is the foundation every angular quantity depends on."""

    @pytest.mark.parametrize(
        "angle", [0.0, 1.0, -1.0, 3.0, -3.0, 7.0, -7.0, 100.0, math.pi - 1e-9]
    )
    def test_wrap_lands_in_range(self, angle: float) -> None:
        """Every wrapped angle lies in the half-open interval [-pi, pi)."""
        wrapped = wrap_scalar_to_pi(angle)
        assert -math.pi <= wrapped < math.pi

    @pytest.mark.parametrize("angle", [0.0, 1.0, -1.0, 3.0, -3.0, 7.0, -7.0])
    def test_wrap_preserves_the_direction(self, angle: float) -> None:
        """Wrapping changes an angle only by whole turns."""
        wrapped = wrap_scalar_to_pi(angle)
        turns = (angle - wrapped) / (2.0 * math.pi)
        assert abs(turns - round(turns)) < 1e-12

    def test_wrap_is_idempotent(self) -> None:
        """Wrapping an already wrapped angle changes nothing."""
        angles = np.linspace(-20.0, 20.0, 401)
        once = wrap_to_pi(angles)
        assert np.allclose(once, wrap_to_pi(once), atol=0.0, rtol=0.0)

    def test_difference_across_the_boundary_is_small(self) -> None:
        """The distance from just below pi to just above minus pi is small."""
        left = np.array([math.pi - 0.01])
        right = np.array([-math.pi + 0.01])
        assert abs(float(wrap_to_pi(right - left)[0])) == pytest.approx(0.02, abs=1e-12)


class TestMotionJacobians:
    """Analytic motion Jacobians against central finite differences."""

    @pytest.mark.parametrize("model", MOTION_MODELS, ids=lambda m: m.name)
    @pytest.mark.parametrize("dt", [0.01, 0.075, 0.5])
    def test_jacobian_matches_finite_difference(self, model: MotionModel, dt: float) -> None:
        """The analytic Jacobian reproduces a central difference of predict."""
        for state in MOTION_STATES[model.name]:
            analytic = model.jacobian(state, dt)
            numeric = _central_difference(
                lambda point, model=model, dt=dt: model.predict(point, dt), state, model.dim
            )
            assert np.allclose(analytic, numeric, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("model", MOTION_MODELS, ids=lambda m: m.name)
    def test_cartesian_jacobian_matches_finite_difference(self, model: MotionModel) -> None:
        """The Cartesian projection Jacobian reproduces a central difference."""
        for state in MOTION_STATES[model.name]:
            analytic = model.cartesian_jacobian(state)
            numeric = _central_difference(model.to_cartesian, state, model.dim)
            assert np.allclose(analytic, numeric, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("model", MOTION_MODELS, ids=lambda m: m.name)
    def test_linear_models_are_exactly_their_jacobian(self, model: MotionModel) -> None:
        """A model claiming linearity must satisfy predict(x) == F @ x."""
        if not model.is_linear:
            pytest.skip("model is nonlinear by declaration")
        for state in MOTION_STATES[model.name]:
            transition = model.jacobian(state, 0.13)
            assert np.allclose(model.predict(state, 0.13), transition @ state, atol=1e-14)


class TestProcessNoise:
    """Structure and reproducibility of the process noise."""

    @pytest.mark.parametrize("model", MOTION_MODELS, ids=lambda m: m.name)
    @pytest.mark.parametrize("dt", [0.01, 0.075, 0.5])
    def test_factor_reproduces_the_covariance(self, model: MotionModel, dt: float) -> None:
        """The closed-form factor L satisfies L @ L.T == Q to machine precision."""
        for state in MOTION_STATES[model.name]:
            factor = model.process_noise_factor(state, dt)
            covariance = model.process_noise(state, dt)
            reconstructed = factor @ factor.T
            scale = max(float(np.max(np.abs(covariance))), 1e-30)
            assert np.allclose(reconstructed, covariance, atol=1e-12 * scale, rtol=1e-12)

    @pytest.mark.parametrize("model", MOTION_MODELS, ids=lambda m: m.name)
    def test_covariance_is_symmetric_positive_semidefinite(self, model: MotionModel) -> None:
        """Process noise is a covariance and must behave like one."""
        for state in MOTION_STATES[model.name]:
            covariance = model.process_noise(state, 0.075)
            assert np.allclose(covariance, covariance.T, atol=1e-15)
            eigenvalues = np.linalg.eigvalsh(covariance)
            assert float(np.min(eigenvalues)) > -1e-12 * max(
                float(np.max(np.abs(covariance))), 1.0
            )

    @pytest.mark.parametrize("model", MOTION_MODELS[:2], ids=lambda m: m.name)
    def test_linear_process_noise_composes_over_substeps(self, model: MotionModel) -> None:
        """One step of 2 dt accumulates the same covariance as two steps of dt.

        This is the property that lets the simulator generate truth on a fine
        grid while the filter runs on the coarse measurement grid. It holds
        exactly for the continuous-time discretisations used by the linear
        models, and only approximately for CTRV, which is why CTRV is excluded.
        """
        state = MOTION_STATES[model.name][0]
        dt = 0.04
        single = model.process_noise(state, 2.0 * dt)
        transition = model.jacobian(state, dt)
        accumulated = (
            transition @ model.process_noise(state, dt) @ transition.T
            + model.process_noise(state, dt)
        )
        assert np.allclose(single, accumulated, atol=1e-15, rtol=1e-12)


class TestConstantVelocityExactness:
    """The noiseless constant velocity prediction has a closed form."""

    def test_zero_noise_prediction_is_exact(self) -> None:
        """With zero process noise the prediction is exactly x + v dt."""
        model = ConstantVelocity(spectral_density=0.0)
        state = np.array([1.5, -2.5, 4.0, -3.0])
        dt = 0.25
        predicted = model.predict(state, dt)
        assert predicted[0] == pytest.approx(1.5 + 4.0 * dt, abs=0.0, rel=0.0)
        assert predicted[1] == pytest.approx(-2.5 - 3.0 * dt, abs=0.0, rel=0.0)
        assert predicted[2] == 4.0
        assert predicted[3] == -3.0
        assert np.count_nonzero(model.process_noise(state, dt)) == 0

    def test_repeated_zero_noise_prediction_matches_one_long_step(self) -> None:
        """Twenty short predictions equal one long prediction exactly."""
        model = ConstantVelocity(spectral_density=0.0)
        state = np.array([1.5, -2.5, 4.0, -3.0])
        stepped = state
        for _ in range(20):
            stepped = model.predict(stepped, 0.05)
        assert np.allclose(stepped, model.predict(state, 1.0), atol=1e-14)


class TestCtrvGeometry:
    """CTRV must reduce to straight-line motion and turn on the right circle."""

    def test_zero_yaw_rate_gives_straight_line_motion(self) -> None:
        """At zero yaw rate the target travels v dt along its heading."""
        model = ConstantTurnRate(spectral_density_accel=0.0, spectral_density_yaw=0.0)
        state = np.array([2.0, -1.0, 6.0, 0.35, 0.0])
        predicted = model.predict(state, 0.4)
        assert predicted[0] == pytest.approx(2.0 + 6.0 * 0.4 * math.cos(0.35), abs=1e-12)
        assert predicted[1] == pytest.approx(-1.0 + 6.0 * 0.4 * math.sin(0.35), abs=1e-12)
        assert predicted[CTRV_HEADING] == pytest.approx(0.35, abs=1e-15)

    def test_constant_turn_stays_on_its_circle(self) -> None:
        """A turning target keeps a constant distance from the turn centre."""
        model = ConstantTurnRate(spectral_density_accel=0.0, spectral_density_yaw=0.0)
        speed, yaw_rate, heading = 8.0, 0.4, 0.9
        radius = speed / yaw_rate
        state = np.array([0.0, 0.0, speed, heading, yaw_rate])
        centre = np.array(
            [radius * math.cos(heading + 0.5 * math.pi), radius * math.sin(heading + 0.5 * math.pi)]
        )
        for _ in range(50):
            state = model.normalize(model.predict(state, 0.05))
            assert float(np.linalg.norm(state[:2] - centre)) == pytest.approx(radius, abs=1e-10)

    def test_small_yaw_rate_is_continuous(self) -> None:
        """The cardinal sine form has no step at the series expansion boundary.

        Only the position is compared. The heading and the yaw rate necessarily
        differ between the sweep points because the yaw rate is what is being
        varied; the question is whether the position, which is where the series
        expansion is used, moves smoothly with it.
        """
        model = ConstantTurnRate()
        base = np.array([1.0, 2.0, 7.0, 0.5, 0.0])
        previous = model.predict(base, 0.1)
        for exponent in range(-12, -3):
            state = np.array([1.0, 2.0, 7.0, 0.5, 10.0**exponent])
            current = model.predict(state, 0.1)
            assert np.allclose(current[:2], previous[:2], atol=1e-8)
            previous = current


class TestMeasurementModels:
    """Sensor Jacobians, residuals, and noise factors."""

    @pytest.mark.parametrize("sensor", [Lidar(), Radar()], ids=lambda s: s.name)
    def test_jacobian_matches_finite_difference(self, sensor: MeasurementModel) -> None:
        """The analytic sensor Jacobian reproduces a central difference."""
        for cartesian in CARTESIAN_STATES:
            analytic = sensor.jacobian(cartesian)
            numeric = _central_difference(sensor.predict, cartesian, 4)
            assert np.allclose(analytic, numeric, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("sensor", [Lidar(), Radar()], ids=lambda s: s.name)
    def test_noise_factor_reproduces_the_covariance(self, sensor: MeasurementModel) -> None:
        """The sensor noise factor L satisfies L @ L.T == R."""
        factor = sensor.noise_factor
        assert np.allclose(factor @ factor.T, sensor.noise_cov, atol=1e-15)

    def test_radar_measurement_is_the_polar_transform(self) -> None:
        """Range, bearing, and range rate match their definitions."""
        radar = Radar()
        cartesian = np.array([3.0, 4.0, 1.0, -2.0])
        predicted = radar.predict(cartesian)
        assert predicted[0] == pytest.approx(5.0, abs=1e-12)
        assert predicted[1] == pytest.approx(math.atan2(4.0, 3.0), abs=1e-12)
        assert predicted[2] == pytest.approx((3.0 * 1.0 + 4.0 * -2.0) / 5.0, abs=1e-12)

    def test_radar_residual_wraps_the_bearing(self) -> None:
        """A bearing residual across the branch cut is small, not almost two pi."""
        radar = Radar()
        observed = np.array([10.0, math.pi - 0.01, 1.0])
        predicted = np.array([10.0, -math.pi + 0.01, 1.0])
        residual = radar.residual(observed, predicted)
        assert abs(float(residual[1])) == pytest.approx(0.02, abs=1e-12)
        assert float(residual[0]) == 0.0

    def test_lidar_residual_is_a_plain_difference(self) -> None:
        """Nothing in a lidar measurement lives on the circle."""
        lidar = Lidar()
        residual = lidar.residual(np.array([4.0, 9.0]), np.array([1.0, 2.0]))
        assert np.allclose(residual, np.array([3.0, 7.0]), atol=0.0)

    def test_invalid_noise_is_rejected(self) -> None:
        """A non-positive standard deviation is a configuration error."""
        with pytest.raises(ValueError, match="positive"):
            Lidar(sigma_x=0.0)
        with pytest.raises(ValueError, match="positive"):
            Radar(sigma_bearing=-1.0)
