"""Motion models.

Three target motion models are provided. Each states its process noise
formulation explicitly and supplies the analytic Jacobian the extended Kalman
filter needs.

Constant velocity and constant acceleration use the exact discretisation of a
continuous-time linear system driven by white noise, so the discrete process
noise composes: propagating over two steps of ``dt`` gives the same second
moment as one step of ``2 * dt``. That property is what lets the simulator
generate ground truth on a fine grid while the filter runs on the coarse
measurement grid without the truth becoming an unfair sample.

CTRV is nonlinear and its process noise is the standard piecewise-constant
acceleration injection, which does not compose exactly. Section "Known
limitations" of ``docs/design-notes.md`` records that.

Every model exposes a Cartesian view ``[px, py, vx, vy]`` together with the
Jacobian of that view. The measurement models are written once against the
Cartesian view, and the chain rule turns them into state-space Jacobians for
whichever motion model is in use.

References
----------
Bar-Shalom, Li, and Kirubarajan, *Estimation with Applications to Tracking and
Navigation*, Wiley, 2001, sections 6.2.2 (continuous white noise acceleration),
6.2.3 (continuous Wiener process acceleration), and 11.7.2 (coordinated turn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.model.angles import wrap_scalar_to_pi

__all__ = [
    "CTRV_HEADING",
    "CTRV_POSITION_X",
    "CTRV_POSITION_Y",
    "CTRV_SPEED",
    "CTRV_YAW_RATE",
    "ConstantAcceleration",
    "ConstantTurnRate",
    "ConstantVelocity",
    "MotionModel",
]

# Named indices into the CTRV state vector, so downstream code never uses a bare
# integer literal to reach for the heading.
CTRV_POSITION_X = 0
CTRV_POSITION_Y = 1
CTRV_SPEED = 2
CTRV_HEADING = 3
CTRV_YAW_RATE = 4

# Below this magnitude of the half-turn angle the series expansions are used.
# The cardinal sine itself is well conditioned everywhere and the switch is
# cosmetic for it, but its derivative, ``(u cos u - sin u) / u**2``, subtracts
# two quantities that agree to order ``u**2`` and divides by ``u**2``, so it
# loses roughly two decimal digits for every factor of ten that ``u`` shrinks.
# At this threshold the closed form still carries about four significant digits;
# below it the series is both faster and more accurate.
_SMALL_ANGLE = 1e-6


@runtime_checkable
class MotionModel(Protocol):
    """A target motion model with an analytic Jacobian and process noise.

    Implementations are stateless and immutable. ``predict`` advances a state by
    ``dt`` seconds without noise, ``jacobian`` is the derivative of ``predict``
    with respect to the state, and ``process_noise`` is the covariance of the
    accumulated disturbance over that interval.
    """

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""

    @property
    def dim(self) -> int:
        """Dimension of the state vector."""

    @property
    def is_linear(self) -> bool:
        """True when ``predict`` is exactly ``jacobian(dt) @ state``."""

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """State components that live on the circle and need wrapping."""

    def predict(self, state: FloatArray, dt: float) -> FloatArray:
        """Advance ``state`` by ``dt`` seconds under the noise-free dynamics."""

    def jacobian(self, state: FloatArray, dt: float) -> FloatArray:
        """Return d(predict)/d(state), shape ``(dim, dim)``."""

    def process_noise(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the process noise covariance over ``dt``, shape ``(dim, dim)``."""

    def process_noise_factor(self, state: FloatArray, dt: float) -> FloatArray:
        """Return ``L`` with ``L @ L.T == process_noise(state, dt)``.

        The factor is given in closed form rather than obtained from a numerical
        decomposition. An eigendecomposition would be a poor choice here: the
        sign of each eigenvector is not fixed by the problem, so two LAPACK
        builds can return factors that differ by a reflection and therefore draw
        different noise realisations from the same seed.
        """

    def to_cartesian(self, state: FloatArray) -> FloatArray:
        """Project the state onto ``[px, py, vx, vy]``."""

    def cartesian_jacobian(self, state: FloatArray) -> FloatArray:
        """Return d(to_cartesian)/d(state), shape ``(4, dim)``."""

    def from_cartesian(self, cartesian: FloatArray) -> FloatArray:
        """Best-effort inverse of ``to_cartesian``, used to seed a filter."""

    def normalize(self, state: FloatArray) -> FloatArray:
        """Wrap the angular components of ``state`` into ``[-pi, pi)``."""


def _sinc(value: float) -> float:
    """Return ``sin(value) / value``, defined as one at the origin."""
    if abs(value) < _SMALL_ANGLE:
        squared = value * value
        return 1.0 - squared / 6.0 + squared * squared / 120.0
    return math.sin(value) / value


def _sinc_derivative(value: float) -> float:
    """Return the derivative of ``_sinc``."""
    if abs(value) < _SMALL_ANGLE:
        squared = value * value
        return -value / 3.0 + value * squared / 30.0
    return (value * math.cos(value) - math.sin(value)) / (value * value)


@dataclass(frozen=True, slots=True)
class ConstantVelocity:
    """Two-dimensional constant velocity model on ``[px, py, vx, vy]``.

    Process noise is the continuous white noise acceleration model: the true
    acceleration is a zero-mean white noise with power spectral density
    ``spectral_density`` in each axis, independently. Integrating that over an
    interval of length ``dt`` gives the standard block

    ``q * [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]``

    per axis. This is the exact second moment of the continuous process, not an
    approximation, which is why it composes over sub-intervals.
    """

    spectral_density: float = 2.0

    def __post_init__(self) -> None:
        if self.spectral_density < 0.0:
            raise ValueError("spectral_density must be non-negative")

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""
        return "constant-velocity"

    @property
    def dim(self) -> int:
        """Dimension of the state vector."""
        return 4

    @property
    def is_linear(self) -> bool:
        """The constant velocity model is exactly linear."""
        return True

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """No component of this state lives on the circle."""
        return ()

    def predict(self, state: FloatArray, dt: float) -> FloatArray:
        """Advance ``state`` by ``dt`` seconds under the noise-free dynamics."""
        return np.asarray(self.jacobian(state, dt) @ np.asarray(state, dtype=np.float64))

    def jacobian(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the constant transition matrix; ``state`` is unused."""
        del state
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 2] = dt
        matrix[1, 3] = dt
        return matrix

    def process_noise(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the continuous white noise acceleration covariance."""
        del state
        q = self.spectral_density
        pos_pos = q * dt**3 / 3.0
        pos_vel = q * dt**2 / 2.0
        vel_vel = q * dt
        matrix = np.zeros((4, 4), dtype=np.float64)
        for pos, vel in ((0, 2), (1, 3)):
            matrix[pos, pos] = pos_pos
            matrix[pos, vel] = pos_vel
            matrix[vel, pos] = pos_vel
            matrix[vel, vel] = vel_vel
        return matrix

    def process_noise_factor(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the closed-form Cholesky factor of ``process_noise``."""
        del state
        root_q = math.sqrt(self.spectral_density)
        root_dt = math.sqrt(dt)
        # Cholesky of [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], written out.
        lower_11 = root_dt * dt / math.sqrt(3.0)
        lower_21 = math.sqrt(3.0 * dt) / 2.0
        lower_22 = root_dt / 2.0
        factor = np.zeros((4, 4), dtype=np.float64)
        for pos, vel in ((0, 2), (1, 3)):
            factor[pos, pos] = root_q * lower_11
            factor[vel, pos] = root_q * lower_21
            factor[vel, vel] = root_q * lower_22
        return factor

    def to_cartesian(self, state: FloatArray) -> FloatArray:
        """The state already is the Cartesian view."""
        return np.array(state, dtype=np.float64, copy=True)

    def cartesian_jacobian(self, state: FloatArray) -> FloatArray:
        """Return the identity; ``state`` is unused."""
        del state
        return np.eye(4, dtype=np.float64)

    def from_cartesian(self, cartesian: FloatArray) -> FloatArray:
        """Return the Cartesian vector unchanged."""
        return np.array(cartesian, dtype=np.float64, copy=True)

    def normalize(self, state: FloatArray) -> FloatArray:
        """Return ``state`` unchanged; nothing here lives on the circle."""
        return np.asarray(state, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ConstantAcceleration:
    """Constant acceleration model on ``[px, py, vx, vy, ax, ay]``.

    Process noise is the continuous Wiener process acceleration model: the jerk
    is a zero-mean white noise with power spectral density ``spectral_density``
    per axis. The accumulated covariance over ``dt`` is the exact second moment
    of that process and therefore composes over sub-intervals in the same way as
    the constant velocity block.
    """

    spectral_density: float = 1.0

    def __post_init__(self) -> None:
        if self.spectral_density < 0.0:
            raise ValueError("spectral_density must be non-negative")

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""
        return "constant-acceleration"

    @property
    def dim(self) -> int:
        """Dimension of the state vector."""
        return 6

    @property
    def is_linear(self) -> bool:
        """The constant acceleration model is exactly linear."""
        return True

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """No component of this state lives on the circle."""
        return ()

    def predict(self, state: FloatArray, dt: float) -> FloatArray:
        """Advance ``state`` by ``dt`` seconds under the noise-free dynamics."""
        return np.asarray(self.jacobian(state, dt) @ np.asarray(state, dtype=np.float64))

    def jacobian(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the constant transition matrix; ``state`` is unused."""
        del state
        matrix = np.eye(6, dtype=np.float64)
        for pos, vel, acc in ((0, 2, 4), (1, 3, 5)):
            matrix[pos, vel] = dt
            matrix[pos, acc] = 0.5 * dt * dt
            matrix[vel, acc] = dt
        return matrix

    def process_noise(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the continuous Wiener process acceleration covariance."""
        del state
        q = self.spectral_density
        block = q * np.array(
            [
                [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
                [dt**3 / 6.0, dt**2 / 2.0, dt],
            ],
            dtype=np.float64,
        )
        matrix = np.zeros((6, 6), dtype=np.float64)
        for axis in ((0, 2, 4), (1, 3, 5)):
            for row, source_row in enumerate(axis):
                for column, source_column in enumerate(axis):
                    matrix[source_row, source_column] = block[row, column]
        return matrix

    def process_noise_factor(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the closed-form Cholesky factor of ``process_noise``."""
        del state
        root_q = math.sqrt(self.spectral_density)
        root_dt = math.sqrt(dt)
        # Cholesky of the continuous Wiener process acceleration block, written out.
        block = np.array(
            [
                [dt * dt * root_dt / (2.0 * math.sqrt(5.0)), 0.0, 0.0],
                [math.sqrt(5.0) * dt * root_dt / 4.0, dt * root_dt / (4.0 * math.sqrt(3.0)), 0.0],
                [math.sqrt(5.0) * root_dt / 3.0, root_dt / math.sqrt(3.0), root_dt / 3.0],
            ],
            dtype=np.float64,
        )
        factor = np.zeros((6, 6), dtype=np.float64)
        for axis in ((0, 2, 4), (1, 3, 5)):
            for row, source_row in enumerate(axis):
                for column, source_column in enumerate(axis):
                    factor[source_row, source_column] = root_q * block[row, column]
        return factor

    def to_cartesian(self, state: FloatArray) -> FloatArray:
        """Drop the acceleration components."""
        return np.asarray(np.asarray(state, dtype=np.float64)[:4], dtype=np.float64).copy()

    def cartesian_jacobian(self, state: FloatArray) -> FloatArray:
        """Return the selection matrix for the first four components."""
        del state
        matrix = np.zeros((4, 6), dtype=np.float64)
        matrix[:4, :4] = np.eye(4, dtype=np.float64)
        return matrix

    def from_cartesian(self, cartesian: FloatArray) -> FloatArray:
        """Seed position and velocity from ``cartesian`` and zero the acceleration."""
        state = np.zeros(6, dtype=np.float64)
        state[:4] = np.asarray(cartesian, dtype=np.float64)
        return state

    def normalize(self, state: FloatArray) -> FloatArray:
        """Return ``state`` unchanged; nothing here lives on the circle."""
        return np.asarray(state, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ConstantTurnRate:
    """Constant turn rate and velocity model on ``[px, py, v, psi, omega]``.

    The noise-free propagation of the exact CTRV solution is written using the
    cardinal sine so that it stays well conditioned as the yaw rate approaches
    zero. With ``u = omega * dt / 2``,

    ``px' = px + v * dt * sinc(u) * cos(psi + u)``
    ``py' = py + v * dt * sinc(u) * sin(psi + u)``

    which reduces continuously to the straight-line limit at ``omega = 0``
    without a case split in the code. The common textbook form divides by
    ``omega`` and needs a branch; branching makes the analytic Jacobian
    disagree with a finite difference taken across the branch, which is exactly
    the kind of defect the Jacobian test is meant to catch.

    Process noise is continuous white noise on the longitudinal acceleration and
    on the yaw acceleration, with power spectral densities
    ``spectral_density_accel`` and ``spectral_density_yaw``. That gives two
    independent continuous white noise acceleration chains, one from
    acceleration into speed into along-track position, and one from yaw
    acceleration into yaw rate into heading, each carrying the exact block

    ``q * [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]``.

    The commonly published alternative injects an acceleration held constant
    over the step, parameterised by a standard deviation rather than a spectral
    density. That form is rejected here because it does not compose: halving the
    step size halves the injected variance, so ground truth generated on a fine
    grid would carry far less disturbance than a filter running on the coarse
    measurement grid assumes, and every consistency test would report a
    conservative filter for a reason that has nothing to do with the filter. The
    continuous-time form composes exactly in the speed, heading, and yaw rate
    components, and approximately in position, because the along-track direction
    rotates within the interval. The position error scales as the square of the
    yaw rate times ``dt`` and is 6e-4 relative at the largest step this package
    uses.
    """

    spectral_density_accel: float = 0.5
    spectral_density_yaw: float = 0.01

    def __post_init__(self) -> None:
        if self.spectral_density_accel < 0.0 or self.spectral_density_yaw < 0.0:
            raise ValueError("spectral densities must be non-negative")

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""
        return "ctrv"

    @property
    def dim(self) -> int:
        """Dimension of the state vector."""
        return 5

    @property
    def is_linear(self) -> bool:
        """CTRV is nonlinear in the heading and the yaw rate."""
        return False

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """The heading lives on the circle."""
        return (CTRV_HEADING,)

    def predict(self, state: FloatArray, dt: float) -> FloatArray:
        """Advance ``state`` by ``dt`` seconds under the noise-free dynamics.

        The heading is deliberately left unwrapped here. Wrapping inside the
        propagation would make it discontinuous at the plus or minus pi
        boundary, and a discontinuous propagation cannot agree with its own
        analytic Jacobian under a finite difference taken across that point.
        Wrapping is the job of ``normalize``, which the filters apply to the
        mean once the linear algebra is done.
        """
        values = np.asarray(state, dtype=np.float64)
        speed = float(values[CTRV_SPEED])
        heading = float(values[CTRV_HEADING])
        yaw_rate = float(values[CTRV_YAW_RATE])

        half = 0.5 * yaw_rate * dt
        chord = speed * dt * _sinc(half)
        midpoint = heading + half

        nxt = np.array(values, dtype=np.float64, copy=True)
        nxt[CTRV_POSITION_X] = values[CTRV_POSITION_X] + chord * math.cos(midpoint)
        nxt[CTRV_POSITION_Y] = values[CTRV_POSITION_Y] + chord * math.sin(midpoint)
        nxt[CTRV_HEADING] = heading + yaw_rate * dt
        return nxt

    def jacobian(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the analytic Jacobian of ``predict``.

        The derivative is taken of the cardinal sine form, so it is valid at
        every yaw rate including zero and matches a central finite difference
        everywhere rather than only away from a branch boundary.
        """
        values = np.asarray(state, dtype=np.float64)
        speed = float(values[CTRV_SPEED])
        heading = float(values[CTRV_HEADING])
        yaw_rate = float(values[CTRV_YAW_RATE])

        half = 0.5 * yaw_rate * dt
        kernel = _sinc(half)
        kernel_prime = _sinc_derivative(half)
        midpoint = heading + half
        cos_mid = math.cos(midpoint)
        sin_mid = math.sin(midpoint)

        matrix = np.eye(5, dtype=np.float64)
        matrix[CTRV_POSITION_X, CTRV_SPEED] = dt * kernel * cos_mid
        matrix[CTRV_POSITION_X, CTRV_HEADING] = -speed * dt * kernel * sin_mid
        matrix[CTRV_POSITION_X, CTRV_YAW_RATE] = (
            speed * dt * 0.5 * dt * (kernel_prime * cos_mid - kernel * sin_mid)
        )
        matrix[CTRV_POSITION_Y, CTRV_SPEED] = dt * kernel * sin_mid
        matrix[CTRV_POSITION_Y, CTRV_HEADING] = speed * dt * kernel * cos_mid
        matrix[CTRV_POSITION_Y, CTRV_YAW_RATE] = (
            speed * dt * 0.5 * dt * (kernel_prime * sin_mid + kernel * cos_mid)
        )
        matrix[CTRV_HEADING, CTRV_YAW_RATE] = dt
        return matrix

    def process_noise(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the continuous white noise acceleration covariance.

        The result has rank four, not five: only two independent scalar
        disturbances act on the target and position is driven through a single
        along-track direction, so the covariance is positive semi-definite
        rather than positive definite. The filters never factor this matrix on
        its own, only the sum with a positive definite prior, so the deficiency
        is harmless.
        """
        values = np.asarray(state, dtype=np.float64)
        heading = float(values[CTRV_HEADING])
        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)

        accel = self.spectral_density_accel
        yaw = self.spectral_density_yaw
        pos_pos = dt**3 / 3.0
        pos_rate = dt**2 / 2.0
        rate_rate = dt

        matrix = np.zeros((5, 5), dtype=np.float64)
        direction = np.array([cos_heading, sin_heading], dtype=np.float64)
        matrix[:2, :2] = accel * pos_pos * np.outer(direction, direction)
        matrix[:2, CTRV_SPEED] = accel * pos_rate * direction
        matrix[CTRV_SPEED, :2] = accel * pos_rate * direction
        matrix[CTRV_SPEED, CTRV_SPEED] = accel * rate_rate
        matrix[CTRV_HEADING, CTRV_HEADING] = yaw * pos_pos
        matrix[CTRV_HEADING, CTRV_YAW_RATE] = yaw * pos_rate
        matrix[CTRV_YAW_RATE, CTRV_HEADING] = yaw * pos_rate
        matrix[CTRV_YAW_RATE, CTRV_YAW_RATE] = yaw * rate_rate
        return matrix

    def process_noise_factor(self, state: FloatArray, dt: float) -> FloatArray:
        """Return the closed-form factor of ``process_noise``, shape ``(5, 4)``.

        Columns one and two carry the acceleration chain, projected onto the
        current heading for the position rows; columns three and four carry the
        yaw acceleration chain.
        """
        values = np.asarray(state, dtype=np.float64)
        heading = float(values[CTRV_HEADING])
        root_dt = math.sqrt(dt)
        # Cholesky of [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], written out.
        lower_11 = root_dt * dt / math.sqrt(3.0)
        lower_21 = math.sqrt(3.0 * dt) / 2.0
        lower_22 = root_dt / 2.0

        root_accel = math.sqrt(self.spectral_density_accel)
        root_yaw = math.sqrt(self.spectral_density_yaw)

        factor = np.zeros((5, 4), dtype=np.float64)
        factor[CTRV_POSITION_X, 0] = root_accel * lower_11 * math.cos(heading)
        factor[CTRV_POSITION_Y, 0] = root_accel * lower_11 * math.sin(heading)
        factor[CTRV_SPEED, 0] = root_accel * lower_21
        factor[CTRV_SPEED, 1] = root_accel * lower_22
        factor[CTRV_HEADING, 2] = root_yaw * lower_11
        factor[CTRV_YAW_RATE, 2] = root_yaw * lower_21
        factor[CTRV_YAW_RATE, 3] = root_yaw * lower_22
        return factor

    def to_cartesian(self, state: FloatArray) -> FloatArray:
        """Resolve the polar velocity onto the Cartesian axes."""
        values = np.asarray(state, dtype=np.float64)
        speed = float(values[CTRV_SPEED])
        heading = float(values[CTRV_HEADING])
        return np.array(
            [
                values[CTRV_POSITION_X],
                values[CTRV_POSITION_Y],
                speed * math.cos(heading),
                speed * math.sin(heading),
            ],
            dtype=np.float64,
        )

    def cartesian_jacobian(self, state: FloatArray) -> FloatArray:
        """Return d(to_cartesian)/d(state), shape ``(4, 5)``."""
        values = np.asarray(state, dtype=np.float64)
        speed = float(values[CTRV_SPEED])
        heading = float(values[CTRV_HEADING])
        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)
        matrix = np.zeros((4, 5), dtype=np.float64)
        matrix[0, CTRV_POSITION_X] = 1.0
        matrix[1, CTRV_POSITION_Y] = 1.0
        matrix[2, CTRV_SPEED] = cos_heading
        matrix[2, CTRV_HEADING] = -speed * sin_heading
        matrix[3, CTRV_SPEED] = sin_heading
        matrix[3, CTRV_HEADING] = speed * cos_heading
        return matrix

    def from_cartesian(self, cartesian: FloatArray) -> FloatArray:
        """Convert ``[px, py, vx, vy]`` to CTRV with a zero yaw rate."""
        values = np.asarray(cartesian, dtype=np.float64)
        velocity_x = float(values[2])
        velocity_y = float(values[3])
        return np.array(
            [
                values[0],
                values[1],
                math.hypot(velocity_x, velocity_y),
                math.atan2(velocity_y, velocity_x),
                0.0,
            ],
            dtype=np.float64,
        )

    def normalize(self, state: FloatArray) -> FloatArray:
        """Wrap the heading into ``[-pi, pi)``."""
        values = np.array(state, dtype=np.float64, copy=True)
        values[CTRV_HEADING] = wrap_scalar_to_pi(float(values[CTRV_HEADING]))
        return values
