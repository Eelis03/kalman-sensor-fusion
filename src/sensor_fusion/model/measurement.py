"""Measurement models.

Both sensors are written against the Cartesian view ``[px, py, vx, vy]`` that
every motion model exposes. The chain rule in the extended Kalman filter turns
``d(measurement)/d(cartesian)`` into ``d(measurement)/d(state)`` by
right-multiplying with the motion model's Cartesian Jacobian, so a sensor is
written once and works with every motion model.

Lidar returns Cartesian position and is linear. Radar returns range, bearing,
and range rate, and is the nonlinear sensor that makes the comparison between
the extended and unscented filters meaningful: with only lidar, both filters and
the linear Kalman filter agree to machine precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.model.angles import wrap_to_pi

__all__ = [
    "RADAR_BEARING",
    "RADAR_RANGE",
    "RADAR_RANGE_RATE",
    "Lidar",
    "MeasurementModel",
    "Radar",
]

RADAR_RANGE = 0
RADAR_BEARING = 1
RADAR_RANGE_RATE = 2

# Range below which the radar geometry is treated as degenerate. At the sensor
# origin the bearing is undefined and the range rate Jacobian is unbounded, so
# both ``predict`` and ``jacobian`` clamp the range to this value to keep the
# arithmetic finite rather than returning infinities into a covariance.
#
# Inside the clamp the returned Jacobian is not the derivative of the clamped
# ``predict``: the true derivative of a clamped function is zero, while what is
# returned is the unclamped derivative evaluated at the clamp radius. That
# inconsistency is deliberate and harmless, because no filter can produce a
# meaningful update from a target sitting on top of the sensor and returning a
# zero Jacobian would silently freeze the estimate instead of making the problem
# visible. Every scenario in this package keeps the target more than 15 m away,
# which a test asserts over 200 seeds, so the clamp is never reached.
_MIN_RANGE = 1e-4


@runtime_checkable
class MeasurementModel(Protocol):
    """A sensor mapping the Cartesian target state to an observation."""

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""

    @property
    def dim(self) -> int:
        """Dimension of the measurement vector."""

    @property
    def is_linear(self) -> bool:
        """True when ``predict`` is exactly ``jacobian @ cartesian``."""

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """Measurement components that live on the circle."""

    @property
    def noise_cov(self) -> FloatArray:
        """Measurement noise covariance, shape ``(dim, dim)``."""

    @property
    def noise_factor(self) -> FloatArray:
        """``L`` with ``L @ L.T == noise_cov``, used by the simulator."""

    def predict(self, cartesian: FloatArray) -> FloatArray:
        """Return the noise-free measurement of ``[px, py, vx, vy]``."""

    def jacobian(self, cartesian: FloatArray) -> FloatArray:
        """Return d(predict)/d(cartesian), shape ``(dim, 4)``."""

    def residual(self, observed: FloatArray, predicted: FloatArray) -> FloatArray:
        """Return ``observed - predicted`` with angular components wrapped."""


def _wrap_angles(vector: FloatArray, angle_indices: tuple[int, ...]) -> FloatArray:
    if not angle_indices:
        return vector
    wrapped = np.array(vector, dtype=np.float64, copy=True)
    index = np.asarray(angle_indices, dtype=np.intp)
    wrapped[index] = wrap_to_pi(wrapped[index])
    return wrapped


@dataclass(frozen=True, slots=True)
class Lidar:
    """Cartesian position sensor with independent Gaussian noise per axis."""

    sigma_x: float = 0.15
    sigma_y: float = 0.15
    name: str = "lidar"

    def __post_init__(self) -> None:
        if self.sigma_x <= 0.0 or self.sigma_y <= 0.0:
            raise ValueError("lidar noise standard deviations must be positive")

    @property
    def dim(self) -> int:
        """Lidar reports two numbers."""
        return 2

    @property
    def is_linear(self) -> bool:
        """Position extraction is a linear map."""
        return True

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """No lidar component lives on the circle."""
        return ()

    @property
    def noise_cov(self) -> FloatArray:
        """Diagonal position noise covariance."""
        return np.diag(np.array([self.sigma_x**2, self.sigma_y**2], dtype=np.float64))

    @property
    def noise_factor(self) -> FloatArray:
        """Diagonal square root of ``noise_cov``."""
        return np.diag(np.array([self.sigma_x, self.sigma_y], dtype=np.float64))

    def predict(self, cartesian: FloatArray) -> FloatArray:
        """Return the position components of ``cartesian``."""
        return np.array(np.asarray(cartesian, dtype=np.float64)[:2], dtype=np.float64, copy=True)

    def jacobian(self, cartesian: FloatArray) -> FloatArray:
        """Return the constant selection matrix; ``cartesian`` is unused."""
        del cartesian
        matrix = np.zeros((2, 4), dtype=np.float64)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        return matrix

    def residual(self, observed: FloatArray, predicted: FloatArray) -> FloatArray:
        """Return the plain difference; no component needs wrapping."""
        return np.asarray(
            np.asarray(observed, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
        )


@dataclass(frozen=True, slots=True)
class Radar:
    """Range, bearing, and range rate sensor located at the origin.

    The measurement is

    ``rho = hypot(px, py)``
    ``phi = atan2(py, px)``
    ``rho_dot = (px * vx + py * vy) / rho``

    which is nonlinear in position and, through the range in the denominator,
    couples position into the range rate. The bearing is an angle, so the
    innovation ``observed - predicted`` must be wrapped: without the wrap a
    target crossing the plus or minus pi boundary produces an innovation of
    almost ``2 * pi``, the gain applies a correction of that size, and the
    estimate is thrown away from the truth.
    """

    sigma_range: float = 0.3
    sigma_bearing: float = 0.03
    sigma_range_rate: float = 0.3
    name: str = "radar"

    def __post_init__(self) -> None:
        if min(self.sigma_range, self.sigma_bearing, self.sigma_range_rate) <= 0.0:
            raise ValueError("radar noise standard deviations must be positive")

    @property
    def dim(self) -> int:
        """Radar reports three numbers."""
        return 3

    @property
    def is_linear(self) -> bool:
        """The polar measurement is nonlinear."""
        return False

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """The bearing lives on the circle."""
        return (RADAR_BEARING,)

    @property
    def noise_cov(self) -> FloatArray:
        """Diagonal polar noise covariance."""
        return np.diag(
            np.array(
                [self.sigma_range**2, self.sigma_bearing**2, self.sigma_range_rate**2],
                dtype=np.float64,
            )
        )

    @property
    def noise_factor(self) -> FloatArray:
        """Diagonal square root of ``noise_cov``."""
        return np.diag(
            np.array(
                [self.sigma_range, self.sigma_bearing, self.sigma_range_rate],
                dtype=np.float64,
            )
        )

    def predict(self, cartesian: FloatArray) -> FloatArray:
        """Return ``[range, bearing, range rate]`` for ``[px, py, vx, vy]``."""
        values = np.asarray(cartesian, dtype=np.float64)
        pos_x = float(values[0])
        pos_y = float(values[1])
        vel_x = float(values[2])
        vel_y = float(values[3])
        range_ = max(math.hypot(pos_x, pos_y), _MIN_RANGE)
        bearing = math.atan2(pos_y, pos_x)
        range_rate = (pos_x * vel_x + pos_y * vel_y) / range_
        return np.array([range_, bearing, range_rate], dtype=np.float64)

    def jacobian(self, cartesian: FloatArray) -> FloatArray:
        """Return the analytic Jacobian d(predict)/d(cartesian), shape ``(3, 4)``."""
        values = np.asarray(cartesian, dtype=np.float64)
        pos_x = float(values[0])
        pos_y = float(values[1])
        vel_x = float(values[2])
        vel_y = float(values[3])
        range_squared = max(pos_x * pos_x + pos_y * pos_y, _MIN_RANGE**2)
        range_ = math.sqrt(range_squared)
        range_cubed = range_squared * range_

        matrix = np.zeros((3, 4), dtype=np.float64)
        matrix[RADAR_RANGE, 0] = pos_x / range_
        matrix[RADAR_RANGE, 1] = pos_y / range_
        matrix[RADAR_BEARING, 0] = -pos_y / range_squared
        matrix[RADAR_BEARING, 1] = pos_x / range_squared
        matrix[RADAR_RANGE_RATE, 0] = pos_y * (vel_x * pos_y - vel_y * pos_x) / range_cubed
        matrix[RADAR_RANGE_RATE, 1] = pos_x * (vel_y * pos_x - vel_x * pos_y) / range_cubed
        matrix[RADAR_RANGE_RATE, 2] = pos_x / range_
        matrix[RADAR_RANGE_RATE, 3] = pos_y / range_
        return matrix

    def residual(self, observed: FloatArray, predicted: FloatArray) -> FloatArray:
        """Return the difference with the bearing component wrapped."""
        difference = np.asarray(observed, dtype=np.float64) - np.asarray(
            predicted, dtype=np.float64
        )
        return _wrap_angles(difference, self.angle_indices)
