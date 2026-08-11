"""The linear Kalman filter, used as an exact reference.

When both the motion model and the sensor are linear, this recursion is the
exact minimum mean square error estimator and there is nothing to approximate.
It exists here so that the extended and unscented filters can be checked against
a known-correct answer rather than against each other.

References
----------
Kalman, "A new approach to linear filtering and prediction problems", Journal of
Basic Engineering 82(1), 1960, pages 35 to 45. DOI 10.1115/1.3662552.

Bucy and Joseph, *Filtering for Stochastic Processes with Applications to
Guidance*, Interscience, 1968, for the stabilised covariance update named after
Joseph and used below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.algorithm.base import GaussianState, UpdateResult, symmetrize
from sensor_fusion.model.measurement import MeasurementModel
from sensor_fusion.model.motion import MotionModel

__all__ = ["KalmanFilter"]


@dataclass(frozen=True, slots=True)
class KalmanFilter:
    """Exact Kalman filter for a linear motion model and linear sensors."""

    motion_model: MotionModel

    def __post_init__(self) -> None:
        if not self.motion_model.is_linear:
            raise ValueError(
                f"{self.motion_model.name} is nonlinear; use ExtendedKalmanFilter "
                "or UnscentedKalmanFilter"
            )

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""
        return "kf"

    @property
    def motion(self) -> MotionModel:
        """The motion model this estimator propagates with."""
        return self.motion_model

    def predict(self, state: GaussianState, dt: float) -> GaussianState:
        """Advance the belief by ``dt`` seconds."""
        transition = self.motion_model.jacobian(state.mean, dt)
        mean = self.motion_model.predict(state.mean, dt)
        cov = transition @ state.cov @ transition.T + self.motion_model.process_noise(
            state.mean, dt
        )
        return GaussianState(mean=mean, cov=symmetrize(cov))

    def update(
        self, state: GaussianState, observation: FloatArray, sensor: MeasurementModel
    ) -> UpdateResult:
        """Fold one observation from ``sensor`` into the belief."""
        if not sensor.is_linear:
            raise ValueError(f"{sensor.name} is nonlinear; the linear Kalman filter cannot use it")

        cartesian = self.motion_model.to_cartesian(state.mean)
        observation_matrix = sensor.jacobian(cartesian) @ self.motion_model.cartesian_jacobian(
            state.mean
        )
        innovation = sensor.residual(observation, sensor.predict(cartesian))
        innovation_cov = symmetrize(
            observation_matrix @ state.cov @ observation_matrix.T + sensor.noise_cov
        )

        gain = np.asarray(
            np.linalg.solve(innovation_cov, observation_matrix @ state.cov).T, dtype=np.float64
        )
        mean = state.mean + gain @ innovation
        identity = np.eye(state.dim, dtype=np.float64)
        closed_loop = identity - gain @ observation_matrix
        # Joseph form (Bucy and Joseph, 1968). It costs one extra pair of matrix
        # products and stays symmetric positive semi-definite even when the gain
        # is not exactly the optimal one, which the simpler (I - K H) P form does
        # not. At the optimal gain the two agree algebraically.
        cov = symmetrize(closed_loop @ state.cov @ closed_loop.T + gain @ sensor.noise_cov @ gain.T)
        nis = float(innovation @ np.linalg.solve(innovation_cov, innovation))
        return UpdateResult(
            state=GaussianState(mean=mean, cov=cov),
            innovation=innovation,
            innovation_cov=innovation_cov,
            nis=nis,
        )
