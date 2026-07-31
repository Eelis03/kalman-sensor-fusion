"""The extended Kalman filter.

The extended filter propagates the mean through the true nonlinearity and the
covariance through its Jacobian. The Jacobians here are analytic, supplied by
the model layer, and checked against central finite differences in the test
suite. A hand-derived Jacobian with a sign error is silently absorbed by the
gain for a while and then diverges, so that check is not optional.

The measurement Jacobian is assembled by the chain rule,
``d(z)/d(state) = d(z)/d(cartesian) @ d(cartesian)/d(state)``, which lets each
sensor be written once against the Cartesian view.

Reference
---------
Bar-Shalom, Li, and Kirubarajan, *Estimation with Applications to Tracking and
Navigation*, Wiley, 2001, section 10.3. DOI 10.1002/0471221279.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.algorithm.base import GaussianState, UpdateResult, symmetrize
from sensor_fusion.model.measurement import MeasurementModel
from sensor_fusion.model.motion import MotionModel

__all__ = ["ExtendedKalmanFilter"]


@dataclass(frozen=True, slots=True)
class ExtendedKalmanFilter:
    """First-order linearised Kalman filter with analytic Jacobians."""

    motion_model: MotionModel

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""
        return "ekf"

    @property
    def motion(self) -> MotionModel:
        """The motion model this estimator propagates with."""
        return self.motion_model

    def predict(self, state: GaussianState, dt: float) -> GaussianState:
        """Advance the belief by ``dt`` seconds.

        The mean goes through the exact nonlinear propagation; only the
        covariance is linearised.
        """
        transition = self.motion_model.jacobian(state.mean, dt)
        mean = self.motion_model.normalize(self.motion_model.predict(state.mean, dt))
        cov = transition @ state.cov @ transition.T + self.motion_model.process_noise(
            state.mean, dt
        )
        return GaussianState(mean=mean, cov=symmetrize(cov))

    def update(
        self, state: GaussianState, observation: FloatArray, sensor: MeasurementModel
    ) -> UpdateResult:
        """Fold one observation from ``sensor`` into the belief.

        The innovation is formed with ``sensor.residual``, which wraps angular
        components. Using a plain subtraction here is the single most common
        cause of a radar filter that tracks correctly for a while and then
        diverges without any warning sign in the covariance.
        """
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
        mean = self.motion_model.normalize(state.mean + gain @ innovation)
        identity = np.eye(state.dim, dtype=np.float64)
        closed_loop = identity - gain @ observation_matrix
        cov = symmetrize(
            closed_loop @ state.cov @ closed_loop.T + gain @ sensor.noise_cov @ gain.T
        )
        nis = float(innovation @ np.linalg.solve(innovation_cov, innovation))
        return UpdateResult(
            state=GaussianState(mean=mean, cov=cov),
            innovation=innovation,
            innovation_cov=innovation_cov,
            nis=nis,
        )
