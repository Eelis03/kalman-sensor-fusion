"""The unscented Kalman filter.

The unscented filter replaces the Jacobian linearisation of the extended filter
with the scaled unscented transform: a deterministic sigma point set is pushed
through the exact nonlinearity and the posterior moments are recovered by
weighted sums. No derivative of the model is ever taken, so a model with a
discontinuous or unavailable Jacobian is usable, and the propagated moments are
accurate to third order for a Gaussian prior instead of first order.

Process noise is treated as additive: it is added to the predicted covariance
rather than carried in an augmented sigma point set. That is the standard choice
when the noise enters the dynamics additively, and it keeps the sigma point set
at ``2 * n + 1`` rather than ``2 * (n + q) + 1`` points. The cost is stated in
``docs/design-notes.md``.

Reference
---------
Wan and van der Merwe, "The unscented Kalman filter for nonlinear estimation",
Proceedings of the IEEE Adaptive Systems for Signal Processing, Communications,
and Control Symposium, 2000, pages 153 to 158. DOI 10.1109/ASSPCC.2000.882463.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.algorithm.base import GaussianState, UpdateResult, symmetrize
from sensor_fusion.algorithm.sigma import (
    ScaledUnscentedSpec,
    sigma_points,
    unscented_covariance,
    unscented_mean,
    unscented_residuals,
)
from sensor_fusion.model.measurement import MeasurementModel
from sensor_fusion.model.motion import MotionModel

__all__ = ["UnscentedKalmanFilter"]


@dataclass(frozen=True, slots=True)
class UnscentedKalmanFilter:
    """Sigma point Kalman filter using the scaled unscented transform."""

    motion_model: MotionModel
    spec: ScaledUnscentedSpec = field(default_factory=ScaledUnscentedSpec)

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""
        return "ukf"

    @property
    def motion(self) -> MotionModel:
        """The motion model this estimator propagates with."""
        return self.motion_model

    def predict(self, state: GaussianState, dt: float) -> GaussianState:
        """Advance the belief by ``dt`` seconds through the unscented transform."""
        angles = self.motion_model.angle_indices
        mean_weights, cov_weights = self.spec.weights(state.dim)
        points = sigma_points(state.mean, state.cov, self.spec)
        propagated = np.stack([self.motion_model.predict(point, dt) for point in points])

        mean = unscented_mean(propagated, mean_weights, angles)
        residuals = unscented_residuals(propagated, mean, angles)
        cov = unscented_covariance(residuals, residuals, cov_weights)
        cov = cov + self.motion_model.process_noise(state.mean, dt)
        return GaussianState(mean=self.motion_model.normalize(mean), cov=symmetrize(cov))

    def update(
        self, state: GaussianState, observation: FloatArray, sensor: MeasurementModel
    ) -> UpdateResult:
        """Fold one observation from ``sensor`` into the belief.

        Both the measurement mean and every residual are formed with angle
        awareness. The bearing of a radar return is meaningless as a plain
        arithmetic average once the sigma point set straddles the plus or minus
        pi boundary.
        """
        state_angles = self.motion_model.angle_indices
        sensor_angles = sensor.angle_indices
        mean_weights, cov_weights = self.spec.weights(state.dim)

        points = sigma_points(state.mean, state.cov, self.spec)
        observed = np.stack(
            [sensor.predict(self.motion_model.to_cartesian(point)) for point in points]
        )

        predicted = unscented_mean(observed, mean_weights, sensor_angles)
        observation_residuals = unscented_residuals(observed, predicted, sensor_angles)
        innovation_cov = symmetrize(
            unscented_covariance(observation_residuals, observation_residuals, cov_weights)
            + sensor.noise_cov
        )
        state_residuals = unscented_residuals(points, state.mean, state_angles)
        cross_cov = unscented_covariance(state_residuals, observation_residuals, cov_weights)

        gain = np.asarray(np.linalg.solve(innovation_cov, cross_cov.T).T, dtype=np.float64)
        innovation = sensor.residual(observation, predicted)
        mean = self.motion_model.normalize(state.mean + gain @ innovation)
        cov = symmetrize(state.cov - gain @ innovation_cov @ gain.T)
        nis = float(innovation @ np.linalg.solve(innovation_cov, innovation))
        return UpdateResult(
            state=GaussianState(mean=mean, cov=cov),
            innovation=innovation,
            innovation_cov=innovation_cov,
            nis=nis,
        )
