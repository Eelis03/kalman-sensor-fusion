"""Recursive estimators behind one Protocol.

Nothing in this layer plots, prints, reads a file, or draws a random number. The
three filters differ only in how they push a Gaussian through a nonlinearity,
and all three satisfy :class:`~sensor_fusion.algorithm.base.StateEstimator`, so
callers swap them without a conditional.
"""

from __future__ import annotations

from sensor_fusion.algorithm.base import (
    GaussianState,
    StateEstimator,
    UpdateResult,
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

__all__ = [
    "ExtendedKalmanFilter",
    "GaussianState",
    "KalmanFilter",
    "ScaledUnscentedSpec",
    "StateEstimator",
    "UnscentedKalmanFilter",
    "UpdateResult",
    "is_positive_semidefinite",
    "safe_cholesky",
    "sigma_points",
    "symmetrize",
    "unscented_covariance",
    "unscented_mean",
    "unscented_residuals",
]
