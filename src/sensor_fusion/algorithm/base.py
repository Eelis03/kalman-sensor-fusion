"""The estimator Protocol and the numerical helpers every filter shares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.model.measurement import MeasurementModel
from sensor_fusion.model.motion import MotionModel

__all__ = [
    "GaussianState",
    "StateEstimator",
    "UpdateResult",
    "is_positive_semidefinite",
    "safe_cholesky",
    "symmetrize",
]


@dataclass(frozen=True, slots=True)
class GaussianState:
    """A Gaussian belief over the target state."""

    mean: FloatArray
    cov: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        cov = np.asarray(self.cov, dtype=np.float64)
        if mean.ndim != 1:
            raise ValueError("mean must be one-dimensional")
        if cov.shape != (mean.size, mean.size):
            raise ValueError("cov must be square and agree with mean")

    @property
    def dim(self) -> int:
        """Dimension of the state."""
        return int(np.asarray(self.mean).size)


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """The outcome of one measurement update.

    ``nis`` is the normalised innovation squared, ``innovation.T @ inv(S) @
    innovation``. Under a correctly specified filter it is chi-square
    distributed with ``innovation.size`` degrees of freedom, which is the basis
    of the consistency machinery in :mod:`sensor_fusion.analysis.consistency`.
    """

    state: GaussianState
    innovation: FloatArray
    innovation_cov: FloatArray
    nis: float


@runtime_checkable
class StateEstimator(Protocol):
    """A recursive Gaussian estimator.

    The linear Kalman filter, the extended Kalman filter, and the unscented
    Kalman filter all satisfy this Protocol, so the pipeline and the analysis
    layers never branch on which one they were handed.
    """

    @property
    def name(self) -> str:
        """Short identifier used in traces and figures."""

    @property
    def motion(self) -> MotionModel:
        """The motion model this estimator propagates with."""

    def predict(self, state: GaussianState, dt: float) -> GaussianState:
        """Advance the belief by ``dt`` seconds."""

    def update(
        self, state: GaussianState, observation: FloatArray, sensor: MeasurementModel
    ) -> UpdateResult:
        """Fold one observation from ``sensor`` into the belief."""


def symmetrize(matrix: FloatArray) -> FloatArray:
    """Return the symmetric part of ``matrix``.

    Covariance recursions are symmetric in exact arithmetic but accumulate an
    antisymmetric component of the order of the rounding error. Left alone that
    component eventually makes a Cholesky factorisation fail, so it is removed
    after every step.
    """
    array = np.asarray(matrix, dtype=np.float64)
    return np.asarray(0.5 * (array + array.T))


def safe_cholesky(matrix: FloatArray, *, jitter: float = 1e-12) -> FloatArray:
    """Return a lower triangular ``L`` with ``L @ L.T`` close to ``matrix``.

    A covariance that is positive definite in exact arithmetic can fail
    ``numpy.linalg.cholesky`` by a few units in the last place. Rather than let
    the filter crash, a diagonal jitter is added and increased by a factor of ten
    per attempt until the factorisation succeeds. The jitter is scaled by the
    mean diagonal entry of the matrix, floored at one, so it stays proportional
    to the magnitude of the covariance rather than to an arbitrary absolute
    value. After eight attempts the jitter has reached ``1e-4`` times that scale,
    which is no longer negligible, so the failure is raised instead of hidden.
    """
    array = symmetrize(matrix)
    scale = max(abs(float(np.trace(array))) / max(array.shape[0], 1), 1.0)
    identity = np.eye(array.shape[0], dtype=np.float64)
    for attempt in range(8):
        offset = 0.0 if attempt == 0 else jitter * scale * (10.0**attempt)
        try:
            factor = np.linalg.cholesky(array + offset * identity)
        except np.linalg.LinAlgError:
            continue
        return np.asarray(factor, dtype=np.float64)
    raise np.linalg.LinAlgError("covariance is not positive definite within tolerance")


def is_positive_semidefinite(matrix: FloatArray, *, tolerance: float = 1e-9) -> bool:
    """True when ``matrix`` is symmetric with no eigenvalue below ``-tolerance``.

    Both the symmetry check and the eigenvalue check are scaled by the largest
    absolute entry of the matrix, floored at one. An absolute threshold would be
    meaningless for a covariance whose entries span several orders of magnitude,
    which they do here: position variance is of the order of a hundredth of a
    square metre while yaw rate variance is smaller again.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return False
    asymmetry = float(np.max(np.abs(array - array.T)))
    magnitude = max(float(np.max(np.abs(array))), 1.0)
    if asymmetry > tolerance * magnitude:
        return False
    eigenvalues = np.linalg.eigvalsh(symmetrize(array))
    return bool(np.min(eigenvalues) >= -tolerance * magnitude)
