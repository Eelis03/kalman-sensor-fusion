"""The scaled unscented transform.

The transform propagates a deterministic set of ``2 * n + 1`` sigma points
through the nonlinearity and recovers the mean and covariance from weighted
sums. For a linear function it reproduces the exact mean and covariance, and for
a nonlinear one it is accurate to third order for Gaussian inputs, against the
first order of a Jacobian linearisation.

Parameter choice
----------------
The scaling is ``lambda = alpha**2 * (n + kappa) - n`` from Julier's scaled
unscented transform. This package defaults to ``alpha = 1.0``, ``kappa = 0.0``,
``beta = 2.0``, which is a deliberate choice and not the ``alpha = 1e-3`` often
quoted:

* ``alpha = 1``, ``kappa = 0`` gives ``lambda = 0``, so the mean weight of the
  centre point is zero and every remaining weight is ``1 / (2 * n)``. All mean
  weights and all covariance weights are then non-negative, and the recovered
  covariance is a non-negative combination of outer products, hence positive
  semi-definite by construction. Formulations with a negative centre weight can
  return an indefinite covariance.
* ``beta = 2`` is optimal for a Gaussian prior, per Wan and van der Merwe.
* ``alpha = 1e-3`` places the sigma points within a thousandth of a standard
  deviation of the mean. The covariance is then recovered by dividing squared
  differences of nearly equal numbers by a weight of the order of ``1e6``, and
  in double precision that cancellation costs roughly six significant digits.
  With ``lambda = 0`` the spread is one standard deviation times ``sqrt(n)`` and
  no cancellation occurs, which is what allows the unscented filter in this
  package to agree with the exact Kalman filter on a linear problem to a
  tolerance near machine precision.

References
----------
Julier, "The scaled unscented transformation", Proceedings of the 2002 American
Control Conference, 2002, pages 4555 to 4559. DOI 10.1109/ACC.2002.1025369.

Wan and van der Merwe, "The unscented Kalman filter for nonlinear estimation",
Proceedings of the IEEE Adaptive Systems for Signal Processing, Communications,
and Control Symposium, 2000, pages 153 to 158. DOI 10.1109/ASSPCC.2000.882463.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.algorithm.base import safe_cholesky
from sensor_fusion.model.angles import wrap_to_pi

__all__ = [
    "ScaledUnscentedSpec",
    "sigma_points",
    "unscented_covariance",
    "unscented_mean",
    "unscented_residuals",
]


@dataclass(frozen=True, slots=True)
class ScaledUnscentedSpec:
    """Parameters of the scaled unscented transform.

    ``alpha`` sets the spread of the sigma points, ``beta`` folds prior knowledge
    of the distribution into the covariance weight of the centre point, and
    ``kappa`` is a secondary scaling. The fields are declared in that order, so
    positional construction is ``ScaledUnscentedSpec(alpha, beta, kappa)``. See
    the module docstring for why the defaults are what they are.
    """

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")

    def lam(self, dim: int) -> float:
        """Return the scaling parameter ``lambda`` for a state of size ``dim``."""
        return self.alpha**2 * (dim + self.kappa) - dim

    def spread(self, dim: int) -> float:
        """Return ``sqrt(dim + lambda)``, the factor applied to the covariance root."""
        total = dim + self.lam(dim)
        if total <= 0.0:
            raise ValueError("dim + lambda must be positive; adjust alpha or kappa")
        return float(np.sqrt(total))

    def weights(self, dim: int) -> tuple[FloatArray, FloatArray]:
        """Return the mean and covariance weights, each of length ``2 * dim + 1``.

        The mean weights sum to one exactly. The covariance weights sum to
        ``1 + (1 - alpha**2 + beta)``; they are not a probability distribution
        and are not expected to sum to one.
        """
        lam = self.lam(dim)
        total = dim + lam
        common = 0.5 / total
        mean_weights = np.full(2 * dim + 1, common, dtype=np.float64)
        mean_weights[0] = lam / total
        cov_weights = np.array(mean_weights, dtype=np.float64, copy=True)
        cov_weights[0] = mean_weights[0] + (1.0 - self.alpha**2 + self.beta)
        return mean_weights, cov_weights


def sigma_points(mean: FloatArray, cov: FloatArray, spec: ScaledUnscentedSpec) -> FloatArray:
    """Return the ``(2 * n + 1, n)`` sigma point set for a Gaussian belief.

    Row zero is the mean. Rows ``1 .. n`` are the mean displaced along the
    columns of the scaled matrix square root, and rows ``n + 1 .. 2 * n`` are
    the mirror displacements, so the set is symmetric about the mean and
    reproduces it exactly under the mean weights.
    """
    centre = np.asarray(mean, dtype=np.float64)
    dim = centre.size
    root = safe_cholesky(np.asarray(cov, dtype=np.float64)) * spec.spread(dim)
    points = np.empty((2 * dim + 1, dim), dtype=np.float64)
    points[0] = centre
    points[1 : dim + 1] = centre + root.T
    points[dim + 1 :] = centre - root.T
    return points


def unscented_mean(
    points: FloatArray, weights: FloatArray, angle_indices: tuple[int, ...] = ()
) -> FloatArray:
    """Return the weighted mean of ``points``, handling angular components.

    Angular components are first unwrapped relative to the centre sigma point,
    then averaged linearly, then wrapped back. Averaging angles directly is
    wrong whenever the set straddles the plus or minus pi boundary: the mean of
    ``+3.14`` and ``-3.14`` is zero, which points the opposite way. Unwrapping
    relative to a reference is used in preference to the circular mean of sines
    and cosines because it stays well defined when some weights are negative,
    which they are for other choices of ``alpha`` and ``kappa``.
    """
    array = np.asarray(points, dtype=np.float64)
    working = np.array(array, dtype=np.float64, copy=True)
    for index in angle_indices:
        reference = float(array[0, index])
        working[:, index] = reference + wrap_to_pi(array[:, index] - reference)
    mean = np.asarray(np.asarray(weights, dtype=np.float64) @ working, dtype=np.float64)
    for index in angle_indices:
        mean[index] = float(wrap_to_pi(np.asarray([mean[index]]))[0])
    return mean


def unscented_residuals(
    points: FloatArray, mean: FloatArray, angle_indices: tuple[int, ...] = ()
) -> FloatArray:
    """Return ``points - mean`` row-wise with angular components wrapped."""
    array = np.asarray(points, dtype=np.float64)
    residuals = array - np.asarray(mean, dtype=np.float64)
    for index in angle_indices:
        residuals[:, index] = wrap_to_pi(residuals[:, index])
    return np.asarray(residuals, dtype=np.float64)


def unscented_covariance(
    left: FloatArray, right: FloatArray, weights: FloatArray
) -> FloatArray:
    """Return the weighted sum of outer products of matching residual rows."""
    weighted = np.asarray(weights, dtype=np.float64)[:, None] * np.asarray(
        left, dtype=np.float64
    )
    return np.asarray(weighted.T @ np.asarray(right, dtype=np.float64), dtype=np.float64)
