"""Angle wrapping.

Every quantity in this package that lives on the circle, the CTRV heading and
the radar bearing, is wrapped into ``[-pi, pi)`` before it is averaged,
differenced, or fed into a covariance. Omitting that wrap is the classic cause
of silent filter divergence: a target crossing the plus or minus pi boundary
produces an innovation near ``2 * pi`` instead of near zero, the filter reads it
as a very large error, and the gain drives the estimate away from the truth.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from sensor_fusion._types import FloatArray

__all__ = ["TWO_PI", "wrap_scalar_to_pi", "wrap_to_pi"]

TWO_PI: Final[float] = 2.0 * float(np.pi)


def wrap_to_pi(angle: FloatArray) -> FloatArray:
    """Wrap every element of ``angle`` into the half-open interval ``[-pi, pi)``.

    The half-open convention places exactly ``pi`` at ``-pi``. That choice is
    arbitrary but must be consistent, because an implementation that maps some
    inputs to ``+pi`` and others to ``-pi`` breaks the idempotence property the
    filters rely on.
    """
    return np.asarray(np.mod(np.asarray(angle, dtype=np.float64) + np.pi, TWO_PI) - np.pi)


def wrap_scalar_to_pi(angle: float) -> float:
    """Wrap a single angle into ``[-pi, pi)``.

    A separate scalar entry point exists because the array form returns a
    zero-dimensional array for a scalar input, and unpacking that at every call
    site is noisier than having two functions.
    """
    return float(np.mod(angle + np.pi, TWO_PI) - np.pi)
