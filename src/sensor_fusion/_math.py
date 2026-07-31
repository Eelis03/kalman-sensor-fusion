"""Small numerical helpers shared by layers that must not import each other.

The pipeline layer may not import the analysis layer, since the analysis layer
reads what the pipeline produces. A helper needed by both therefore cannot live
in either, and duplicating it in both is how two copies drift apart.
"""

from __future__ import annotations

import numpy as np

from sensor_fusion._types import FloatArray

__all__ = ["rmse"]


def rmse(errors: FloatArray) -> float:
    """Return the root mean square norm of a stack of error vectors.

    For an ``(n, 2)`` input this is ``sqrt(mean(ex**2 + ey**2))``, the root mean
    square of the Euclidean error magnitude, not the per-axis value. An empty
    stack returns ``nan``, because a run with no updates has no error and
    reporting zero would claim a perfect result.
    """
    array = np.asarray(errors, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(array * array, axis=1))))
