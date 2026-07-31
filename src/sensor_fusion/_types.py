"""Shared array type aliases.

Every public function in this package accepts and returns dense float64 arrays.
Keeping the alias in one place lets the type checker enforce that uniformly
without repeating the ``numpy.typing`` import in each module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["FloatArray"]

FloatArray = NDArray[np.float64]
