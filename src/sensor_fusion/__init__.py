"""Extended and unscented Kalman filters fusing simulated radar and lidar.

The package is organised in five layers with a one-way dependency:

* ``model``: motion and measurement models, pure functions with no input or
  output and no randomness.
* ``algorithm``: the linear, extended, and unscented Kalman filters behind one
  Protocol, with no plotting and no randomness.
* ``pipeline``: scenario simulation, the asynchronous fusion runner, the Monte
  Carlo harness, and the structured trace.
* ``analysis``: normalised innovation squared, normalised estimation error
  squared, root mean square error, and figures.
* ``examples``: thin wiring scripts with no logic of their own.
"""

from __future__ import annotations

from sensor_fusion.algorithm import (
    ExtendedKalmanFilter,
    GaussianState,
    KalmanFilter,
    ScaledUnscentedSpec,
    StateEstimator,
    UnscentedKalmanFilter,
    UpdateResult,
)
from sensor_fusion.model import (
    ConstantAcceleration,
    ConstantTurnRate,
    ConstantVelocity,
    Lidar,
    MeasurementModel,
    MotionModel,
    Radar,
)

__all__ = [
    "ConstantAcceleration",
    "ConstantTurnRate",
    "ConstantVelocity",
    "ExtendedKalmanFilter",
    "GaussianState",
    "KalmanFilter",
    "Lidar",
    "MeasurementModel",
    "MotionModel",
    "Radar",
    "ScaledUnscentedSpec",
    "StateEstimator",
    "UnscentedKalmanFilter",
    "UpdateResult",
    "__version__",
]

__version__ = "0.1.0"
