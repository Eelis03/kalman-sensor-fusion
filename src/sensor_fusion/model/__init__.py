"""Pure motion and measurement models.

This layer performs no input or output, holds no mutable state, and knows
nothing about filters. Everything here is a deterministic function of a state
vector, a time step, and the model's own parameters.
"""

from __future__ import annotations

from sensor_fusion.model.angles import TWO_PI, wrap_scalar_to_pi, wrap_to_pi
from sensor_fusion.model.measurement import (
    RADAR_BEARING,
    RADAR_RANGE,
    RADAR_RANGE_RATE,
    Lidar,
    MeasurementModel,
    Radar,
)
from sensor_fusion.model.motion import (
    CTRV_HEADING,
    CTRV_POSITION_X,
    CTRV_POSITION_Y,
    CTRV_SPEED,
    CTRV_YAW_RATE,
    ConstantAcceleration,
    ConstantTurnRate,
    ConstantVelocity,
    MotionModel,
)

__all__ = [
    "CTRV_HEADING",
    "CTRV_POSITION_X",
    "CTRV_POSITION_Y",
    "CTRV_SPEED",
    "CTRV_YAW_RATE",
    "RADAR_BEARING",
    "RADAR_RANGE",
    "RADAR_RANGE_RATE",
    "TWO_PI",
    "ConstantAcceleration",
    "ConstantTurnRate",
    "ConstantVelocity",
    "Lidar",
    "MeasurementModel",
    "MotionModel",
    "Radar",
    "wrap_scalar_to_pi",
    "wrap_to_pi",
]
