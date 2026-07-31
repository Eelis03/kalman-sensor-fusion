"""Scenario generation, filter driving, and Monte Carlo repetition.

This layer owns randomness and time. It knows how to build ground truth, when
each sensor reports, what to do with a late report, and how to repeat all of
that under independent noise. It does not compute metrics and it does not plot.
"""

from __future__ import annotations

from sensor_fusion.pipeline.fusion import (
    FusionSettings,
    InitialUncertainty,
    matched_belief,
    run_filter,
)
from sensor_fusion.pipeline.montecarlo import MonteCarloResult, RunBuilder, run_monte_carlo
from sensor_fusion.pipeline.scenarios import (
    boundary_crossing_target,
    straight_target,
    turning_target,
    with_latency,
)
from sensor_fusion.pipeline.simulator import (
    Scenario,
    ScenarioConfig,
    SensorSchedule,
    simulate,
)
from sensor_fusion.pipeline.trace import (
    FilterTrace,
    Measurement,
    OutOfOrderPolicy,
    StepRecord,
)

__all__ = [
    "FilterTrace",
    "FusionSettings",
    "InitialUncertainty",
    "Measurement",
    "MonteCarloResult",
    "OutOfOrderPolicy",
    "RunBuilder",
    "Scenario",
    "ScenarioConfig",
    "SensorSchedule",
    "StepRecord",
    "boundary_crossing_target",
    "matched_belief",
    "run_filter",
    "run_monte_carlo",
    "simulate",
    "straight_target",
    "turning_target",
    "with_latency",
]
