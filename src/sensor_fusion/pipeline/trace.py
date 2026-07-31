"""Structured records produced by a filter run.

Nothing in this module computes a metric or draws a figure. It defines what a
run records so that the analysis layer has one stable shape to work against, and
so that a regression test can pin quantities without reaching into a filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sensor_fusion._types import FloatArray
from sensor_fusion.model.measurement import MeasurementModel

__all__ = ["FilterTrace", "Measurement", "OutOfOrderPolicy", "StepRecord"]


class OutOfOrderPolicy(StrEnum):
    """How a run treats a measurement that arrives after a later one.

    Sensors on a real vehicle report over buses with different, variable
    latencies. A radar return stamped at ``t`` can reach the fusion node after a
    lidar return stamped at ``t + 0.05``, and a filter that simply applies
    measurements in arrival order would be asked to propagate backwards in time.

    ``BUFFER`` holds arrivals in a reorder buffer of fixed depth and releases
    them in timestamp order once the buffer's latency budget has elapsed. This
    restores the correct order for every measurement whose latency is within
    budget, at the cost of that budget in output delay.

    ``DISCARD`` applies measurements in arrival order and drops any whose
    timestamp precedes the filter's current time. It adds no delay and loses
    information.

    Neither is optimal. The optimal treatment retrodicts the filter state back
    to the measurement time and folds the late measurement in there. That is the
    out-of-sequence measurement problem, and it is not implemented here; see
    ``docs/design-notes.md``.
    """

    BUFFER = "buffer"
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class Measurement:
    """One sensor report with both its timestamp and its arrival time.

    ``time`` is when the observation was taken and is what the filter must
    propagate to. ``arrival_time`` is when the fusion node received it and is
    what determines processing order before any reordering is applied.
    """

    time: float
    arrival_time: float
    sensor: MeasurementModel
    value: FloatArray
    truth_index: int

    @property
    def sensor_name(self) -> str:
        """Name of the sensor that produced this report."""
        return self.sensor.name


@dataclass(frozen=True, slots=True)
class StepRecord:
    """The filter state and consistency statistics after one measurement update.

    ``nees_state`` is the normalised estimation error squared in the filter's own
    state space, and is ``nan`` unless the filter's motion model compares equal
    to the model that generated the truth, process noise parameters included.
    ``nees_cartesian`` is always defined and is computed on the Cartesian view,
    using the first-order projection of the covariance; for a nonlinear motion
    model that projection is an approximation, which is why the native statistic
    is kept separate.
    """

    time: float
    sensor_name: str
    mean: FloatArray
    cov: FloatArray
    innovation: FloatArray
    nis: float
    nis_dof: int
    truth_cartesian: FloatArray
    estimate_cartesian: FloatArray
    nees_cartesian: float
    nees_state: float


@dataclass(frozen=True, slots=True)
class FilterTrace:
    """Everything one filter run produced, plus its measurement bookkeeping."""

    filter_name: str
    motion_name: str
    policy: OutOfOrderPolicy
    records: tuple[StepRecord, ...]
    processed: int
    discarded: int
    out_of_order_arrivals: int

    @property
    def times(self) -> FloatArray:
        """Measurement timestamps of every processed update."""
        return np.array([record.time for record in self.records], dtype=np.float64)

    @property
    def position_error(self) -> FloatArray:
        """Per-update Cartesian position error vector, shape ``(n, 2)``."""
        return np.array(
            [record.truth_cartesian[:2] - record.estimate_cartesian[:2] for record in self.records],
            dtype=np.float64,
        ).reshape(len(self.records), 2)

    @property
    def velocity_error(self) -> FloatArray:
        """Per-update Cartesian velocity error vector, shape ``(n, 2)``."""
        return np.array(
            [record.truth_cartesian[2:] - record.estimate_cartesian[2:] for record in self.records],
            dtype=np.float64,
        ).reshape(len(self.records), 2)

    @property
    def nees_cartesian(self) -> FloatArray:
        """Per-update Cartesian normalised estimation error squared."""
        return np.array([record.nees_cartesian for record in self.records], dtype=np.float64)

    @property
    def nees_state(self) -> FloatArray:
        """Per-update native-state NEES, ``nan`` where it is undefined."""
        return np.array([record.nees_state for record in self.records], dtype=np.float64)

    def nis(self, sensor_name: str) -> FloatArray:
        """Per-update normalised innovation squared for one sensor."""
        return np.array(
            [record.nis for record in self.records if record.sensor_name == sensor_name],
            dtype=np.float64,
        )

    def nis_times(self, sensor_name: str) -> FloatArray:
        """Timestamps matching :meth:`nis` for one sensor."""
        return np.array(
            [record.time for record in self.records if record.sensor_name == sensor_name],
            dtype=np.float64,
        )

    @property
    def sensor_names(self) -> tuple[str, ...]:
        """Distinct sensor names in the order first seen."""
        seen: list[str] = []
        for record in self.records:
            if record.sensor_name not in seen:
                seen.append(record.sensor_name)
        return tuple(seen)
