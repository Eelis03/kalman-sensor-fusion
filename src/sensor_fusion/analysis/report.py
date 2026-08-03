"""Turning stacked Monte Carlo statistics into consistency verdicts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensor_fusion.analysis.consistency import ConsistencyReport, consistency_report
from sensor_fusion.pipeline.montecarlo import MonteCarloResult

__all__ = ["DIVERGENCE_FACTOR", "FilterAssessment", "assess"]

# Cartesian position and velocity, the space every NEES comparison is made in
# when the filter's motion model is not the model that generated the truth.
_CARTESIAN_DOF = 4

# A run whose time-averaged NEES exceeds this multiple of its degrees of freedom
# is counted as a lost track rather than an inconsistent one. The distinction
# matters because the mean over runs is not robust: in radar-only tracking a
# single run in forty that loses the target can raise the mean NEES by two
# orders of magnitude while the other thirty-nine are unremarkable, and reading
# that mean as a statement about the typical run is simply wrong.
DIVERGENCE_FACTOR = 10.0


@dataclass(frozen=True, slots=True)
class FilterAssessment:
    """Consistency verdicts and accuracy for one Monte Carlo campaign."""

    filter_name: str
    motion_name: str
    runs: int
    nees: ConsistencyReport
    nis: tuple[ConsistencyReport, ...]
    position_rmse: float
    velocity_rmse: float
    median_run_nees: float
    diverged_runs: int

    def lines(self) -> tuple[str, ...]:
        """Return the assessment as printable lines."""
        head = (
            f"{self.filter_name} with {self.motion_name} over {self.runs} runs: "
            f"position RMSE {self.position_rmse:.4f} m, "
            f"velocity RMSE {self.velocity_rmse:.4f} m/s"
        )
        robust = (
            f"median run NEES {self.median_run_nees:.3f} against expected {self.nees.dof}, "
            f"{self.diverged_runs} of {self.runs} runs lost the track"
        )
        return (head, self.nees.summary(), robust, *(report.summary() for report in self.nis))


def assess(result: MonteCarloResult, confidence: float = 0.95) -> FilterAssessment:
    """Build the consistency assessment for one Monte Carlo result.

    The NEES statistic is taken in the filter's own state space when the filter's
    motion model generated the truth, because the chi-square assumptions hold
    there without any projection at all. When the models differ there is no
    common state space and the Cartesian view is used instead, with its second
    moment taken by the sigma point projection in
    :func:`sensor_fusion.pipeline.fusion.cartesian_moment`. The statistic is
    labelled with the space it was taken in, because the degrees of freedom
    differ between the two.
    """
    if result.native_nees_available:
        nees_samples = result.nees_state
        nees = consistency_report(
            f"NEES (state, {result.motion_name})", nees_samples, result.state_dim, confidence
        )
    else:
        nees_samples = result.nees_cartesian
        nees = consistency_report(
            "NEES (Cartesian projection)", nees_samples, _CARTESIAN_DOF, confidence
        )

    per_run = nees_samples.mean(axis=1)
    diverged = int(np.count_nonzero(per_run > DIVERGENCE_FACTOR * nees.dof))

    reports = tuple(
        consistency_report(f"NIS ({name})", nis_samples, result.nis_dof[name], confidence)
        for name, nis_samples in sorted(result.nis.items())
    )
    return FilterAssessment(
        filter_name=result.filter_name,
        motion_name=result.motion_name,
        runs=result.runs,
        nees=nees,
        nis=reports,
        position_rmse=result.mean_position_rmse,
        velocity_rmse=result.mean_velocity_rmse,
        median_run_nees=float(np.median(per_run)),
        diverged_runs=diverged,
    )
