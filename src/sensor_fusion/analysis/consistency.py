"""Filter consistency testing.

A Kalman filter reports both an estimate and the covariance it claims for that
estimate. Root mean square error checks the estimate. Nothing checks the claimed
covariance except a consistency test, and a filter whose covariance is wrong is
dangerous in a way that a merely inaccurate one is not: everything downstream,
gating, data association, and any risk decision taken on the estimate, trusts
that number.

Two statistics are used.

The normalised innovation squared, ``NIS = nu.T @ inv(S) @ nu``, needs no ground
truth and is therefore computable on a real vehicle. Under a correctly specified
filter it is chi-square distributed with ``dim(z)`` degrees of freedom.

The normalised estimation error squared, ``NEES = e.T @ inv(P) @ e``, needs
ground truth and is therefore a simulation-only tool. Under a correctly
specified filter it is chi-square distributed with ``dim(x)`` degrees of
freedom.

Averaging ``N`` independent samples of a chi-square variable with ``d`` degrees
of freedom gives a variable distributed as ``chi2(N * d) / N``, which is the
interval used below. Independence across Monte Carlo runs is exact because each
run uses its own noise realisation. Independence across time is not exact,
because the estimation error is serially correlated, so the interval is applied
to the across-run average at each time step, where it is valid, and the grand
mean over time is then reported against that same interval as a summary.

A summary computed that way errs towards accepting rather than rejecting.
Averaging further over time cannot increase the variance of the statistic beyond
that of a single time step, however strongly correlated the steps are, so the
grand mean is at least as tightly concentrated as the interval assumes. It will
therefore not step outside the interval by chance more often than the nominal
rate, and a filter it classifies as consistent has not been let off by the
approximation.

References
----------
Bar-Shalom, Li, and Kirubarajan, *Estimation with Applications to Tracking and
Navigation*, Wiley, 2001, section 5.4, "Consistency of state estimators".
DOI 10.1002/0471221279.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.stats import chi2

from sensor_fusion._types import FloatArray

__all__ = ["ConsistencyReport", "Verdict", "chi2_interval", "consistency_report"]


class Verdict(StrEnum):
    """The outcome of a consistency test.

    ``OPTIMISTIC`` means the statistic sits above its upper bound: the actual
    errors are larger than the covariance claims, so the filter is overconfident.
    That is the dangerous direction, because a gate built from that covariance
    will reject valid measurements and any decision taken on it will understate
    the risk.

    ``CONSERVATIVE`` means the statistic sits below its lower bound: the filter
    claims more uncertainty than it has. Estimates stay valid but information is
    being wasted, and the filter converges more slowly than it could.
    """

    CONSISTENT = "consistent"
    OPTIMISTIC = "optimistic"
    CONSERVATIVE = "conservative"


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """The evidence behind one consistency verdict."""

    statistic: str
    dof: int
    runs: int
    steps: int
    mean: float
    lower: float
    upper: float
    inside_fraction: float
    above_fraction: float
    below_fraction: float
    verdict: Verdict

    def summary(self) -> str:
        """Return a single line stating the verdict and the numbers behind it."""
        return (
            f"{self.statistic}: mean {self.mean:.3f} against expected {self.dof} "
            f"and the {self.runs}-run interval [{self.lower:.3f}, {self.upper:.3f}], "
            f"{self.inside_fraction * 100:.1f} percent of steps inside "
            f"and {self.above_fraction * 100:.1f} percent above, "
            f"verdict {self.verdict.value}"
        )


def chi2_interval(dof: int, runs: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Return the two-sided interval for the mean of ``runs`` chi-square samples.

    Each sample has ``dof`` degrees of freedom. The sum of ``runs`` such samples
    is chi-square with ``runs * dof`` degrees of freedom, so the interval for the
    mean is the chi-square quantile pair divided by ``runs``.
    """
    if dof < 1 or runs < 1:
        raise ValueError("dof and runs must be at least one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    tail = 0.5 * (1.0 - confidence)
    total = dof * runs
    lower = float(chi2.ppf(tail, total)) / runs
    upper = float(chi2.ppf(1.0 - tail, total)) / runs
    return lower, upper


def consistency_report(
    statistic: str, samples: FloatArray, dof: int, confidence: float = 0.95
) -> ConsistencyReport:
    """Classify a filter from a ``(runs, steps)`` array of chi-square statistics.

    The verdict is taken from the grand mean against the interval for the
    across-run average, and the fraction of individual time steps inside that
    interval is reported alongside as supporting evidence. For a consistent
    filter that fraction should sit near the confidence level.

    ``above_fraction`` is reported next to the verdict rather than left in the
    record, because the two answer different questions. A filter converging from
    a loose prior can carry a covariance that is badly wrong for the first
    fraction of a second and perfectly good for the remaining nine seconds, and
    its grand mean will look the same as that of a filter which is mildly wrong
    throughout. The fraction of steps above the bound separates those two, and
    the shape of the trace over time separates them completely, which is what the
    figures in ``docs/figures`` are for.

    No burn-in is trimmed. A filter that needs a second before its covariance can
    be trusted has a property worth reporting rather than one worth hiding, and
    the verdict should describe the whole run the caller asked for.
    """
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("samples must have shape (runs, steps)")
    runs, steps = array.shape
    if steps == 0:
        raise ValueError("samples must contain at least one step")

    lower, upper = chi2_interval(dof, runs, confidence)
    per_step = np.asarray(array.mean(axis=0), dtype=np.float64)
    grand_mean = float(per_step.mean())
    above = float(np.count_nonzero(per_step > upper)) / steps
    below = float(np.count_nonzero(per_step < lower)) / steps

    if grand_mean > upper:
        verdict = Verdict.OPTIMISTIC
    elif grand_mean < lower:
        verdict = Verdict.CONSERVATIVE
    else:
        verdict = Verdict.CONSISTENT

    return ConsistencyReport(
        statistic=statistic,
        dof=dof,
        runs=runs,
        steps=steps,
        mean=grand_mean,
        lower=lower,
        upper=upper,
        inside_fraction=1.0 - above - below,
        above_fraction=above,
        below_fraction=below,
        verdict=verdict,
    )
