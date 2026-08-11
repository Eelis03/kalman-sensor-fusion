"""Innovation whiteness testing.

The normalised innovation squared asks whether the covariance the filter claims
is the right size. It says nothing about whether one innovation is related to
the next, and those are separate questions. An innovation is the part of a
measurement the filter could not predict, so under a correct model the sequence
is white: consecutive innovations are uncorrelated. Where they are not, the
filter is leaving structure on the table that a better model would have removed,
and it can do that while its magnitude test passes, because a residual can be
correctly sized and still be predictable.

The statistic is the time-average normalised autocorrelation of Bar-Shalom, Li,
and Kirubarajan (2001) section 5.4, at lag ``l``,

``rho(l) = sum_k nu(k).T nu(k + l) / sqrt(sum_k nu(k).T nu(k) * sum_k nu(k + l).T nu(k + l))``

summed over every run as well as every step, since the runs are independent
realisations of the same sequence.

The innovations fed in are normalised to unit covariance first, which is what
:attr:`~sensor_fusion.pipeline.trace.StepRecord.normalized_innovation` records.
Raw innovations would leave the inner product above adding square metres to
square radians for the radar, and the weight each component then carried would
be an artefact of the units the sensor was written in. Once normalised, each
component contributes unit variance under the null, so a correlation formed from
``runs * (steps - l) * dim`` scalar products has standard error
``1 / sqrt(runs * (steps - l) * dim)``.

References
----------
Bar-Shalom, Li, and Kirubarajan, *Estimation with Applications to Tracking and
Navigation*, Wiley, 2001, section 5.4, "Consistency of state estimators".
DOI 10.1002/0471221279.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from scipy.stats import chi2

from sensor_fusion._types import FloatArray

__all__ = ["DEFAULT_LAGS", "Whiteness", "WhitenessReport", "whiteness_report"]

# Lags tested unless the caller asks for others. Lag one is where a motion model
# that cannot follow the target shows up first, because the manoeuvre it omits
# persists from one update to the next. Two more are carried so that a slowly
# decaying correlation is visible as a decay rather than as one unlucky sample.
DEFAULT_LAGS: Final[tuple[int, ...]] = (1, 2, 3)


class Whiteness(StrEnum):
    """The outcome of an innovation whiteness test.

    ``CORRELATED`` means at least one lag exceeded its bound. The usual cause is
    a motion model that does not match the target and the usual sign is
    positive, since the part of the motion the model omits does not change
    between one update and the next. A negative correlation at lag one is the
    opposite complaint, an update that overcorrects and is pulled back at the
    following step, which is what an over-large gain looks like.
    """

    WHITE = "white"
    CORRELATED = "correlated"


@dataclass(frozen=True, slots=True)
class WhitenessReport:
    """The evidence behind one whiteness verdict."""

    statistic: str
    runs: int
    steps: int
    dim: int
    lags: tuple[int, ...]
    correlations: tuple[float, ...]
    bounds: tuple[float, ...]
    verdict: Whiteness

    def summary(self) -> str:
        """Return a single line stating the verdict and the numbers behind it.

        The lag reported is the one furthest out relative to its own bound, not
        the one with the largest correlation. The bounds widen slightly with the
        lag, because a longer lag leaves fewer pairs to average.
        """
        ratios = [
            abs(value) / bound for value, bound in zip(self.correlations, self.bounds, strict=True)
        ]
        index = int(np.argmax(ratios))
        return (
            f"whiteness ({self.statistic}): correlation {self.correlations[index]:+.4f} "
            f"at lag {self.lags[index]} against a bound of {self.bounds[index]:.4f} "
            f"over {self.runs} runs of {self.steps} updates, "
            f"verdict {self.verdict.value}"
        )


def whiteness_report(
    statistic: str,
    innovations: FloatArray,
    lags: Sequence[int] = DEFAULT_LAGS,
    confidence: float = 0.95,
) -> WhitenessReport:
    """Test a ``(runs, steps, dim)`` stack of normalised innovations for whiteness.

    ``innovations`` must already be scaled to unit covariance; see the module
    docstring for why raw innovations are not usable here. The stack produced by
    :func:`~sensor_fusion.pipeline.montecarlo.run_monte_carlo` under
    ``normalized_innovations`` is in exactly this form.

    Every lag is read against the same confidence, so each bound carries a
    Bonferroni correction for their number: three lags at 95 percent each would
    reject a genuinely white sequence about one time in seven rather than one in
    twenty. The lags are not independent of one another, so the correction is
    conservative, which is the direction the intervals in
    :mod:`sensor_fusion.analysis.consistency` also err in.
    """
    array = np.asarray(innovations, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("innovations must have shape (runs, steps, dim)")
    runs, steps, dim = array.shape
    if not lags:
        raise ValueError("at least one lag is required")
    if min(lags) < 1 or max(lags) >= steps:
        raise ValueError("every lag must lie between one and the step count minus one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    # The two-sided standard normal quantile, taken as the square root of the
    # chi-square quantile with one degree of freedom, since the square of a
    # standard normal is exactly that variable. Reusing chi2 keeps this package's
    # dependency on scipy.stats at the one function it already needs.
    per_lag = 1.0 - (1.0 - confidence) / len(lags)
    quantile = math.sqrt(float(chi2.ppf(per_lag, 1)))

    correlations: list[float] = []
    bounds: list[float] = []
    for lag in lags:
        head = array[:, : steps - lag, :]
        tail = array[:, lag:, :]
        scale = math.sqrt(float(np.sum(head * head)) * float(np.sum(tail * tail)))
        if scale == 0.0:
            raise ValueError("innovations are identically zero, so no correlation is defined")
        correlations.append(float(np.sum(head * tail)) / scale)
        bounds.append(quantile / math.sqrt(runs * (steps - lag) * dim))

    verdict = (
        Whiteness.CORRELATED
        if any(abs(value) > bound for value, bound in zip(correlations, bounds, strict=True))
        else Whiteness.WHITE
    )
    return WhitenessReport(
        statistic=statistic,
        runs=runs,
        steps=steps,
        dim=dim,
        lags=tuple(lags),
        correlations=tuple(correlations),
        bounds=tuple(bounds),
        verdict=verdict,
    )
