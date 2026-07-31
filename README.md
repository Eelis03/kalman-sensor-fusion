# Kalman Sensor Fusion

Extended and unscented Kalman filters fusing simulated radar and lidar with NIS consistency validation.

[![CI](https://github.com/Eelis03/kalman-sensor-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/kalman-sensor-fusion/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This package tracks a manoeuvring target from two asynchronous sensors, a lidar
reporting Cartesian position and a radar reporting range, bearing, and range
rate, using a linear Kalman filter, an extended Kalman filter, or an unscented
Kalman filter behind a single Protocol. It ships the machinery to decide whether
a filter's reported covariance can be believed: normalised innovation squared
and normalised estimation error squared with chi-square bounds, and a Monte
Carlo harness that classifies a filter as consistent, optimistic, or
conservative with the evidence attached. It is written for engineers choosing
between filter formulations for a tracking or fusion task and wanting the choice
made on measured numbers.

## Problem

A tracking filter reports two things: an estimate and the covariance it claims
for that estimate. Root mean square error checks the first. Nothing checks the
second unless you deliberately test it, and a filter whose covariance is wrong is
more dangerous than one that is merely inaccurate, because measurement gating,
data association, and any risk decision taken downstream all trust that number.

The problem this package addresses is therefore twofold. First, fuse two sensors
that report at different rates, on unrelated phases, with different transport
delays, one of them through a nonlinear polar measurement, without the usual
silent failure modes: an unwrapped angular innovation, a covariance that drifts
out of symmetry, or a Jacobian with a sign error that the gain absorbs until it
does not. Second, produce evidence about whether the resulting filter is
consistent, in a form that distinguishes a filter that is overconfident from one
that is merely imprecise.

## Approach

Three estimators are implemented behind one Protocol. The linear Kalman filter
(Kalman, 1960) is exact when the model and sensor are linear and exists so that
the two nonlinear filters can be checked against a known-correct answer rather
than against each other. The extended Kalman filter (Bar-Shalom, Li, and
Kirubarajan, 2001, section 10.3) propagates the mean through the true
nonlinearity and the covariance through analytic Jacobians, which the test suite
checks against central finite differences. The unscented Kalman filter (Wan and
van der Merwe, 2000) replaces the linearisation with the scaled unscented
transform (Julier, 2002), taking no derivative at all.

Three motion models are provided: constant velocity and constant acceleration,
both using the exact discretisation of a continuous-time system driven by white
noise, and constant turn rate and velocity, written in a cardinal sine form that
stays well conditioned and differentiable as the yaw rate passes through zero.
Consistency is assessed with normalised innovation squared and normalised
estimation error squared against chi-square bounds (Bar-Shalom, Li, and
Kirubarajan, 2001, section 5.4), averaged across independent Monte Carlo runs.
The alternatives that were considered and rejected, including the augmented-state
unscented filter, the interacting multiple model estimator, and full
out-of-sequence measurement retrodiction, are recorded in
[docs/design-notes.md](docs/design-notes.md).

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/kalman-sensor-fusion.git
cd kalman-sensor-fusion
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
from sensor_fusion.algorithm import UnscentedKalmanFilter
from sensor_fusion.analysis.report import assess
from sensor_fusion.pipeline.fusion import matched_belief
from sensor_fusion.pipeline.montecarlo import run_monte_carlo
from sensor_fusion.pipeline.scenarios import turning_target


def build(scenario):
    """Use the model that generated the truth, so the filter is correctly specified."""
    return UnscentedKalmanFilter(scenario.config.truth_model), matched_belief(scenario)


result = run_monte_carlo(turning_target(steps=2000), build, runs=30, seed=1)
for line in assess(result).lines():
    print(line)
```

Swapping `UnscentedKalmanFilter` for `ExtendedKalmanFilter` is the only change
needed to compare the two, and swapping the motion model for `ConstantVelocity`
or `ConstantAcceleration` is the only change needed to compare those.

Runnable examples live in `examples/`:

```bash
uv run python examples/compare_filters.py
uv run python examples/consistency_study.py
uv run python examples/asynchronous_fusion.py
uv run python examples/ekf_versus_ukf.py
```

Each accepts `--quick` for a short run, and `--runs`, `--steps`, and `--seed` to
change the configuration.

## Results

All numbers below are the output of the commands shown, on a fixed seed. The base
scenario, used by every section except the regime sweep, is a target whose speed
and yaw rate both random walk, observed by a lidar at 10 Hz and a radar at
13.33 Hz on an unrelated phase, over 10 s of simulated time on a 200 Hz truth
grid. It is defined once in `src/sensor_fusion/pipeline/scenarios.py` and shared
by the examples, the tests, and this document. Root mean square error is averaged
over independent Monte Carlo runs and quoted with the standard deviation across
those runs.

### Fusion against single-sensor baselines

From `uv run python examples/compare_filters.py`, 60 runs, seed 11:

| Configuration | Position RMSE (m) | Velocity RMSE (m/s) |
| --- | --- | --- |
| EKF, lidar and radar | 0.1195 +/- 0.0146 | 0.5204 +/- 0.1339 |
| UKF, lidar and radar | 0.1183 +/- 0.0124 | 0.5162 +/- 0.1140 |
| EKF, lidar only | 0.1384 +/- 0.0118 | 0.6432 +/- 0.1256 |
| EKF, radar only | 0.6217 +/- 0.1595 | 0.8463 +/- 0.2280 |

Fusing both sensors beats either alone: 15 percent better in position than lidar
alone and 81 percent better than radar alone. The gap between the extended and
unscented filters in this configuration is 1.0 percent in position and 0.8
percent in velocity, an order of magnitude smaller than the 12 percent
run-to-run spread, so on this configuration the two are not distinguishable on
accuracy.

The same scenario with the same unscented filter under three motion models, so
the effect of the motion model is separated from the effect of the filter:

| Motion model | Position RMSE (m) | Velocity RMSE (m/s) |
| --- | --- | --- |
| Constant velocity | 0.1642 +/- 0.0125 | 1.2380 +/- 0.1861 |
| Constant acceleration | 0.1594 +/- 0.0109 | 1.0566 +/- 0.0880 |
| CTRV, matching the truth | 0.1183 +/- 0.0124 | 0.5162 +/- 0.1140 |

Choosing the right motion model is worth 28 percent in position and a factor of
2.4 in velocity here, which is far more than the choice between the extended and
unscented filters is worth on the same data.

On a linear problem, lidar only and a constant velocity target, both nonlinear
filters reproduce the exact Kalman filter. The largest disagreement over 101
updates is 0 for the extended filter, which reduces to the same arithmetic, and
9.148e-14 in the mean with 7.050e-15 in the covariance for the unscented filter.

### Consistency verdicts with evidence

From `uv run python examples/consistency_study.py`, 60 runs, seed 2026. Bounds
are the two-sided 95 percent interval for the average of 60 independent
chi-square samples.

| Filter | Statistic | Expected | Measured mean | 60-run interval | Steps inside | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| CTRV, correctly specified | NEES | 5 | 5.076 | [4.232, 5.831] | 97.4 percent | consistent |
| CTRV, correctly specified | NIS lidar | 2 | 1.963 | [1.526, 2.537] | 92.1 percent | consistent |
| CTRV, correctly specified | NIS radar | 3 | 3.000 | [2.412, 3.651] | 93.2 percent | consistent |
| CV, spectral density 2 | NEES | 4 | 7.901 | [3.316, 4.747] | 2.1 percent | optimistic |
| CV, spectral density 2 | NIS lidar | 2 | 3.457 | [1.526, 2.537] | 10.9 percent | optimistic |
| CV, spectral density 2 | NIS radar | 3 | 3.770 | [2.412, 3.651] | 40.6 percent | optimistic |
| CV, spectral density 0.05 | NEES | 4 | 684.510 | [3.316, 4.747] | 0.9 percent | optimistic |
| CV, spectral density 0.05 | NIS lidar | 2 | 65.845 | [1.526, 2.537] | 1.0 percent | optimistic |
| CV, spectral density 0.05 | NIS radar | 3 | 45.045 | [2.412, 3.651] | 1.5 percent | optimistic |
| CV, spectral density 4000 | NEES | 4 | 1.996 | [3.316, 4.747] | 0.0 percent | conservative |
| CV, spectral density 4000 | NIS lidar | 2 | 0.620 | [1.526, 2.537] | 0.0 percent | conservative |
| CV, spectral density 4000 | NIS radar | 3 | 1.598 | [2.412, 3.651] | 2.3 percent | conservative |

The correctly specified filter lands on its expected value to within 2 percent
on all three statistics, and the fraction of time steps inside the interval sits
near or above the nominal 95 percent.

The row worth looking at is the constant velocity filter at spectral density 2.
It holds the track on all 60 runs and its position RMSE, 0.2150 m, is under twice
that of the correctly specified filter. Nothing about its output looks broken.
Its covariance is nevertheless wrong by roughly a factor of two, NEES 7.901
against an expected 4, so a gate sized from that covariance would be half the
width it should be. This is the failure that a root mean square error comparison
cannot see, and it is the reason the consistency machinery exists.

Starving the same filter further, to spectral density 0.05, turns the same defect
into outright track loss: NEES 684.5 and all 60 runs lost. Flooding it instead
makes it conservative, at NEES 1.996 against 4: it never loses the track and its
position error is only 0.3606 m, but its velocity RMSE is 5.60 m/s against
0.50 m/s for the correctly specified filter, because it discards almost
everything it has learned at each step.

Two properties of the correctly specified case are worth stating because they
are what makes the whole comparison meaningful. The truth is drawn from the
filter's own initial covariance and driven by the filter's own process noise, so
the chi-square assumptions genuinely hold. And the NEES here is taken in the
filter's native state space, not through a linearised Cartesian projection,
which would introduce an approximation into the test itself.

### When the unscented filter earns its cost

From `uv run python examples/ekf_versus_ukf.py`, 40 runs, seed 7. This sweep uses
its own scenario, a target starting 63 m from the sensor with a stronger yaw rate
disturbance, because the point is to vary the strength of the nonlinearity rather
than to reproduce the base scenario. NEES is reported as both the mean over runs
and the median over runs, because the mean is not robust: a single run in forty
that loses the target can raise it by two orders of magnitude. Degrees of freedom
is 5 throughout.

| Sensors | Filter | Position RMSE (m) | NEES mean | NEES median | Verdict | Tracks lost | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lidar and radar | EKF | 0.1328 | 5.752 | 5.242 | consistent | 0 of 40 | 1.18 s |
| Lidar and radar | UKF | 0.1300 | 4.929 | 4.865 | consistent | 0 of 40 | 2.13 s |
| Radar only, 13.3 Hz, bearing sigma 0.03 | EKF | 0.7425 | 8.953 | 5.477 | optimistic | 2 of 40 | 1.12 s |
| Radar only, 13.3 Hz, bearing sigma 0.03 | UKF | 0.6966 | 5.318 | 5.008 | consistent | 0 of 40 | 1.87 s |
| Radar only, 5 Hz, bearing sigma 0.10 | EKF | 2.7928 | 67.652 | 5.500 | optimistic | 1 of 40 | 0.80 s |
| Radar only, 5 Hz, bearing sigma 0.10 | UKF | 2.6250 | 59.304 | 5.228 | optimistic | 1 of 40 | 1.07 s |
| Radar only, 2.5 Hz, bearing sigma 0.20 | EKF | 4.6022 | 17.179 | 5.665 | optimistic | 1 of 40 | 0.75 s |
| Radar only, 2.5 Hz, bearing sigma 0.20 | UKF | 4.3807 | 5.899 | 5.128 | consistent | 0 of 40 | 0.82 s |

The accuracy difference is small everywhere, between 2 and 6 percent in position.
The difference that matters is elsewhere. With both sensors present both filters
are consistent. With radar alone the extended filter is classified optimistic in
all three regimes and loses tracks in all three, while the unscented filter stays
consistent in two of the three and loses no track in either of those. The
exception is the 5 Hz regime with a bearing standard deviation of 0.10 rad, where
both filters are optimistic and both lose one track: the unscented transform
recovers the moments of a nonlinearity, it does not rescue a filter from a
measurement that is too poor and too infrequent to support the state.

The cost is roughly a factor of two in wall clock time on the fused
configuration, where the eleven sigma points of the five-dimensional state
dominate. The times are wall clock on one machine and are quoted for the ratio,
not the absolute value.

### Asynchronous arrival and out-of-order handling

From `uv run python examples/asynchronous_fusion.py`, 40 runs, seed 5, with mean
transport latencies of 0.02 s for lidar and 0.09 s for radar and 0.01 s of
jitter. Of 234 reports per run, 88.5 on average arrive after a report with a
later timestamp.

| Policy | Position RMSE (m) | Velocity RMSE (m/s) | Processed | Discarded |
| --- | --- | --- | --- | --- |
| No delay, reference | 0.1180 +/- 0.0123 | 0.5238 +/- 0.0984 | 234.0 | 0.0 |
| Delayed, reorder buffer 0.15 s | 0.1180 +/- 0.0123 | 0.5238 +/- 0.0984 | 234.0 | 0.0 |
| Delayed, reorder buffer 0.00 s | 0.1252 +/- 0.0111 | 0.5827 +/- 0.1082 | 145.5 | 88.5 |
| Delayed, discard late reports | 0.1252 +/- 0.0111 | 0.5827 +/- 0.1082 | 145.5 | 88.5 |

A reorder buffer with a latency budget covering every arrival reproduces the
undelayed result exactly, to the last digit, at a cost of 0.15 s of output
delay. Discarding late reports throws away 38 percent of the measurements and
pays 6 percent in position error and 11 percent in velocity error. A buffer with
a zero latency budget is no better than discarding, which is the expected result:
with nothing held back there is nothing left to reorder. The scenario
draws truth, measurement noise, and transport latency from separate random
streams, so attaching latency changes only when reports arrive and not what they
say; the comparison is between policies and nothing else.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/sensor_fusion/_types.py` | The float64 array alias every public signature uses |
| `src/sensor_fusion/_math.py` | Root mean square error, needed by two layers that may not import each other |
| `src/sensor_fusion/model/angles.py` | Wrapping into `[-pi, pi)` |
| `src/sensor_fusion/model/motion.py` | Constant velocity, constant acceleration, and CTRV models with analytic Jacobians, process noise, and closed-form noise factors |
| `src/sensor_fusion/model/measurement.py` | Lidar and radar models with analytic Jacobians and angle-aware residuals |
| `src/sensor_fusion/algorithm/base.py` | The `StateEstimator` Protocol, `GaussianState`, and shared numerical helpers |
| `src/sensor_fusion/algorithm/kf.py` | Linear Kalman filter, exact for a linear model and linear sensor |
| `src/sensor_fusion/algorithm/ekf.py` | Extended Kalman filter with analytic Jacobians and Joseph form covariance update |
| `src/sensor_fusion/algorithm/sigma.py` | Scaled unscented transform: sigma points, weights, angle-aware moments |
| `src/sensor_fusion/algorithm/ukf.py` | Unscented Kalman filter |
| `src/sensor_fusion/pipeline/simulator.py` | Ground truth and measurement generation with independent random streams |
| `src/sensor_fusion/pipeline/scenarios.py` | The named scenarios used by examples, tests, and this README |
| `src/sensor_fusion/pipeline/fusion.py` | Asynchronous sequential fusion with the out-of-order policies |
| `src/sensor_fusion/pipeline/montecarlo.py` | Repeating a scenario under independent noise and stacking the statistics |
| `src/sensor_fusion/pipeline/trace.py` | The structured record of a run |
| `src/sensor_fusion/analysis/consistency.py` | NIS, NEES, chi-square intervals, and the verdict |
| `src/sensor_fusion/analysis/report.py` | Turning stacked Monte Carlo statistics into an assessment |
| `src/sensor_fusion/analysis/metrics.py` | Root mean square error |
| `src/sensor_fusion/analysis/figures.py` | Track, error, and consistency figures |
| `examples/` | Thin wiring scripts with no logic of their own |

The dependency direction is one way: `model` knows nothing of anything else,
`algorithm` uses `model`, `pipeline` uses both, `analysis` reads what `pipeline`
produces, and `examples` only wires those together.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

Tier one covers the mathematics. Both nonlinear filters are checked against the
exact Kalman filter on a linear Gaussian problem; every analytic Jacobian is
checked against a central finite difference; the unscented transform is checked
to reproduce the mean and covariance of a linear map exactly and the mean of a
quadratic form, which a Jacobian linearisation misses by construction; sigma
point mean weights are checked to sum to one; every covariance is checked to stay
symmetric and positive semi-definite through every prediction and update; a
noiseless constant velocity prediction is checked to be exact and to compose over
sub-steps; and a correctly specified filter is checked to fall inside its
chi-square bounds under a fixed seed. Angle wrapping is exercised by a target
whose bearing and heading both cross the plus or minus pi boundary, with a
negative control that replaces the wrapped residual by a plain subtraction and
asserts that the same run then goes visibly wrong.

Tier two pins a recorded reference run. It pins integer counts, classifications,
closed-form quantities, and aggregate metrics of the numerically stable filter
recursion. It deliberately does not pin the statistics of the mis-specified
filters by value, because a filter running far outside its assumptions is not a
well-conditioned system and its exact numbers are not a sound thing to require of
another machine; those cases are asserted by classification and by wide
qualitative bounds instead. The reasoning is written out in the module docstring
of `tests/test_regression.py`.

Tier three loads each script in `examples/` and calls its `main` with `--quick`,
asserting a zero return, non-empty output, and that scripts claiming to write
figures do write them.

The full suite of 189 tests runs in under 10 seconds.

## References

Algorithms:

- Kalman, R. E. "A New Approach to Linear Filtering and Prediction Problems."
  *Journal of Basic Engineering* 82, no. 1 (1960): 35 to 45.
  DOI [10.1115/1.3662552](https://doi.org/10.1115/1.3662552).
- Julier, S. J. "The Scaled Unscented Transformation." *Proceedings of the 2002
  American Control Conference* (2002): 4555 to 4559.
  DOI [10.1109/ACC.2002.1025369](https://doi.org/10.1109/ACC.2002.1025369).
- Wan, E. A., and R. van der Merwe. "The Unscented Kalman Filter for Nonlinear
  Estimation." *Proceedings of the IEEE Adaptive Systems for Signal Processing,
  Communications, and Control Symposium* (2000): 153 to 158.
  DOI [10.1109/ASSPCC.2000.882463](https://doi.org/10.1109/ASSPCC.2000.882463).
- Julier, S. J., and J. K. Uhlmann. "Unscented Filtering and Nonlinear
  Estimation." *Proceedings of the IEEE* 92, no. 3 (2004): 401 to 422.
  DOI [10.1109/JPROC.2003.823141](https://doi.org/10.1109/JPROC.2003.823141).
- Bar-Shalom, Y., X. R. Li, and T. Kirubarajan. *Estimation with Applications to
  Tracking and Navigation: Theory, Algorithms and Software.* Wiley, 2001.
  DOI [10.1002/0471221279](https://doi.org/10.1002/0471221279). Sections 5.4
  (consistency of state estimators), 6.2 (discretised continuous-time process
  noise), 10.3 (the extended Kalman filter), and 11.7 (coordinated turn models).
- Bar-Shalom, Y. "Update with Out-of-Sequence Measurements in Tracking: Exact
  Solution." *IEEE Transactions on Aerospace and Electronic Systems* 38, no. 3
  (2002): 769 to 777.
  DOI [10.1109/TAES.2002.1039398](https://doi.org/10.1109/TAES.2002.1039398).
  Cited as the treatment of late measurements that this package does not
  implement.
- Blom, H. A. P., and Y. Bar-Shalom. "The Interacting Multiple Model Algorithm
  for Systems with Markovian Switching Coefficients." *IEEE Transactions on
  Automatic Control* 33, no. 8 (1988): 780 to 783.
  DOI [10.1109/9.1299](https://doi.org/10.1109/9.1299). Cited as a rejected
  alternative in the design notes.
- Bucy, R. S., and P. D. Joseph. *Filtering for Stochastic Processes with
  Applications to Guidance.* Interscience, 1968. Reprinted by AMS Chelsea, 2005,
  ISBN 978-0-8218-3782-5. The Joseph form covariance update used in the linear
  and extended filters is named for and attributed to Joseph, and is given here.
- Schmidt, S. F. "Application of State-Space Methods to Navigation Problems."
  *Advances in Control Systems* 3 (1966): 293 to 340.
  DOI [10.1016/B978-1-4831-6716-9.50011-4](https://doi.org/10.1016/B978-1-4831-6716-9.50011-4).
  The first flight application of the extended Kalman filter, and the source of
  the numerical stability practices this implementation follows.

Dependencies:

- [NumPy](https://numpy.org/) (>= 2.0), BSD 3-Clause. Array arithmetic, linear
  algebra, and the seeded random generators that make every scenario
  reproducible.
- [SciPy](https://scipy.org/) (>= 1.14), BSD 3-Clause. `scipy.stats.chi2` only,
  for the chi-square quantiles behind the consistency intervals.
- [Matplotlib](https://matplotlib.org/) (>= 3.9), Matplotlib license, a
  BSD-style permissive license. Figures, used under the non-interactive Agg
  backend.
- [pytest](https://pytest.org/) (>= 8.3), MIT. Test runner, development only.
- [Ruff](https://docs.astral.sh/ruff/) (>= 0.8), MIT. Linter and import sorter,
  development only.
- [mypy](https://mypy-lang.org/) (>= 1.13), MIT. Static type checker in strict
  mode, development only.

## License

Released under the MIT license. See [LICENSE](LICENSE).
