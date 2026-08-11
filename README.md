# Kalman Sensor Fusion

Extended and unscented Kalman filters fusing simulated radar and lidar, with the
consistency machinery to decide which of them you should actually pay for.

[![CI](https://github.com/Eelis03/kalman-sensor-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/kalman-sensor-fusion/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Three stacked panels of across-run mean NEES against time with the 95 percent chi-square interval shaded. The correctly specified filter stays inside the interval for the whole run, the under-noised filter climbs out of it within a second and never comes back, and the over-noised filter sits below it throughout.](docs/figures/nees-specification.png)

An unscented Kalman filter costs about twice what an extended one costs on the
same problem. The usual advice is to pay it, on the grounds that the unscented
transform propagates moments to third order where a Jacobian linearisation
manages first. This repository was built to find out when that difference is
worth anything, by measuring it rather than by citing it.

The short answer, from the numbers below. On the scenario most tracking demos
use, a target circling about 64 m from the sensor, the two filters are
indistinguishable: 0.1182 m against 0.1194 m in position, both consistent, and
they stay indistinguishable all the way down to a 2.5 Hz radar with a bearing
standard deviation of 0.20 rad. Move the same target out to 100 m and let it
leave, and the extended filter's covariance is wrong even with both sensors
present, at a normalised estimation error squared of 7.750 against an expected 5,
while the unscented filter sits at 4.982. With radar alone the extended filter
loses 8 tracks in 40 against the unscented filter's 2.

The deciding variable is not which sensor you have. It is how wide the covariance
gets relative to the curvature of the measurement across it, and the fastest way
to find out is to run both filters and look at which one stays inside its
chi-square bounds. That test is what most of this package is.

The three panels above are what that test looks like when it is not summarised.
Each shows the across-run mean of the normalised estimation error squared for one
filter, against the interval it should sit in. All three have a single number
attached to them in the table further down, and none of those numbers tells you
that the under-noised filter is wrong from the first second onward while the
over-noised one is wrong evenly across the whole run.

## Results

Every number here is printed by a script in `examples/`, named above the table it
produced, on the seed given. Root mean square error is averaged over independent
Monte Carlo runs and quoted with the standard deviation across those runs.

Two scenarios are used, both defined in
`src/sensor_fusion/pipeline/scenarios.py` and shared by the examples, the tests,
and this document. The base scenario is a target whose speed and yaw rate both
random walk on a circle of radius 12.5 m centred 64 m from the sensor, observed
by a lidar at 10 Hz and a radar at 13.33 Hz on an unrelated phase, over 10 s on a
200 Hz truth grid. The distant scenario starts 100 m out at 14 m/s on an outward
heading and turns at 0.35 rad/s under a yaw disturbance five times larger.

### When the unscented filter earns its cost

From `uv run python examples/ekf_versus_ukf.py`, 40 runs, seed 7. The same
four-rung sensor ladder is applied to both targets, so the only thing varying
across a block is what observes the target. Degrees of freedom is 5 throughout.
NEES is given as the mean over runs and the median over runs, because the mean is
not robust: one run in forty that loses the target moves it by two orders of
magnitude.

Base scenario, target circling 64 m from the sensor:

| Sensors | Filter | Position RMSE (m) | NEES mean | NEES median | Steps above bound | Verdict | Lost | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lidar and radar | EKF | 0.1194 | 4.878 | 4.766 | 4.3 percent | consistent | 0 of 40 | 1.40 s |
| Lidar and radar | UKF | 0.1182 | 4.730 | 4.717 | 0.9 percent | consistent | 0 of 40 | 2.74 s |
| Radar only, 13.3 Hz, sigma 0.03 | EKF | 0.6323 | 5.381 | 4.883 | 13.4 percent | consistent | 0 of 40 | 1.27 s |
| Radar only, 13.3 Hz, sigma 0.03 | UKF | 0.6269 | 4.969 | 4.682 | 1.5 percent | consistent | 0 of 40 | 1.99 s |
| Radar only, 5 Hz, sigma 0.10 | EKF | 1.6891 | 5.125 | 4.926 | 7.8 percent | consistent | 0 of 40 | 0.93 s |
| Radar only, 5 Hz, sigma 0.10 | UKF | 1.7123 | 4.996 | 4.593 | 0.0 percent | consistent | 0 of 40 | 1.20 s |
| Radar only, 2.5 Hz, sigma 0.20 | EKF | 2.9741 | 5.323 | 4.909 | 19.2 percent | consistent | 0 of 40 | 0.83 s |
| Radar only, 2.5 Hz, sigma 0.20 | UKF | 3.0100 | 5.075 | 4.878 | 7.7 percent | consistent | 0 of 40 | 0.99 s |

Nothing separates the two filters here. Position RMSE agrees to within 1.4
percent on every rung, in both directions, against a run-to-run spread of about
12 percent. Every campaign is classified consistent. The extended filter spends
more of the run above its upper bound on three rungs out of four, which is the
only trace of a difference, and it is not enough to change a decision. On this
target the extended filter is the correct choice, because it is the cheaper one.

Distant scenario, target 100 m out and leaving:

| Sensors | Filter | Position RMSE (m) | NEES mean | NEES median | Steps above bound | Verdict | Lost | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lidar and radar | EKF | 0.1520 | 7.750 | 5.408 | 13.7 percent | optimistic | 0 of 40 | 1.72 s |
| Lidar and radar | UKF | 0.1316 | 4.982 | 4.937 | 3.4 percent | consistent | 0 of 40 | 2.57 s |
| Radar only, 13.3 Hz, sigma 0.03 | EKF | 2.0377 | 103.821 | 10.923 | 85.1 percent | optimistic | 8 of 40 | 1.22 s |
| Radar only, 13.3 Hz, sigma 0.03 | UKF | 1.4042 | 10.439 | 5.012 | 54.5 percent | optimistic | 2 of 40 | 1.84 s |
| Radar only, 5 Hz, sigma 0.10 | EKF | 6.7567 | 165.469 | 9.095 | 100.0 percent | optimistic | 4 of 40 | 0.83 s |
| Radar only, 5 Hz, sigma 0.10 | UKF | 5.2671 | 43.850 | 5.637 | 94.1 percent | optimistic | 1 of 40 | 1.17 s |
| Radar only, 2.5 Hz, sigma 0.20 | EKF | 12.3791 | 109.596 | 8.159 | 100.0 percent | optimistic | 6 of 40 | 0.77 s |
| Radar only, 2.5 Hz, sigma 0.20 | UKF | 10.3633 | 56.885 | 5.533 | 92.3 percent | optimistic | 4 of 40 | 0.98 s |

![Across-run mean NEES against time on a logarithmic axis for the extended and unscented Kalman filters given identical data from the distant target with both sensors. The extended filter starts at twenty times its expected value and takes more than a second to fall inside the shaded chi-square interval, while the unscented filter is inside it from the first update.](docs/figures/nees-ekf-ukf.png)

The first row is the interesting one, and the figure is why. Both filters see the
same measurements, neither loses a track, and the accuracy gap is 13 percent,
which on its own would be a rounding error in this business. The covariance gap
is the whole width of the interval. What the mean of 7.750 does not say, and the
figure does, is that the extended filter is not mildly wrong throughout: it
starts at about twenty times its expected value and takes more than a second to
come back, and the unscented filter starting from exactly the same loose prior
never leaves. If your filter is initialised often, from re-acquisition or from
track birth, that first second is not a detail.

With radar alone the difference stops being about the covariance and starts being
about whether there is a track at all. Both filters are classified optimistic on
all three rungs, which is the honest reading: the unscented transform recovers
the moments of a nonlinearity, it does not manufacture information, and a target
at 100 to 270 m seen only in polar coordinates at 2.5 Hz is not well observed by
anything. Within that, the extended filter loses 8, 4 and 6 tracks against 2, 1
and 4, and its median run NEES is between 1.5 and 2.2 times the unscented
filter's on every rung.

The cost is a factor of 2.0 in wall clock time on the fused base configuration
and 1.2 to 1.6 on the radar-only rungs, where the state is updated less often.
Times are wall clock on one machine and are quoted for the ratio.

### Deciding whether a covariance can be believed

Root mean square error checks the estimate. Nothing checks the covariance the
filter claims for that estimate unless you test it deliberately, and a filter
whose covariance is wrong is more dangerous than one that is merely imprecise,
because gating, association, and every risk decision taken downstream trust that
number.

From `uv run python examples/consistency_study.py`, 60 runs, seed 2026. Bounds
are the two-sided 95 percent interval for the average of 60 independent
chi-square samples. The three filters in the figure at the top of this page are
the first, second and fourth blocks here.

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

The correctly specified filter lands on its expected value to within 2 percent on
all three statistics, and the fraction of time steps inside the interval sits at
or above the nominal 95 percent on NEES. That is the calibration the rest of the
table is read against, and it is meaningful only because the truth is drawn from
the filter's own initial covariance and driven by the filter's own process noise,
so the chi-square assumptions genuinely hold rather than approximately hold.

The row worth looking at is the constant velocity filter at spectral density 2.
It holds the track on all 60 runs and its position RMSE, 0.2150 m, is under twice
that of the correctly specified filter. Nothing about its output looks broken.
Its covariance is nevertheless wrong by roughly a factor of two, NEES 7.901
against an expected 4, so a gate sized from that covariance would be half the
width it should be. This is the failure a root mean square error comparison
cannot see, and it is the reason the consistency machinery exists.

Starving the same filter further, to spectral density 0.05, turns the same defect
into outright track loss: NEES 684.510 and all 60 runs lost. Flooding it instead
makes it conservative, at 1.996 against 4. It then never loses the track and its
position error is only 0.3606 m, but its velocity RMSE is 5.5953 m/s against
0.5016 m/s for the correctly specified filter, because it throws away almost
everything it has learned at every step.

Magnitude is not the only thing that can be wrong with an innovation. A residual
can be the right size and still be predictable, and a filter whose consecutive
innovations correlate is leaving structure that a better model would have
removed. The same script prints the time-average autocorrelation of each
sensor's innovations, scaled to unit covariance first so that the radar's metres
and radians carry equal weight, against the two-sided bound for a white sequence
corrected for the three lags tested. Like NIS, it needs no ground truth.

| Filter | Sensor | Largest correlation | At lag | Bound | Verdict |
| --- | --- | --- | --- | --- | --- |
| CTRV, correctly specified | lidar | -0.0095 | 2 | 0.0220 | white |
| CTRV, correctly specified | radar | +0.0110 | 1 | 0.0155 | white |
| CV, spectral density 2 | lidar | +0.4483 | 3 | 0.0221 | correlated |
| CV, spectral density 2 | radar | +0.3239 | 3 | 0.0156 | correlated |
| CV, spectral density 0.05 | lidar | +0.9654 | 1 | 0.0219 | correlated |
| CV, spectral density 0.05 | radar | +0.9301 | 1 | 0.0155 | correlated |
| CV, spectral density 4000 | lidar | -0.1387 | 1 | 0.0219 | correlated |
| CV, spectral density 4000 | radar | -0.0114 | 1 | 0.0155 | white |

The correctly specified filter is white on both sensors, which is again the
calibration the rest of the table is read against. The two starved filters are
correlated and positively so, which is what an unmodelled turn looks like: the
part of the motion the model omits does not change between one update and the
next, so neither does the residual it leaves.

The last two rows are the ones a magnitude test cannot produce. The flooded
filter is classified conservative by NEES and by both NIS statistics, all saying
the same thing in the same direction, and its lidar innovations are correlated at
-0.1387 against a bound of 0.0219. The sign is the finding. A filter assuming a
spectral density of 4000 takes each measurement at close to face value, so the
noise it absorbs at one update returns with the opposite sign in the next
innovation, and the first difference of a white sequence is not white. Optimistic
and conservative are the two things a magnitude test can say. Overcorrection is a
third, and this is the statistic that says it.

NIS and NEES are both reported because they answer different questions. NIS needs
no ground truth and can therefore be computed against a live sensor feed; it only
tests the filter against its own predictions. NEES needs ground truth and is
therefore a simulation tool; it tests the estimate against the world. Where the
filter's motion model is the model that generated the truth, NEES is taken in the
filter's native state space and no projection is involved. Where it is not, the
Cartesian view is used and its second moment is computed in closed form rather
than through a Jacobian, so the statistic still has expectation 4 exactly. That
is written up under Closed limitations in
[docs/design-notes.md](docs/design-notes.md).

### Fusing two sensors against using one

From `uv run python examples/compare_filters.py`, 60 runs, seed 11, base
scenario:

| Configuration | Position RMSE (m) | Velocity RMSE (m/s) |
| --- | --- | --- |
| EKF, lidar and radar | 0.1195 +/- 0.0146 | 0.5204 +/- 0.1339 |
| UKF, lidar and radar | 0.1183 +/- 0.0124 | 0.5162 +/- 0.1140 |
| EKF, lidar only | 0.1384 +/- 0.0118 | 0.6432 +/- 0.1256 |
| EKF, radar only | 0.6217 +/- 0.1595 | 0.8463 +/- 0.2280 |

![Estimated tracks against ground truth for the base scenario, with the sensor marked at the origin and a dashed line marking the closest approach of 54 m. The tracks using lidar are indistinguishable from the truth at this scale, while the radar-only track wanders visibly off it.](docs/figures/tracks.png)

Fusing both sensors beats either alone. Holding the filter fixed at the extended
one, adding the radar to the lidar is worth 14 percent in position and adding the
lidar to the radar is worth 81 percent.

The figure shows why the sensor is drawn. Range, bearing, and range rate are all
degenerate at the origin, so a scenario whose target passes close to it measures
a singularity as much as it measures a filter. Every named scenario here is
placed to keep well clear, and a test asserts a minimum range of 15 m over 200
seeds for each of them. That test exists because the problem happened twice:
once in the original turning scenario, and once in the sweep configuration that
produced the earlier version of the table above, which lived as literals inside
an example script and was therefore never checked. Both cases are written up in
[docs/design-notes.md](docs/design-notes.md).

Choosing the motion model matters more on this scenario than choosing the filter.
The same unscented filter under three models:

| Motion model | Position RMSE (m) | Velocity RMSE (m/s) |
| --- | --- | --- |
| Constant velocity | 0.1642 +/- 0.0125 | 1.2380 +/- 0.1861 |
| Constant acceleration | 0.1594 +/- 0.0109 | 1.0566 +/- 0.0880 |
| CTRV, matching the truth | 0.1183 +/- 0.0124 | 0.5162 +/- 0.1140 |

That is 28 percent in position and a factor of 2.4 in velocity, against the 1.0
percent the filter choice is worth on the same data.

On a linear problem, lidar only and a constant velocity target, both nonlinear
filters reproduce the exact Kalman filter. The largest disagreement over 101
updates is 0 for the extended filter, which reduces to the same arithmetic, and
9.148e-14 in the mean with 7.050e-15 in the covariance for the unscented filter.

### Asynchronous arrival and out-of-order reports

From `uv run python examples/asynchronous_fusion.py`, 40 runs, seed 5, with mean
transport latencies of 0.02 s for lidar and 0.09 s for radar and 0.01 s of
jitter. Of 234 reports per run, 88.5 on average arrive after a report carrying a
later timestamp.

| Policy | Position RMSE (m) | Velocity RMSE (m/s) | Processed | Discarded |
| --- | --- | --- | --- | --- |
| No delay, reference | 0.1180 +/- 0.0123 | 0.5238 +/- 0.0984 | 234.0 | 0.0 |
| Delayed, reorder buffer 0.15 s | 0.1180 +/- 0.0123 | 0.5238 +/- 0.0984 | 234.0 | 0.0 |
| Delayed, reorder buffer 0.00 s | 0.1252 +/- 0.0111 | 0.5827 +/- 0.1082 | 145.5 | 88.5 |
| Delayed, discard late reports | 0.1252 +/- 0.0111 | 0.5827 +/- 0.1082 | 145.5 | 88.5 |

A reorder buffer whose latency budget covers every arrival reproduces the
undelayed result exactly, to the last digit, at a cost of 0.15 s of output delay.
Discarding late reports throws away 38 percent of the measurements and pays 6
percent in position error and 11 percent in velocity error. A zero-budget buffer
is no better than discarding, which is the expected result: with nothing held
back there is nothing to reorder. The scenario draws truth, measurement noise,
and transport latency from separate random streams, so attaching latency changes
only when reports arrive and not what they say.

## Installation

Requires Python 3.12 or later. Continuous integration runs the whole suite on
3.12 and 3.13, on Linux and on Windows, so the version floor in `pyproject.toml`
is a tested claim rather than a declared one.

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

The package ships a `py.typed` marker, so annotations are visible to a type
checker running against code that imports it.

## Reproducing every number here

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

Each table above is one script:

```bash
uv run python examples/ekf_versus_ukf.py        # the two sweep tables
uv run python examples/consistency_study.py     # the consistency table
uv run python examples/compare_filters.py       # fusion and motion model tables
uv run python examples/asynchronous_fusion.py   # the out-of-order table
```

Each accepts `--quick` for a short run, and `--runs`, `--steps`, and `--seed` to
change the configuration.

The three figures in this document are regenerated by one command:

```bash
uv run python examples/make_figures.py
```

They are committed snapshots, not build artefacts. Continuous integration does
not compare them byte for byte against a fresh run, because matplotlib output is
not byte reproducible across platforms or across its own releases and such a
check would fail for reasons that have nothing to do with this package. What is
checked is that the code paths that draw them still run, that the files are
present, and that they stay inside the repository's 250 KB figure budget.

```bash
uv run pytest --cov=src/sensor_fusion --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Line coverage is 98 percent. Continuous integration enforces a floor of 96, which
is the measured value rounded down and reduced by two, so that a refactor does
not fail the build while a module arriving with no tests at all does.

The suite has 223 tests and runs in about 30 seconds. Two of them are skipped by
declaration, both because they assert a property only a linear model can have. It
is built in three tiers. Tier one covers the mathematics: both nonlinear filters
against the exact Kalman filter on a linear Gaussian problem, every analytic
Jacobian against a central finite difference, the unscented transform against the
mean of a quadratic form that a linearisation misses by construction, every
covariance for symmetry and positive semi-definiteness through every step, the
closed-form Cartesian second moment against a 400000 sample Monte Carlo
projection, a correctly specified filter against its chi-square bounds, and that
same filter's innovations against the whiteness bound with a constant velocity
filter on the turning target as the control that the test can fire at all. Angle
wrapping is exercised by a target whose bearing and heading both cross the plus
or minus pi cut, with a negative control that replaces the wrapped residual by a
plain subtraction and asserts that the same run then goes visibly wrong.

Tier two pins a recorded reference run: integer counts, classifications,
closed-form quantities, and aggregate metrics of the numerically stable filter
recursion. It deliberately does not pin the statistics of the mis-specified
filters by value, because a filter running far outside its assumptions is not a
well-conditioned system and its exact numbers are not a sound thing to require of
another machine. The reasoning is in the module docstring of
`tests/test_regression.py`.

Tier three loads each script in `examples/` and calls its `main` with `--quick`,
asserting a zero return, non-empty output, and that scripts claiming to write
figures do write them.

## How it is put together

Three estimators sit behind one `StateEstimator` Protocol, so the pipeline and
analysis layers never branch on which one they hold. The linear Kalman filter
(Kalman, 1960) exists so the two nonlinear filters have a known-correct answer to
be checked against rather than only each other. The extended filter (Bar-Shalom,
Li, and Kirubarajan, 2001, section 10.3) propagates the mean through the true
nonlinearity and the covariance through analytic Jacobians. The unscented filter
(Wan and van der Merwe, 2000) replaces the linearisation with the scaled
unscented transform (Julier, 2002) and takes no derivative at all.

Three motion models are provided: constant velocity and constant acceleration,
both using the exact discretisation of a continuous-time system driven by white
noise so that the process noise composes over sub-intervals, and constant turn
rate and velocity, written in a cardinal sine form that stays well conditioned
and differentiable as the yaw rate passes through zero.

| Layer | Contents |
| --- | --- |
| `model/` | Angle wrapping, the three motion models with their Jacobians and closed-form noise factors, and the lidar and radar measurement models with angle-aware residuals |
| `algorithm/` | The `StateEstimator` Protocol, `GaussianState`, shared numerical helpers, the three filters, and the scaled unscented transform |
| `pipeline/` | Ground truth and measurement generation on independent random streams, the named scenarios, asynchronous sequential fusion with the out-of-order policies, the Monte Carlo harness, and the structured record of a run |
| `analysis/` | NIS, NEES, innovation whiteness, chi-square intervals, verdicts, and figures |
| `examples/` | Thin wiring scripts with no logic of their own |

The dependency direction is one way: `model` knows nothing of anything else,
`algorithm` uses `model`, `pipeline` uses both, `analysis` reads what `pipeline`
produces, and `examples` only wires those together.

Design decisions, the alternatives that were rejected and why, the limitations
that remain, and the one that was closed together with what closing it cost are
recorded in [docs/design-notes.md](docs/design-notes.md).

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
- [pytest](https://pytest.org/) (>= 8.3), MIT, and
  [pytest-cov](https://pytest-cov.readthedocs.io/) (>= 6.0), MIT. Test runner and
  coverage measurement, development only.
- [Ruff](https://docs.astral.sh/ruff/) (>= 0.8), MIT. Linter and import sorter,
  development only.
- [mypy](https://mypy-lang.org/) (>= 1.13), MIT. Static type checker in strict
  mode, development only.

## License

Released under the MIT license. See [LICENSE](LICENSE).
