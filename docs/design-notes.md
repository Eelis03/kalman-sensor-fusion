# Design notes for Kalman Sensor Fusion

## Method selection

### Three estimators behind one Protocol

The package implements a linear Kalman filter, an extended Kalman filter, and an
unscented Kalman filter, all satisfying one `StateEstimator` Protocol so that the
pipeline and analysis layers never branch on which one they hold.

The linear filter (Kalman, 1960) is not there for production use. It is there
because it is exact when the model and sensor are linear, which gives the two
nonlinear filters something known-correct to be checked against. Checking the
extended filter against the unscented filter would only establish that they
agree, not that either is right. On the linear scenario in this package the
extended filter reproduces the linear filter bit for bit, since it reduces to the
same arithmetic, and the unscented filter agrees to 9.1e-14 in the mean.

The extended filter (Bar-Shalom, Li, and Kirubarajan, 2001, section 10.3)
propagates the mean through the exact nonlinearity and the covariance through
analytic Jacobians. Analytic Jacobians were chosen over automatic or numerical
differentiation because the models are small and closed form, so the derivative
is a few lines, and because the correctness of those lines is checkable: the test
suite compares every Jacobian against a central finite difference at several
states and time steps. A Jacobian with a sign error does not raise; the gain
absorbs it and the filter tracks acceptably for a while before diverging, so that
comparison is not optional.

The unscented filter (Wan and van der Merwe, 2000) replaces the linearisation
with the scaled unscented transform (Julier, 2002). It takes no derivative and
propagates moments accurately to third order for a Gaussian prior, against the
first order of a linearisation.

### Unscented transform parameters

The scaling is `lambda = alpha**2 * (n + kappa) - n`. The defaults are
`alpha = 1.0`, `kappa = 0.0`, `beta = 2.0`, which is a deliberate departure from
the `alpha = 1e-3` often quoted, for two reasons.

With `alpha = 1` and `kappa = 0` the scaling is zero, the mean weight of the
centre sigma point is zero, and every remaining weight is `1 / (2 * n)`. All mean
and covariance weights are then non-negative, so the recovered covariance is a
non-negative combination of outer products and is positive semi-definite by
construction. A negative centre weight, which other parameter choices produce,
can return an indefinite covariance from a perfectly well posed problem.

The second reason is numerical. With `alpha = 1e-3` the sigma points sit within a
thousandth of a standard deviation of the mean, and the covariance is recovered
by dividing squared differences of nearly equal numbers by a weight of order
`1e6`. In double precision that cancellation costs roughly six significant
digits. With `lambda = 0` the spread is one standard deviation times the square
root of the dimension and no cancellation occurs, which is what allows the
unscented filter here to agree with the exact Kalman filter to 1e-14 rather than
1e-8. `beta = 2` is the value Wan and van der Merwe give as optimal for a
Gaussian prior.

### Process noise formulations

Constant velocity uses continuous white noise acceleration and constant
acceleration uses continuous Wiener process acceleration, both from Bar-Shalom,
Li, and Kirubarajan (2001) section 6.2. Both are the exact second moment of the
underlying continuous-time process, which means they compose: propagating over
two intervals of `dt` accumulates the same covariance as one interval of
`2 * dt`. That property is not decorative. The simulator generates ground truth
on a 200 Hz grid while the filter runs on the 10 Hz and 13.33 Hz measurement
grids, and without composition the truth would carry a different amount of
disturbance from what the filter assumes, so every consistency test would report
a conservative filter for a reason unrelated to the filter.

CTRV was originally written with the commonly published piecewise-constant
acceleration form, parameterised by standard deviations rather than spectral
densities. Measured NEES came out near 3.0 against an expected 5.0 and the filter
looked conservative. The cause was exactly the composition failure above: that
formulation injects variance proportional to `dt**2` per step, so refining the
grid by a factor of fifteen reduced the total injected disturbance by the same
factor. It was replaced with two continuous white noise acceleration chains, one
from longitudinal acceleration into speed into along-track position and one from
yaw acceleration into yaw rate into heading. Measured NEES is now 5.076 against
an expected 5.

That replacement is exact in the speed, heading, and yaw rate components and
approximate in position, because the along-track direction rotates within the
interval. Comparing one step of `dt` against the accumulation of the same
covariance over fifteen sub-steps at a yaw rate of 0.4 rad/s gives a relative
error of 2.7e-4 at `dt = 0.05` and 6.1e-4 at `dt = 0.075`, which is the largest
step the measurement schedule produces. The error scales as the square of
`omega * dt`, and at that magnitude it is not detectable in the consistency
statistics.

### CTRV written with the cardinal sine

The usual CTRV propagation divides by the yaw rate and needs a separate
straight-line branch near zero. That branch is a defect, not just an
inconvenience: the propagation becomes non-differentiable across it, so the
analytic Jacobian cannot agree with a finite difference taken across the
boundary, and the test that would catch a Jacobian error has to be weakened to
avoid the region.

Writing the same solution with the cardinal sine removes the branch. With
`u = omega * dt / 2`,

```
px' = px + v * dt * sinc(u) * cos(psi + u)
py' = py + v * dt * sinc(u) * sin(psi + u)
psi' = psi + 2 * u
```

which reduces continuously to straight-line motion at zero yaw rate. Below an
argument of 1e-6 both `sinc` and its derivative switch to their series
expansions. For `sinc` itself that switch is cosmetic, since `sin(u) / u` is well
conditioned everywhere. For the derivative it is not: `(u cos u - sin u) / u**2`
subtracts two quantities that agree to order `u**2` and divides by `u**2`, losing
about two decimal digits per factor of ten in `u`. The Jacobian is the
analytic derivative of this form and matches a central finite difference at every
yaw rate tested, including exactly zero.

### Angle handling

Every angular quantity is wrapped into `[-pi, pi)` before it is averaged,
differenced, or fed into a covariance. The radar bearing is wrapped in the
innovation, the CTRV heading is wrapped in the state, and the unscented filter
unwraps its sigma points relative to the centre point before averaging them.

Unwrapping relative to a reference was chosen over the circular mean of sines and
cosines because it stays well defined when some weights are negative, which they
are for parameter choices other than the default.

The propagation deliberately does not wrap. Wrapping inside `predict` would make
it discontinuous at the boundary, and a discontinuous propagation cannot agree
with its own analytic Jacobian. Wrapping is done by `normalize`, which the
filters apply to the mean once the linear algebra is finished.

The test suite includes a negative control: a radar whose residual is a plain
subtraction, run on a scenario whose bearing and heading both cross the boundary.
The maximum position error is more than twenty times worse. That is what the
wrapping is for.

### Covariance updates

The linear and extended filters use the Joseph form, named for Peter Joseph and
given in Bucy and Joseph (1968),

```
P = (I - K H) P (I - K H).T + K R K.T
```

rather than the shorter `(I - K H) P`. It costs one extra pair of matrix products
and stays symmetric positive semi-definite even when the gain is not exactly
optimal, which the short form does not. Every covariance is symmetrised after
every step, because the recursion accumulates an antisymmetric component of the
order of the rounding error that eventually makes a Cholesky factorisation fail.

### Consistency testing

Normalised innovation squared and normalised estimation error squared are used
together because they answer different questions. NIS needs no ground truth and
is therefore computable on a real vehicle, but it only tests the filter against
its own predictions. NEES needs ground truth and is therefore a simulation tool,
but it tests the estimate against the world.

The interval is the two-sided 95 percent interval for the average of `runs`
independent chi-square samples, which is `chi2(runs * dof) / runs`. Independence
across Monte Carlo runs is exact, because each run has its own noise realisation.
Independence across time is not exact, because the estimation error is serially
correlated. The interval is therefore applied to the across-run average at each
time step, where it is valid, and the grand mean over time is reported against
that same interval as a summary. That summary is conservative: averaging
correlated samples cannot make the mean less variable than the interval assumes,
so the test will not falsely reject.

The verdict is taken from the grand mean and the fraction of time steps inside
the interval is reported alongside as supporting evidence. For a consistent
filter that fraction should sit near the confidence level, and it does: 97.4
percent for NEES on the correctly specified run.

The mean over runs is reported alongside the median over runs, because the mean
is not robust. In radar-only tracking a single run in forty that loses the target
raises the mean NEES from about 5.5 to 67.7 while the other thirty-nine are
unremarkable, and reading that mean as a statement about the typical run would be
wrong. A run whose time-averaged NEES exceeds ten times its degrees of freedom is
counted as a lost track and reported separately.

### Asynchronous fusion

Sensors are fused sequentially: each report is applied on its own as soon as it
is released, and the prediction covers whatever interval separates it from the
previous one. Because the sensors are conditionally independent given the state,
applying two reports with the same timestamp one after the other gives the same
posterior as stacking them into one taller measurement. The test suite verifies
this exactly for two linear sensors and to within the linearisation error for a
linear and a nonlinear sensor together.

Two out-of-order policies are implemented and both are stated explicitly rather
than left to emerge from the code. `BUFFER` holds arrivals in a reorder buffer
and releases them in timestamp order once the buffer's latency budget has
elapsed, which restores the correct order for every report whose latency is
within budget at the cost of that budget in output delay. `DISCARD` applies
reports in arrival order and drops any whose timestamp precedes the filter clock,
which adds no delay and loses information. With the measured latencies, the
buffer reproduces the undelayed answer exactly and discarding throws away 38
percent of the reports.

### When the unscented filter is worth its extra cost

This is the question the package was built to answer, and the answer is taken
from the measured numbers in the README results section rather than from the
usual claim that the unscented filter is simply better.

On the fused configuration, lidar and radar together on a turning target, the
unscented filter is 1.0 percent better in position RMSE and 0.8 percent better in
velocity RMSE than the extended filter, against a run-to-run standard deviation
of about 12 percent. Both filters are consistent, both keep all forty tracks, and
the unscented filter costs roughly a factor of two in wall clock time. On that
configuration it is not worth it. The lidar is linear and reports at 10 Hz, so
the covariance never has time to grow wide enough for the curvature of the polar
measurement to matter across it, which is precisely the condition under which a
Jacobian linearisation is a good approximation.

Removing the lidar changes the answer, and it changes it in the covariance rather
than in the accuracy. With radar alone at 13.3 Hz the accuracy gap is still only
6 percent, but the extended filter's NEES rises to 8.953 against an expected 5,
which classifies it as optimistic, and it loses 2 of 40 tracks. The unscented
filter stays at 5.318, is classified consistent, and loses none. Degrading the
radar further, to 2.5 Hz with a bearing standard deviation of 0.20 rad, gives the
same pattern: extended filter NEES 17.179 and one track lost, unscented filter
NEES 5.899 and none.

There is a limit to this, and the sweep shows where it is. At 5 Hz with a bearing
standard deviation of 0.10 rad both filters are optimistic and both lose a track,
at NEES 67.652 and 59.304. The unscented transform recovers the moments of a
nonlinearity accurately; it does not manufacture information. Once the
measurement is too poor and too infrequent to support the state, neither filter
has a trustworthy covariance and the answer is a better sensor or a better motion
model, not a better moment propagation.

The rule that falls out of these numbers is about the width of the covariance
relative to the curvature of the measurement, not about the filter. Where a
linear sensor keeps the covariance narrow, the two filters are interchangeable
and the extended one is cheaper. Where the only information is nonlinear and
arrives slowly enough for the covariance to grow between updates, but not so
slowly that the state stops being observable in practice, the extended filter's
covariance stops being trustworthy before its estimate stops being usable, and
that is the failure the extra cost buys protection against. The practical test is
not to argue about it: run both, measure NEES, and look at which one stays inside
its bounds.

## Rejected alternatives

### Augmented-state unscented filter

The unscented filter treats process noise as additive, adding `Q` to the
predicted covariance rather than carrying the noise in an augmented sigma point
set. The augmented form propagates the noise through the nonlinearity as well,
which is more accurate when the noise enters non-additively, as it does for CTRV
where the disturbance is a body-frame acceleration.

It was not chosen because it raises the sigma point count from `2 * n + 1` to
`2 * (n + q) + 1`, which for the five-dimensional CTRV state with two noise
sources is 15 points instead of 11, a 36 percent increase in the dominant cost of
the filter. The measured consistency of the additive form leaves nothing for the
augmented form to recover: NEES is 5.076 against an expected 5.0, and NIS is
1.963 and 3.000 against expected 2 and 3. Buying accuracy that the statistics say
is already there is not a trade worth making. If the process noise were larger
relative to the state, or the steps longer, this decision would need revisiting.

### Square root unscented filter

The square root form propagates a Cholesky factor of the covariance rather than
the covariance itself, using a QR update and a Cholesky rank one update. It
guarantees positive definiteness by construction and halves the condition number
of the quantity being propagated.

It was rejected because the problem does not need it. The covariance is checked
for symmetry and non-negative eigenvalues after every prediction and update
across an 800 step scenario in the test suite and never fails, and the default
sigma point weights are already all non-negative, which is where most
indefiniteness in an unscented filter originates. The square root form would add
a QR factorisation and a rank one update per step to solve a problem that is not
occurring, at a real cost in readability.

### Interacting multiple model estimator

The IMM (Blom and Bar-Shalom, 1988) runs a bank of filters with different motion
models and mixes them through a Markov transition matrix. It is the standard
answer to a target that alternates between straight and turning motion, and it
would beat any single model on such a target.

It was rejected as out of scope. The comparison this package exists to make is
between two ways of pushing a Gaussian through a nonlinearity, and an IMM would
run whichever of them was chosen inside each of its modes; it addresses a
different question. The motion model comparison in the results section shows what
the single-model choice costs on this scenario, which is the honest way to leave
the door open: CTRV gives 0.1183 m position RMSE against 0.1642 m for constant
velocity and 0.1594 m for constant acceleration, so a mode-switching target would
be where an IMM starts to pay.

### Out-of-sequence measurement retrodiction

The optimal treatment of a late measurement is neither buffering nor discarding
it. It is to retrodict the filter state back to the measurement time, fold the
measurement in there, and propagate forward again, which Bar-Shalom (2002) gives
in exact form for one lag.

It was rejected on complexity. The exact solution requires storing the state and
covariance history over the lag window plus the cross-covariance between the
current and retrodicted states, and it must be extended to multiple lags to be
useful with the latency distribution here. The reorder buffer achieves the same
answer to the last digit whenever the latency budget covers the arrival, which
the measured latencies do, and the only cost is output delay. For a system that
cannot afford 0.15 s of delay, retrodiction is the correct next step, and the
measured cost of the alternative, 6 percent in position error and 11 percent in
velocity error from discarding, is what it would have to beat.

### Numerical or automatic differentiation of the Jacobians

Finite differences would remove the hand-derived Jacobians and the possibility of
an error in them. They were rejected because the finite difference is exactly
what the test suite uses to check the analytic form, and using it in the
implementation as well would leave the check comparing a thing to itself. Beyond
that, a central difference costs `2 * n` model evaluations per Jacobian against
one closed-form expression, and it introduces a step size that has to be tuned
against truncation and rounding error.

### Eigendecomposition for the noise factors

The obvious way to draw from a positive semi-definite covariance is to
eigendecompose it and scale the eigenvectors. That was rejected for
reproducibility. The sign of an eigenvector is not determined by the problem, so
two LAPACK builds can legitimately return factors differing by a reflection, and
the same seed would then produce different data on a different machine, breaking
every regression test for a reason that has nothing to do with the code. Every
noise factor here is closed form instead: an explicit Cholesky expression for the
linear models and an explicit gain matrix for CTRV, each verified against the
covariance it claims to factor in the test suite.

## Known limitations

### Single target, no data association

There is one target and every measurement belongs to it. Real radar and lidar
return clutter and multiple objects, and the association problem is usually
harder than the filtering problem. Adding it would mean gating on the innovation
covariance, which this package already computes, followed by a nearest neighbour
or probabilistic data association step. Nothing in the current design obstructs
that, but nothing implements it either.

### The consistency intervals assume time independence in the summary

The per-time-step interval is exact, since it relies only on independence across
Monte Carlo runs. The grand mean over time is reported against the same interval,
and the estimation error is serially correlated, so that comparison is an
approximation. It errs in the safe direction, as argued above, but a filter that
is marginally inconsistent could be classified as consistent by it. A block
bootstrap over time would remove the assumption.

### The Cartesian NEES is a linearised projection

When the filter's motion model is the model that generated the truth, NEES is
computed in the filter's own state space and is exact. When they differ, as in
the mis-specified campaigns, the covariance is projected onto the Cartesian view
through the first-order Jacobian and the statistic inherits that approximation.
The statistic is labelled differently in the two cases so that no reader mistakes
one for the other, but the mis-specified numbers should be read as strong
qualitative evidence rather than as calibrated values.

### CTRV process noise is approximate in position

The two continuous white noise acceleration chains compose exactly in speed,
heading, and yaw rate, but the position contribution assumes the along-track
direction is fixed over the interval. The relative error is 6.1e-4 at the largest
step the measurement schedule produces, and it scales as the square of
`omega * dt`. At the yaw rates and step sizes used here it does not show up in
the consistency statistics; at high yaw rate with long steps it would, and the
filter would appear conservative in position. An exact treatment would integrate
the covariance through the rotating frame rather than freezing the heading.

### Radar geometry near the sensor

Range, bearing, and range rate are all degenerate as the target approaches the
sensor origin. The implementation floors the range at 1e-4 m to keep the
arithmetic finite, but a target that passes within a few metres of the sensor
will produce a badly conditioned measurement whatever the floor does.

An earlier version of the turning scenario did exactly that. Its nominal circular
path enclosed the sensor, so under the process noise the target could pass within
0.33 m of the origin, and 3 percent of seeds came inside 1 m. Every published
number was then partly a statement about a singularity rather than about a
filter. The scenario was moved onto a circle of radius 12.5 m centred 64 m away,
which puts the sensor well outside the path, and a test now asserts a minimum
range of 15 m over 200 seeds for all three scenarios so that a future edit to a
starting state cannot quietly reintroduce the problem. The measured worst
approach is 18.3 m for the turning target, 25.0 m for the straight target, and
23.9 m for the boundary crossing target.

### Initial transient in the consistency statistics

The across-run NEES sits above its upper bound for roughly the first 0.4 s of
each run, while the filter converges from its initial covariance. That is
expected behaviour for a filter starting from a deliberately loose prior, and it
is visible in the top panel of the figure that `examples/consistency_study.py`
writes. It is included in the grand mean rather than trimmed, which makes the
reported verdict slightly harder to pass than it would be with a burn-in period
removed.

### Simulated sensors only

The measurement models are idealised: additive zero-mean Gaussian noise with a
fixed diagonal covariance, no detection probability below one, no bias, no
range-dependent noise, no multipath, no extended-object effects. Every number in
the results section is a statement about this simulation and not about any real
sensor. The consistency machinery is the part that transfers directly, since NIS
requires no ground truth and can be computed against a live sensor feed.
