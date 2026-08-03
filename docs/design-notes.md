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
is not robust. Tracking the distant target on radar alone, the extended filter
loses eight runs in forty and those eight carry the mean NEES to 103.8 while the
median run sits at 10.9; reading that mean as a statement about the typical run
would be wrong in both directions at once. A run whose time-averaged NEES exceeds
ten times its degrees of freedom is counted as a lost track and reported
separately, so the count and the median together say what the mean cannot.

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

The same four-rung sensor ladder is run against two targets, because the answer
turns out to be a property of the pair rather than of the sensor.

On the base scenario, a target 50 to 70 m out turning at 0.8 rad/s, the two
filters are interchangeable on every rung. Fused they differ by 1.0 percent in
position RMSE, against a run-to-run standard deviation of about 12 percent. With
the lidar removed they still agree to within 1 percent, and both stay consistent
all the way down to a 2.5 Hz radar with a bearing standard deviation of 0.20 rad.
The unscented filter costs roughly a factor of two in wall clock time and buys
nothing measurable. The reason is that the covariance never grows wide enough,
relative to the curvature of the polar measurement, for the difference between a
Jacobian and a sigma point set to show up in it.

Moving the target out to 100 m and letting it leave changes the answer, and it
changes it in the covariance rather than in the accuracy. With both sensors
present the extended filter's NEES is 7.750 against an expected 5, which
classifies it as optimistic, while the unscented filter sits at 4.982 and is
consistent. Neither loses a track and the accuracy gap is 13 percent. The shape
matters more than the mean here and is in `docs/figures/nees-ekf-ukf.png`: the
extended filter enters at twenty times its expected value and needs more than a
second to fall inside its interval, spending 13.7 percent of the run above it
against the unscented filter's 3.4 percent. From the same loose prior, on the
same data, one of these filters can be believed immediately and the other cannot.

With radar alone the gap widens into lost tracks. At 13.3 Hz the extended filter
loses 8 of 40 and the unscented filter 2 of 40; at 5 Hz and a bearing standard
deviation of 0.10 rad, 4 against 1; at 2.5 Hz and 0.20 rad, 6 against 4. The
median run NEES, which is the robust statistic here because a single lost run
moves the mean by two orders of magnitude, runs 10.923, 9.095 and 8.159 for the
extended filter against 5.012, 5.637 and 5.533 for the unscented one.

There is a limit to this, and the sweep shows where it is. On the last two rungs
of the distant target both filters are classified optimistic and both lose
tracks. The unscented transform recovers the moments of a nonlinearity
accurately; it does not manufacture information. Once the measurement is too poor
and too infrequent to support the state, neither filter has a trustworthy
covariance and the answer is a better sensor or a better motion model, not a
better moment propagation.

The rule that falls out of these numbers is about the width of the covariance
relative to the curvature of the measurement, not about the filter and not about
the sensor on its own. Where the covariance stays narrow, whether because a
linear sensor keeps it narrow or because the target is close enough that the
polar measurement is nearly linear across it, the two filters are
interchangeable and the extended one is cheaper. Where the covariance is wide,
whether at initialisation or because the only information is nonlinear and
arrives slowly, the extended filter's covariance stops being trustworthy before
its estimate stops being usable, and that is the failure the extra cost buys
protection against. The practical test is not to argue about it: run both,
measure NEES, and look at which one stays inside its bounds.

An earlier version of this section reported the opposite split, with the fused
case interchangeable and the radar-only case separating the filters. That was
measured on a sweep scenario whose target could pass within half a metre of the
sensor, so the separation it found was substantially a statement about the polar
singularity. See the geometry entry under Known limitations for what was wrong
and how it is now prevented.

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

## Closed limitations

### The Cartesian NEES was a linearised projection

What it said. When the filter's motion model is the model that generated the
truth, NEES is computed in the filter's own state space and is exact. When they
differ, as in the mis-specified campaigns, the covariance was projected onto the
Cartesian view through the first-order Jacobian and the statistic inherited that
approximation, so the mis-specified numbers had to be read as qualitative
evidence rather than as calibrated values.

What was wrong with it. A Cartesian NEES divides the error by a matrix that is
supposed to be the covariance of that error. `J P J.T` is the first-order
approximation to that covariance and is exact only when `to_cartesian` is linear.
For the CTRV view, which resolves a polar velocity onto Cartesian axes, it is
not: at a heading standard deviation of 0.3 rad on a 14 m/s target it understates
the variance of the x velocity by 29.7 percent, and the statistic normalised by
it has expectation 5.514 rather than 4. A verdict taken against chi-square bounds
for four degrees of freedom is then wrong by 38 percent before the filter has
done anything.

What replaced it. Each motion model now returns the exact second moment of its
own Cartesian view in closed form, as `MotionModel.cartesian_moment`. For the
constant velocity and constant acceleration models the view is linear and the
answer is the identity or a selection, which is `J P J.T` exactly. For CTRV the
velocity is `v * exp(i * psi)` with `v` and `psi` jointly Gaussian, and every
entry of the answer reduces to an expectation of the form `E[X exp(i t psi)]`,
which has a closed form obtained by differentiating the joint moment generating
function. The moment is taken about the estimate the filter actually reports,
`to_cartesian(mean)`, rather than about the mean of the transformed distribution,
because that is the point the error is measured from and the two differ by the
bias the curvature introduces.

What it cost. About fifty lines in the model layer and a derivation that has to
be right, since an algebra error here would silently miscalibrate a statistic
rather than raise. The test suite therefore checks it against a 400000 sample
Monte Carlo projection that shares none of the algebra: the closed form lands
0.09 percent away in Frobenius norm, which is the sampling error, and the
Jacobian projection lands 10.2 percent away. Runtime cost is nil, since the
closed form is a handful of flops against the Jacobian product it replaces.
Nothing published changed, because every mis-specified campaign in this package
runs a constant velocity filter whose Cartesian view was already linear.

What remains. The projected error is not Gaussian even when the state is, so the
statistic has exactly the right expectation but is only approximately chi-square
distributed. The verdict is taken from the mean, which is now exact; the interval
around it is not. A filter compared in a space it does not work in is still being
asked a slightly different question from one compared natively, which is why the
two statistics are still labelled apart.

An unscented projection was tried first and rejected. Pushing the same sigma
point set the unscented filter already builds through `to_cartesian` is better
than the Jacobian and still not exact: on the same belief it lands 3.8 percent
away in Frobenius norm against the Jacobian's 10.2 percent, understating the
x velocity variance by 12.1 percent where the Jacobian understates it by 29.7
percent, and leaving the statistic with an expectation of 4.259 rather than 4.
It also costs a Cholesky factorisation and eleven projections per recorded step.
Replacing an approximation whose error is known with one whose error is merely
smaller, and paying more for it, is not a trade worth making when a closed form
exists.

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
which puts the sensor well outside the path, and a test asserts a minimum range
of 15 m over 200 seeds so that a future edit to a starting state cannot quietly
reintroduce the problem.

That fix was then found to be incomplete, and the way it was incomplete is worth
recording. The test covered the three scenarios defined in the scenarios module.
The regime sweep comparing the extended and unscented filters used a fourth
configuration, written as literals inside `examples/ekf_versus_ukf.py`, which
therefore no test ever saw. That configuration started 63 m out on a tangential
heading with a yaw disturbance an order of magnitude stronger than the base
scenario, and its nominal circle enclosed the sensor: the worst of 200 seeds came
within 0.21 m of the origin, and the worst of the forty seeds the published sweep
actually used came within 0.49 m. The comparison that section existed to make was
therefore measuring the polar singularity as much as it was measuring a filter,
and the conclusion it reached changed once the geometry was corrected.

The scenario now lives in the scenarios module as `distant_target`, starting
100 m out on an outward heading, and is covered by the same 200-seed test. The
measured worst approach is 18.3 m for the turning target, 25.0 m for the straight
target, 23.9 m for the boundary crossing target, and 35.2 m for the distant
target. The general lesson is the narrow one: a scenario that is not named in the
scenarios module is a scenario nothing checks, and the guard is only worth what
its coverage is.

### Initial transient in the consistency statistics

A filter starting from a deliberately loose prior can carry a covariance that is
badly wrong for the first fraction of a second and perfectly good afterwards. On
the base scenario there is no such transient to speak of: the correctly specified
filter spends 2.6 percent of steps above its upper bound across the whole run,
which is what a 95 percent interval predicts, and the top panel of
`docs/figures/nees-specification.png` shows the trace starting inside. An earlier
version of this entry claimed the statistic sat above its bound for the first
0.4 s of every run. It does not, and did not once the turning scenario was moved
off the sensor; the claim was a leftover.

On the distant scenario the transient is real and large enough to decide a
verdict, which is the entry in the results section worth reading twice: the
extended filter enters at twenty times its expected value and its grand mean is
above the interval because of that, not because of anything in the remaining nine
seconds.

Nothing is trimmed. The verdict describes the whole run the caller asked for,
and the fraction of steps above the upper bound is reported next to it so that a
transient and a persistent error are not read as the same thing. That fraction
is a summary and not a substitute for the trace, which is why
`docs/figures/nees-ekf-ukf.png` is in the README next to the numbers. A burn-in
parameter was considered and rejected: a window chosen from the same data the
verdict is then taken on is a selection on the outcome, and a fixed one would be
right for one scenario and wrong for the next.

### Simulated sensors only

The measurement models are idealised: additive zero-mean Gaussian noise with a
fixed diagonal covariance, no detection probability below one, no bias, no
range-dependent noise, no multipath, no extended-object effects. Every number in
the results section is a statement about this simulation and not about any real
sensor. The consistency machinery is the part that transfers directly, since NIS
requires no ground truth and can be computed against a live sensor feed.
