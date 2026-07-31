"""Tier one: simulator determinism, asynchronous scheduling, and policy behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_fusion.algorithm.ukf import UnscentedKalmanFilter
from sensor_fusion.model.motion import (
    ConstantAcceleration,
    ConstantTurnRate,
    ConstantVelocity,
    MotionModel,
)
from sensor_fusion.pipeline.fusion import (
    FusionSettings,
    InitialUncertainty,
    matched_belief,
    run_filter,
)
from sensor_fusion.pipeline.scenarios import (
    BASE_DT,
    LIDAR_PERIOD_STEPS,
    MINIMUM_SAFE_RANGE,
    RADAR_OFFSET_STEPS,
    RADAR_PERIOD_STEPS,
    boundary_crossing_target,
    straight_target,
    turning_target,
    with_latency,
)
from sensor_fusion.pipeline.simulator import Scenario, ScenarioConfig, simulate
from sensor_fusion.pipeline.trace import OutOfOrderPolicy


def _delayed_pair() -> tuple[Scenario, Scenario]:
    """Return the same scenario without and with transport latency."""
    base = turning_target(steps=1200)
    return simulate(base, seed=21), simulate(with_latency(base), seed=21)


class TestSimulator:
    """Ground truth generation must be deterministic and correctly scheduled."""

    def test_same_seed_gives_identical_data(self) -> None:
        """Two runs of one seed agree bit for bit."""
        first = simulate(turning_target(steps=400), seed=17)
        second = simulate(turning_target(steps=400), seed=17)
        assert np.array_equal(first.truth_states, second.truth_states)
        assert all(
            np.array_equal(left.value, right.value)
            for left, right in zip(first.measurements, second.measurements, strict=True)
        )

    def test_different_seeds_give_different_data(self) -> None:
        """Independent realisations are actually independent."""
        first = simulate(turning_target(steps=400), seed=17)
        second = simulate(turning_target(steps=400), seed=18)
        assert not np.allclose(first.truth_states, second.truth_states)

    def test_measurement_times_lie_on_the_truth_grid(self) -> None:
        """Every measurement indexes a truth sample exactly, with no interpolation."""
        scenario = simulate(turning_target(steps=400), seed=2)
        for measurement in scenario.measurements:
            assert measurement.time == pytest.approx(
                scenario.times[measurement.truth_index], abs=1e-12
            )

    def test_the_two_sensors_never_report_together(self) -> None:
        """Coprime periods with an offset make the schedule genuinely asynchronous."""
        scenario = simulate(turning_target(steps=2000), seed=2)
        lidar = {m.time for m in scenario.measurements if m.sensor_name == "lidar"}
        radar = {m.time for m in scenario.measurements if m.sensor_name == "radar"}
        assert not lidar & radar
        gaps = np.diff(np.array(sorted(lidar | radar)))
        assert float(np.max(gaps)) > float(np.min(gaps)), "gaps must vary"

    def test_measurement_counts_follow_the_schedule(self) -> None:
        """Counts are what the periods and offsets imply."""
        steps = 2000
        scenario = simulate(turning_target(steps=steps), seed=2)
        lidar = sum(1 for m in scenario.measurements if m.sensor_name == "lidar")
        radar = sum(1 for m in scenario.measurements if m.sensor_name == "radar")
        assert lidar == len(range(0, steps + 1, LIDAR_PERIOD_STEPS))
        assert radar == len(range(RADAR_OFFSET_STEPS, steps + 1, RADAR_PERIOD_STEPS))

    def test_latency_changes_arrival_but_not_content(self) -> None:
        """Separate random streams keep the measurement values fixed under latency."""
        plain, delayed = _delayed_pair()
        assert np.array_equal(plain.truth_states, delayed.truth_states)
        by_key = {(m.sensor_name, m.time): m.value for m in plain.measurements}
        for measurement in delayed.measurements:
            assert np.array_equal(
                by_key[(measurement.sensor_name, measurement.time)], measurement.value
            )
        assert any(m.arrival_time > m.time for m in delayed.measurements)

    def test_process_noise_can_be_switched_off(self) -> None:
        """A noiseless truth follows the model exactly."""
        config = straight_target(steps=100)
        noiseless = ScenarioConfig(
            truth_model=config.truth_model,
            initial_mean=config.initial_mean,
            initial_cov=np.zeros((4, 4)),
            schedules=config.schedules,
            base_dt=config.base_dt,
            steps=config.steps,
            process_noise=False,
        )
        scenario = simulate(noiseless, seed=1)
        expected = np.asarray(config.initial_mean, dtype=np.float64).copy()
        assert np.allclose(scenario.truth_states[0], expected, atol=1e-14)
        for index in range(config.steps):
            expected = config.truth_model.predict(expected, BASE_DT)
            assert np.allclose(scenario.truth_states[index + 1], expected, atol=1e-12)

    def test_invalid_configuration_is_rejected(self) -> None:
        """Configuration errors surface at construction, not mid-run."""
        config = straight_target(steps=10)
        with pytest.raises(ValueError, match="at least one sensor"):
            ScenarioConfig(
                truth_model=config.truth_model,
                initial_mean=config.initial_mean,
                initial_cov=config.initial_cov,
                schedules=(),
            )


class TestScenarioGeometry:
    """The radar geometry must never approach its singularity."""

    @pytest.mark.parametrize(
        "config",
        [straight_target(steps=2000), turning_target(steps=2000), boundary_crossing_target()],
        ids=["straight", "turning", "boundary"],
    )
    def test_target_stays_clear_of_the_sensor(self, config: ScenarioConfig) -> None:
        """No seed brings the target within the safe range of the origin.

        Range, bearing, and range rate are all degenerate at the sensor origin.
        A scenario whose target passes close to it measures the behaviour of that
        singularity as much as the behaviour of the filter, so every published
        number would be partly a statement about the geometry. Two hundred seeds
        are checked rather than the handful the examples use, because the
        approach distance depends on the process noise realisation and the worst
        seed is what matters.
        """
        closest = np.inf
        for seed in range(200):
            cartesian = simulate(config, seed=seed).truth_cartesian
            closest = min(closest, float(np.min(np.hypot(cartesian[:, 0], cartesian[:, 1]))))
        assert closest > MINIMUM_SAFE_RANGE


class TestOutOfOrderPolicies:
    """Late reports must be handled by a stated policy, not by accident."""

    def test_the_stream_is_genuinely_out_of_order(self) -> None:
        """Guard the guard: latency must actually reorder the arrivals."""
        _, delayed = _delayed_pair()
        newest = -np.inf
        inversions = 0
        for measurement in delayed.measurements:
            if measurement.time < newest:
                inversions += 1
            newest = max(newest, measurement.time)
        assert inversions > 20

    def test_a_sufficient_buffer_recovers_the_undelayed_answer(self) -> None:
        """With a budget covering every latency the buffer is lossless.

        The reordered run must reproduce the run with no latency exactly, since
        the measurement values are identical and the buffer restores their order.
        """
        plain, delayed = _delayed_pair()
        model = plain.config.truth_model
        settings = FusionSettings(OutOfOrderPolicy.BUFFER, latency_budget=0.15)
        reference = run_filter(
            UnscentedKalmanFilter(model), plain, matched_belief(plain), settings
        )
        reordered = run_filter(
            UnscentedKalmanFilter(model), delayed, matched_belief(delayed), settings
        )
        assert reordered.discarded == 0
        assert reordered.processed == reference.processed
        assert np.allclose(reference.times, reordered.times, atol=1e-12)
        for left, right in zip(reference.records, reordered.records, strict=True):
            assert np.allclose(left.mean, right.mean, atol=1e-12)

    def test_discarding_loses_reports(self) -> None:
        """The cheap policy drops the late reports."""
        _, delayed = _delayed_pair()
        model = delayed.config.truth_model
        buffered = run_filter(
            UnscentedKalmanFilter(model),
            delayed,
            matched_belief(delayed),
            FusionSettings(OutOfOrderPolicy.BUFFER, 0.15),
        )
        dropped = run_filter(
            UnscentedKalmanFilter(model),
            delayed,
            matched_belief(delayed),
            FusionSettings(OutOfOrderPolicy.DISCARD),
        )
        assert dropped.discarded > 0
        assert dropped.processed < buffered.processed
        assert dropped.out_of_order_arrivals == buffered.out_of_order_arrivals

    def test_the_filter_clock_never_moves_backwards(self) -> None:
        """Whatever the policy, updates are applied in non-decreasing time order."""
        _, delayed = _delayed_pair()
        model = delayed.config.truth_model
        for policy in (OutOfOrderPolicy.BUFFER, OutOfOrderPolicy.DISCARD):
            trace = run_filter(
                UnscentedKalmanFilter(model),
                delayed,
                matched_belief(delayed),
                FusionSettings(policy, 0.15),
            )
            assert np.all(np.diff(trace.times) >= -1e-12)

    def test_sensor_selection_filters_the_stream(self) -> None:
        """A single-sensor baseline sees only that sensor."""
        scenario = simulate(turning_target(steps=800), seed=4)
        model = scenario.config.truth_model
        for name in ("lidar", "radar"):
            trace = run_filter(
                UnscentedKalmanFilter(model),
                scenario,
                matched_belief(scenario),
                FusionSettings(sensors=(name,)),
            )
            assert trace.sensor_names == (name,)


class TestInitialUncertainty:
    """Seeding a filter whose model differs from the truth model."""

    @pytest.mark.parametrize(
        "model",
        [ConstantVelocity(), ConstantAcceleration(), ConstantTurnRate()],
        ids=lambda m: m.name,
    )
    def test_seed_round_trips_through_the_cartesian_view(self, model: MotionModel) -> None:
        """The seeded mean reproduces the Cartesian state it was built from."""
        cartesian = np.array([14.0, -6.0, 3.0, 8.0])
        state = InitialUncertainty().build(model, cartesian)
        assert np.allclose(model.to_cartesian(state.mean), cartesian, atol=1e-12)
        assert state.cov.shape == (model.dim, model.dim)
        assert float(np.min(np.linalg.eigvalsh(state.cov))) > 0.0
