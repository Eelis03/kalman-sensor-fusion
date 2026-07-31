"""Asynchronous arrival and out-of-order handling.

Lidar and radar report at 10 Hz and 13.33 Hz on an unrelated phase, and each
report reaches the fusion node after a transport delay. The radar delay is the
longer one, so a radar report stamped at ``t`` regularly arrives after a lidar
report stamped later than ``t`` and the stream is genuinely out of order.

The scenario draws its truth, its measurement noise, and its transport latency
from separate random streams, so attaching latency changes only when reports
arrive and not what they say. The comparison below is therefore between policies
and nothing else.

    uv run python examples/asynchronous_fusion.py
    uv run python examples/asynchronous_fusion.py --quick
"""

from __future__ import annotations

import argparse

import numpy as np

from sensor_fusion.algorithm import UnscentedKalmanFilter
from sensor_fusion.analysis.metrics import summarize
from sensor_fusion.pipeline.fusion import FusionSettings, matched_belief, run_filter
from sensor_fusion.pipeline.scenarios import turning_target, with_latency
from sensor_fusion.pipeline.simulator import ScenarioConfig, simulate
from sensor_fusion.pipeline.trace import OutOfOrderPolicy

POLICIES = (
    ("no delay, reorder buffer", None, FusionSettings(OutOfOrderPolicy.BUFFER, 0.15)),
    ("delayed, reorder buffer 0.15 s", True, FusionSettings(OutOfOrderPolicy.BUFFER, 0.15)),
    ("delayed, reorder buffer 0.00 s", True, FusionSettings(OutOfOrderPolicy.BUFFER, 0.0)),
    ("delayed, discard late reports", True, FusionSettings(OutOfOrderPolicy.DISCARD)),
)


def main(argv: list[str] | None = None) -> int:
    """Run the policy comparison and print the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    args = parser.parse_args(argv)

    runs = 3 if args.quick else args.runs
    steps = 200 if args.quick else args.steps
    base = turning_target(steps=steps)
    delayed = with_latency(base, lidar_latency=0.02, radar_latency=0.09, jitter=0.01)
    model = base.truth_model

    print(f"turning target, {runs} runs of {steps} base steps, base seed {args.seed}")
    print(
        f"lidar every {base.schedules[0].period_steps} base steps, radar every "
        f"{base.schedules[1].period_steps}, offset {base.schedules[1].offset_steps}"
    )
    print("mean lidar latency 0.02 s, mean radar latency 0.09 s, jitter 0.01 s\n")

    for label, use_latency, settings in POLICIES:
        config: ScenarioConfig = delayed if use_latency else base
        position: list[float] = []
        velocity: list[float] = []
        processed: list[int] = []
        discarded: list[int] = []
        inversions: list[int] = []
        for index in range(runs):
            scenario = simulate(config, seed=args.seed + index)
            trace = run_filter(
                UnscentedKalmanFilter(model), scenario, matched_belief(scenario), settings
            )
            summary = summarize(trace)
            position.append(summary.position_rmse)
            velocity.append(summary.velocity_rmse)
            processed.append(trace.processed)
            discarded.append(trace.discarded)
            inversions.append(trace.out_of_order_arrivals)
        print(
            f"{label:32s} position RMSE {np.mean(position):.4f} +/- {np.std(position):.4f} m, "
            f"velocity RMSE {np.mean(velocity):.4f} +/- {np.std(velocity):.4f} m/s"
        )
        print(
            f"{'':32s} out-of-order arrivals {np.mean(inversions):.1f}, "
            f"processed {np.mean(processed):.1f}, discarded {np.mean(discarded):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
