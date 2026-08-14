"""Phase 0.2 — measure the car instead of trusting the spec sheet.

    python -m tools.probe --headless

Writes ``artifacts/dynamics.json``. **Needs Isaac Sim and a CUDA GPU.**

Seven numbers, and one of them decides whether this whole approach can work:

1. **wheelbase** — from the articulation's own body positions, confirmed by
   driving a full-lock circle and measuring its radius.
2. **minimum turn radius**, ``L_wb / tan(0.488)``, compared against what the
   official track's corridor can actually deliver. If ``R_min`` is larger, the
   tightest corner cannot be taken by steering alone at any speed, a pure-pursuit
   teacher will fail Gate 1, and the remaining time has to come from rotating the
   car with the throttle — which is a different project. **This is the go/no-go.**
3. **forward acceleration** against speed, which falls off with the motor curve.
4. **braking deceleration**.
5. **lateral limit**, and whether it varies with steering angle.
6. **rollover threshold.** If the car tips before it slides, the grip limit is the
   wrong number everywhere and every speed target built on it is above what the
   car survives. Live possibility on a 1.27 kg chassis.
7. **steering lag** — control steps for the wheels to reach the commanded angle.

This file is only the shell: environment, stepping, and the verdict. All the
measurement logic lives in :mod:`tools.probe_analysis`, which imports without a
simulator and is therefore covered by ``tests/test_probe.py``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Measure the car's real limits.")
parser.add_argument("--out", default="artifacts/dynamics.json", help="Where to write.")
parser.add_argument(
    "--scale", type=float, default=None, help="Track scale factor (default: 40)."
)
parser.add_argument(
    "--allow-cpu", action="store_true", help="Run on CPU. Slow, and cars collide."
)
parser.add_argument(
    "--dump",
    default=None,
    metavar="PATH",
    help="Also write the raw per-step traces here, for diagnosing a measurement.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────────────────────────────────────────────────────────
#  Isaac Sim is up from here on.
# ─────────────────────────────────────────────────────────────────────────────

from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from lituanicax_sdk.vehicle import TIMING  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.harness import FixedPoses, make_env  # noqa: E402
from tools.probe_analysis import (  # noqa: E402
    PROBE_SCALE,
    TOTAL_STEPS,
    Recorder,
    analyse,
    build_script,
    layout,
    probe_track,
    straightest_point,
)
from tools.profile import DEFAULT_HALF_WIDTH_M, widest_achievable_radius  # noqa: E402


def report(measured, notes: dict, widest: float) -> int:
    """Print everything, and return the Gate 0 verdict as an exit status."""
    print()
    print(measured.describe())
    print()
    print(
        f"  wheelbase                  {measured.wheelbase_m:.4f} m "
        f"({notes['wheelbase_source']})"
    )
    print(
        f"  full-lock circle radius    {notes['full_lock_radius_m']:.3f} m "
        f"(fit rms {notes['full_lock_fit_rms_m'] * 1000:.0f} mm, implies "
        f"L_wb {notes['wheelbase_from_circle_m']:.3f} m)"
    )
    print(f"  minimum turn radius        {measured.r_min_m:.3f} m")
    print(f"  corridor can deliver       {widest:.3f} m on the official track")
    print(
        f"  acceleration               {measured.a_accel_m_s2:.2f} m/s^2 average to "
        f"80% of top speed ({notes['steps_to_80pc_top_speed']} steps)"
    )
    print(
        f"  braking                    {measured.a_brake_m_s2:.2f} m/s^2 "
        f"({notes['steps_to_stop']} steps to a stop)"
    )
    print(f"  top speed reached          {notes['top_speed_reached_m_s']:.2f} m/s")
    print(f"  lateral limit              {measured.a_lat_max_m_s2:.2f} m/s^2")
    if measured.rollover_a_lat_m_s2 is not None:
        print(
            "  ROLLED OVER at             "
            f"{measured.rollover_a_lat_m_s2:.2f} m/s^2 lateral"
        )
    else:
        print("  rollover                   never — the car slid before it tipped")
    print(
        f"  steering lag               {measured.steer_lag_steps:.1f} steps "
        f"({measured.steer_lag_steps * TIMING.step_dt * 1000:.0f} ms)"
    )

    print()
    for entry in notes["lateral_by_steer"]:
        tipped = (
            f", tipped at step {entry['tipped_at_step']}"
            if "tipped_at_step" in entry
            else ""
        )
        print(
            f"    steer {entry['steer']:.2f} -> "
            f"a_lat {entry['a_lat']:.2f} m/s^2{tipped}"
        )
    if notes["a_lat_varies_with_steer"]:
        print(
            f"    the limit varies by {notes['a_lat_spread_m_s2']:.1f} m/s^2 across "
            "steering angles, so one\n    number is a simplification the search "
            "will have to absorb."
        )
    if "wheelbase_disagreement" in notes:
        print(f"\n  NOTE: {notes['wheelbase_disagreement']}")

    print()
    if measured.r_min_m <= widest:
        print(
            f"  GATE 0 PASS: R_min {measured.r_min_m:.3f} m fits inside the "
            f"{widest:.3f} m the corridor can\n               deliver, so a "
            "steerable racing line exists."
        )
        verdict = 0
    else:
        print(
            f"  GATE 0 FAIL: R_min {measured.r_min_m:.3f} m exceeds the "
            f"{widest:.3f} m the corridor can\n               deliver. The tightest "
            "corner cannot be taken by steering alone at any\n               speed, "
            "so a pure-pursuit teacher will not complete a lap — the car\n"
            "               has to be rotated with the throttle, which this control "
            "law\n               cannot discover. Read this before debugging Gate 1."
        )
        verdict = 3

    if measured.rollover_limited:
        print(
            "\n  NOTE: the car tips before it slides, so the rollover threshold "
            "replaces the\n        grip limit everywhere and every speed target is "
            "built on it."
        )
    return verdict


def main() -> int:
    scale = args_cli.scale if args_cli.scale is not None else PROBE_SCALE
    track = probe_track(scale)

    geometry = TrackGeometry.from_track(
        Track(track, device="cpu"), spacing_m=0.05 * scale
    )
    straight = straightest_point(geometry, run_m=60.0)
    print(f"[probe] track scaled {scale:g}x -> {geometry.length:.0f} m loop")
    print(
        f"[probe] straight run from ({straight[0]:.1f}, {straight[1]:.1f}) "
        f"heading {math.degrees(straight[2]):.0f} deg"
    )

    lateral_envs, lag_env, num_envs = layout()

    # Every measurement starts on the straight: it is the only place with room for
    # a car to do something violent without meeting a wall.
    env = make_env(
        num_envs=num_envs,
        track=track,
        spawn=FixedPoses([straight] * num_envs),
        # Nothing may end an episode — the probes need the whole trace, including
        # what happens after the car tips over.
        official_rules=False,
        stall_rule=False,
        episode_length_s=(TOTAL_STEPS + 10) * TIMING.step_dt,
        allow_cpu=args_cli.allow_cpu,
    )

    script = build_script(num_envs, lateral_envs, lag_env)
    recorder = Recorder()

    env.reset()
    recorder.add(env.latest_car)
    for step in range(TOTAL_STEPS):
        env.step(script(step).to(env.device))
        recorder.add(env.latest_car)

    if args_cli.dump:
        import numpy as np  # noqa: PLC0415

        path = Path(args_cli.dump)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            position=recorder.positions(),
            lateral_envs=np.array(list(lateral_envs)),
            lag_env=np.array(lag_env),
            **{field: recorder.array(field) for field in Recorder.FIELDS},
        )
        print(f"[probe] raw traces -> {path}")

    measured, notes = analyse(recorder, env.robot.data, lateral_envs, lag_env)

    # The go/no-go needs the *official* track's corridor, not the scaled one.
    official_geometry = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"))
    widest, _ = widest_achievable_radius(
        official_geometry, half_width=DEFAULT_HALF_WIDTH_M
    )

    verdict = report(measured, notes, widest)
    path = measured.save(args_cli.out)
    print(f"\n[probe] written to {path}")
    return verdict


if __name__ == "__main__":
    status = main()
    try:
        simulation_app.close()
    except Exception as error:  # pragma: no cover — teardown is not the measurement
        print(f"[probe] the simulator did not shut down cleanly: {error}")
    raise SystemExit(status)
