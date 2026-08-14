"""How much lateral acceleration can this car actually hold, and until when?

    python -m tools.grip --headless

**Needs Isaac Sim and a CUDA GPU.** Writes ``artifacts/grip.json``.

:mod:`tools.probe` answers seven questions about the car and gets one of them
wrong in a way that matters more than the other six. It reports
``a_lat_max = 8.83 m/s²`` as the cornering limit, having taken the highest
lateral acceleration the car reached at a steering angle where it did not tip.
But at those angles the car was **steering**-limited, not grip-limited — the
servo is effort-limited and could not hold more lock — so 8.83 is a number the
car happened to stop at, not one it could not exceed. The same run saw
12.22 m/s² at 0.8 lock before tipping.

That matters because 8.83 sets every corner speed in the quasi-static profile,
and the profile is what the controller chases. On the official track the model
says 15.0 s at 8.83 and 13.3 at 13.2 — and a lap of 14.3 s is known to be
achievable, so the model is not merely conservative, it is wrong.

The manoeuvre here isolates the question. Each car holds one steering angle and
accelerates *gently* — throttle ramped over hundreds of steps rather than pinned
— so lateral acceleration creeps up through the whole range instead of jumping
to wherever the car happens to settle at full power. Whatever ends the run, grip
or rollover, is then recorded with the lateral acceleration that caused it.

The distinction the results draw:

* **slid** — the car stopped gaining lateral acceleration while the steering was
  still held. That is the grip limit, and it is a hard ceiling.
* **tipped** — ``up_axis`` collapsed. That is the rollover threshold, and it is a
  ceiling only for *sustained* cornering; a brief peak through a short corner can
  exceed it, which is one reason a real lap beats the quasi-static estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Measure the real cornering limit.")
parser.add_argument("--out", default="artifacts/grip.json")
parser.add_argument("--scale", type=float, default=None)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--allow-cpu", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────────────────────────────────────────────────────────
#  Isaac Sim is up from here on.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lituanicax_sdk.track import Track  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.harness import FixedPoses, make_env  # noqa: E402
from tools.probe_analysis import (  # noqa: E402
    PROBE_SCALE,
    Recorder,
    impact_step,
    probe_track,
    straightest_point,
    tip_step,
)

#: Steering angles held, as a fraction of full lock. Denser at the top than the
#: seven-number probe, because that is where the answer lives.
LEVELS = (0.35, 0.5, 0.65, 0.8, 0.9, 1.0)

#: Steps spent ramping the throttle from nothing to full. Long, deliberately:
#: the whole point is to creep up on the limit rather than arrive at it.
RAMP_STEPS = 700


def script(step: int, num_envs: int) -> torch.Tensor:
    actions = torch.zeros(num_envs, 2)
    throttle = min(step / RAMP_STEPS, 1.0)
    for index, level in enumerate(LEVELS):
        actions[index, 0] = throttle
        actions[index, 1] = level
    return actions


def analyse(recorder: Recorder) -> tuple[dict, list[dict]]:
    speed = recorder.array("speed_forward")
    yaw_rate = recorder.array("yaw_rate")
    up_axis = recorder.array("up_axis")
    lateral_speed = recorder.array("speed_lateral")
    steer_angle = recorder.array("steer_angle")
    a_lat = np.abs(speed * yaw_rate)

    rows = []
    for index, level in enumerate(LEVELS):
        tipped = tip_step(up_axis[:, index])
        hit = impact_step(speed[:, index])
        ended = min(x for x in (tipped, hit, len(speed)) if x is not None)
        window = slice(max(ended - 30, 0), max(ended - 2, 1))

        peak = float(np.max(a_lat[: max(ended - 2, 1), index]))
        held = float(np.median(a_lat[window, index]))
        rows.append(
            {
                "steer": level,
                "peak_a_lat": peak,
                "held_a_lat": held,
                "speed_m_s": float(np.median(speed[window, index])),
                "drift_m_s": float(np.median(np.abs(lateral_speed[window, index]))),
                "wheels_rad": float(np.median(np.abs(steer_angle[window, index]))),
                "ended": (
                    "tipped"
                    if tipped is not None
                    else ("hit a wall" if hit is not None else "survived")
                ),
                "ended_at_step": int(ended),
            }
        )

    tipped_at = [r["peak_a_lat"] for r in rows if r["ended"] == "tipped"]
    survived = [r["held_a_lat"] for r in rows if r["ended"] == "survived"]
    summary = {
        "rollover_a_lat_m_s2": min(tipped_at) if tipped_at else None,
        "highest_sustained_m_s2": max(survived) if survived else None,
        "highest_seen_m_s2": max(r["peak_a_lat"] for r in rows),
        "levels": rows,
    }
    return summary, rows


def main() -> int:
    scale = args_cli.scale if args_cli.scale is not None else PROBE_SCALE
    track = probe_track(scale)
    geometry = TrackGeometry.from_track(
        Track(track, device="cpu"), spacing_m=0.05 * scale
    )
    straight = straightest_point(geometry, run_m=60.0)

    env = make_env(
        num_envs=len(LEVELS),
        track=track,
        spawn=FixedPoses([straight] * len(LEVELS)),
        official_rules=False,
        stall_rule=False,
        episode_length_s=(args_cli.steps + 10) / 30.0,
        allow_cpu=args_cli.allow_cpu,
    )

    recorder = Recorder()
    env.reset()
    recorder.add(env.latest_car)
    for step in range(args_cli.steps):
        env.step(script(step, len(LEVELS)).to(env.device))
        recorder.add(env.latest_car)

    summary, rows = analyse(recorder)

    print("\n  steering   held a_lat   peak a_lat   speed   drift   wheels   ended")
    for row in rows:
        print(
            f"  {row['steer']:>8.2f} {row['held_a_lat']:>11.2f} "
            f"{row['peak_a_lat']:>12.2f} {row['speed_m_s']:>7.2f} "
            f"{row['drift_m_s']:>7.2f} {row['wheels_rad']:>8.3f}   {row['ended']}"
        )

    print()
    if summary["rollover_a_lat_m_s2"] is not None:
        print(
            f"  Tips at {summary['rollover_a_lat_m_s2']:.2f} m/s^2 sustained. That is "
            "the ceiling for a long\n  corner; a short one can exceed it, because "
            "rolling takes time as well as force."
        )
    if summary["highest_sustained_m_s2"] is not None:
        print(
            f"  Held {summary['highest_sustained_m_s2']:.2f} m/s^2 without ending the "
            "run."
        )
    print(f"  Highest seen at any moment: {summary['highest_seen_m_s2']:.2f} m/s^2.")
    print(
        "\n  tools.probe reports a_lat_max from sustained cornering only, and the\n"
        "  quasi-static profile is built on it. Anything here above that figure is\n"
        "  corner speed the model is leaving on the table."
    )

    path = Path(args_cli.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n[grip] written to {path}")
    return 0


if __name__ == "__main__":
    status = main()
    import threading

    closing = threading.Thread(target=simulation_app.close, daemon=True)
    closing.start()
    closing.join(timeout=30.0)
    sys.stdout.flush()
    os._exit(status)
