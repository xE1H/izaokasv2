"""The official evaluation. This is how a team's solution is scored.

    evaluate                                   # newest checkpoint, 100 agents
    evaluate --checkpoint logs/<run>/model_1000.pt
    evaluate --agents 100 --batch-size 25      # if 25 cars do not fit, lower it
    evaluate --spawn-presets                   # the track's preset points, not (0, 0)

    python -m lituanicax_sdk.evaluate --help   # without the helper script

**The format.** One hundred agents each get a single attempt. An agent is
placed on the official track and drives until one of three things happens:

* it **completes a lap** — its time is recorded, the episode ends on the spot,
  and that is its result;
* it **crashes** — hits a wall or rolls over, freezes, and the attempt ends;
* it **stalls** — stays barely moving too long, as a car pinned against a wall
  does, and the attempt ends.

The fastest of the hundred laps is the team's submission. Not the mean, not the
median: a team is judged on the best lap it can produce, the same way a
qualifying session works.

Every agent starts at the **world origin (0, 0)**, facing along the track, so
all hundred attempts begin from the same point. The origin sits on the
centerline a little before the start/finish line, so an attempt is an out-lap
of the short stretch to the line and then one timed lap back to it. Pass
``--spawn-presets`` to start from the track's preset points instead.

**What is fixed, and why.** The evaluation runs under
:mod:`lituanicax_sdk.rules`: crashing into a wall or rolling over ends the
attempt, a crashed car freezes rather than scraping onward, and a car that
stops making progress is cut off rather than sitting out the attempt window.
Your own ``compute_terminations`` is *not* used — otherwise a team with an
aggressive stall rule and a team with none would be running different
sessions. Your observations obviously still are: the policy could not run
otherwise.

**Batching.** A hundred cars will not fit on an 8 GB card alongside the track,
so they run in rounds of ``--batch-size``. The rounds are independent attempts
in one simulator session; the result is the same as running all hundred at
once, and it fits.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Officially evaluate a trained policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="team_solution",
        help="Python module that registers the task (default: team_solution).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="LituanicaX-Race-Team-v0",
        help="Registered task id to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Policy to evaluate (default: the most recent in logs/).",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=100,
        help="How many independent attempts to run (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="How many attempts to simulate at once. Lower this if you run out "
        "of GPU memory (default: 25).",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="official",
        help="Which official track to evaluate on.",
    )
    parser.add_argument(
        "--spawn-presets",
        action="store_true",
        default=False,
        help="Start from the track's preset points instead of the world origin.",
    )
    parser.add_argument(
        "--attempt-seconds",
        type=float,
        default=90.0,
        help="How long a single attempt may last before it counts as a failure.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to write the JSON result (default: next to the checkpoint).",
    )
    return parser


parser = build_parser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Isaac Sim's launcher inspects argv; nothing here is meant for Hydra.
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────────────────────────────────────────────────────────
#  Everything below runs with Isaac Sim already up.
# ─────────────────────────────────────────────────────────────────────────────

from typing import cast  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectRLEnv  # noqa: E402
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from lituanicax_sdk import (  # noqa: E402
    RaceEnv,
    RaceEnvCfg,
    sdk_fingerprint,
    tracks,
    verify_integrity,
)
from lituanicax_sdk.runs import find_checkpoint  # noqa: E402
from lituanicax_sdk.spawn import OriginSpawnManager  # noqa: E402

#: A failed attempt has no lap time.
NO_LAP = float("inf")


def main() -> None:
    modified = verify_integrity()
    importlib.import_module(args_cli.solution)  # registers the task

    if not tracks.is_official(args_cli.track):
        raise SystemExit(
            f"'{args_cli.track}' is not an official track. A submission is only "
            f"meaningful on {sorted(tracks.OFFICIAL_TRACKS)}."
        )

    # The registry is untyped — it hands back whatever the task registered.
    # Every task the evaluator can score is a RaceEnv one, so say so.
    env_cfg = cast(
        RaceEnvCfg, load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    )
    agent_cfg = cast(
        RslRlOnPolicyRunnerCfg,
        load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"),
    )

    batch_size = min(args_cli.batch_size, args_cli.agents)
    rounds = math.ceil(args_cli.agents / batch_size)

    # ── The evaluation conditions, identical for every team ───────────────
    env_cfg.track = tracks.get(args_cli.track)
    env_cfg.scene.num_envs = batch_size
    env_cfg.episode_length_s = args_cli.attempt_seconds
    env_cfg.enforce_official_rules = True
    env_cfg.terminate_on_lap = True
    env_cfg.official_stall_rule = True
    env_cfg.seed = 0
    if not args_cli.spawn_presets:
        # All attempts from the same point, so the hundred are a fair sample
        # of the policy rather than of the spawn distribution.
        env_cfg.spawn_manager = OriginSpawnManager()
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    checkpoint = find_checkpoint(args_cli.checkpoint)
    print(f"\n[evaluate] policy    : {checkpoint}")
    print(f"[evaluate] track     : {args_cli.track}")
    spawn = "preset points" if args_cli.spawn_presets else "world origin (0, 0)"
    print(f"[evaluate] spawn     : {spawn}")
    print(f"[evaluate] attempts  : {args_cli.agents} "
          f"({rounds} rounds of {batch_size})")
    print(f"[evaluate] sdk       : {sdk_fingerprint()}\n")

    env = RslRlVecEnvWrapper(
        cast(DirectRLEnv, gym.make(args_cli.task, cfg=env_cfg, render_mode=None)),
        clip_actions=agent_cfg.clip_actions,
    )

    runner = OnPolicyRunner(
        env, class_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    lap_times: list[float] = []
    with torch.inference_mode():
        for index in range(rounds):
            remaining = args_cli.agents - len(lap_times)
            times = run_round(
                env, policy, index + 1, rounds, min(batch_size, remaining)
            )
            lap_times.extend(times[:remaining])

    report = build_report(
        lap_times, checkpoint, modified, cast(RaceEnv, env.unwrapped)
    )
    print_report(report)
    write_report(report, checkpoint)

    env.close()


def run_round(
    env: RslRlVecEnvWrapper, policy, index: int, rounds: int, count: int
) -> list[float]:
    """Run one batch of attempts to completion. Returns a lap time per agent.

    Every car drives until it completes a lap or fails. Once an agent's attempt
    is over its result is banked and it is ignored for the rest of the round —
    Isaac Lab respawns it automatically, but that second life does not count.
    One attempt per agent is the whole point of the format.
    """
    race = cast(RaceEnv, env.unwrapped)
    device = race.device
    num_envs = race.num_envs
    timer = race.lap_timer
    step_dt = race.step_dt
    max_steps = int(args_cli.attempt_seconds / step_dt) + 2

    obs, _ = env.reset()
    settled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    result = torch.full((num_envs,), NO_LAP, device=device)

    for _ in range(max_steps):
        obs, _, dones, _ = env.step(policy(obs))

        # A lap completing this step. Read it before anything else, because a
        # car that also terminated has already been respawned by now — but
        # `just_finished` and `finished_laps_s` still describe the step we
        # just took, whereas `last_lap_time_s` was cleared for respawned cars.
        lapped = timer.just_finished & ~settled
        if lapped.any():
            lap_times = torch.full((num_envs,), NO_LAP, device=device)
            lap_times[timer.just_finished] = timer.finished_laps_s
            result[lapped] = lap_times[lapped]
            settled |= lapped

        # Anything else that ended the episode is a failed attempt: a crash, a
        # stall, a roll-over, or running out of time.
        settled |= dones.bool() & ~settled

        if bool(settled.all()):
            break

    times = sorted(float(t) for t in result[:count])
    finished = [t for t in times if t != NO_LAP]
    best = f"{finished[0]:.3f} s" if finished else "—"
    print(
        f"  round {index}/{rounds}:  {len(finished):3d}/{count} completed a lap"
        f"   best {best}"
    )
    return times


def build_report(
    lap_times: list[float], checkpoint, modified: list[str], env: RaceEnv
) -> dict:
    finished = sorted(t for t in lap_times if t != NO_LAP)
    attempts = len(lap_times)
    return {
        # The submission.
        "best_lap_time_s": finished[0] if finished else None,
        # Context for it.
        "attempts": attempts,
        "laps_completed": len(finished),
        "completion_rate": len(finished) / attempts if attempts else 0.0,
        "median_lap_time_s": finished[len(finished) // 2] if finished else None,
        "worst_lap_time_s": finished[-1] if finished else None,
        "all_lap_times_s": finished,
        "track": args_cli.track,
        "track_length_m": env.track.track_length,
        "checkpoint": str(checkpoint),
        "sdk_fingerprint": sdk_fingerprint(),
        "sdk_modified": modified,
    }


def print_report(report: dict) -> None:
    line = "═" * 62
    print("\n" + line)
    best = report["best_lap_time_s"]

    if best is None:
        print("  NO SUBMISSION")
        print(f"  None of {report['attempts']} agents completed a lap.")
        print("  Each attempt has to reach the start/finish line and then get")
        print("  all the way round it without touching a wall.")
        print(line)
        return

    speed = report["track_length_m"] / best
    print(f"  SUBMISSION      {best:8.3f} s        ← fastest of "
          f"{report['attempts']} agents")
    print(line)
    print(f"  average speed   {speed:8.2f} m/s      over "
          f"{report['track_length_m']:.1f} m")
    print(
        f"  completed       {report['laps_completed']:5d}/{report['attempts']}"
        f"          {report['completion_rate']:.0%} of attempts"
    )
    print(f"  median lap      {report['median_lap_time_s']:8.3f} s")
    print(f"  slowest lap     {report['worst_lap_time_s']:8.3f} s")
    print(f"  track           {report['track']} ({report['track_length_m']:.2f} m)")
    print(f"  sdk             {report['sdk_fingerprint']}")
    if report["sdk_modified"]:
        print("  WARNING: the SDK was modified — this result is not comparable.")
        for name in report["sdk_modified"]:
            print(f"           {name}")
    print(line)


def write_report(report: dict, checkpoint) -> None:
    out = args_cli.out or str(checkpoint.parent / "submission.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\n[evaluate] written to {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
