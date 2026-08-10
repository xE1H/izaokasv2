"""The official score: ten cars, one lap each, timed from the moment they are
put down, and the fastest lap counts.

    benchmark                                      # newest checkpoint
    benchmark --headless                           # no window, quicker for it
    benchmark --checkpoint logs/<run>/model_1000.pt
    benchmark --spawn 3.9 3.3 --spawn-yaw -156     # start somewhere else
    benchmark --agents 1 --spawn-jitter 0          # one car, exactly straight
    benchmark --no-submit                          # score it, keep it private

    python -m lituanicax_sdk.benchmark --help      # without the helper script

This is ``play.py`` with the watching taken out and a stopwatch put in. Ten
cars are placed on the official track and each drives until

* it **crosses the line it started on**, having gone all the way round — that
  time is recorded, and the attempt ends on the spot;
* it **touches a wall** or rolls over — the same wall the car hits while you
  are training, the car's centre within 0.15 m of it — and the attempt ends;
* it **stops making progress**, as a car wedged nose-first against a wall does,
  or the attempt window runs out.

**The fastest of the ten laps is the team's submission.** Not the mean, not the
median: a team is judged on the best lap it can produce, the way a qualifying
session works.

**The clock starts the instant the car is spawned** and stops the instant it
crosses back over the line, so the time is the whole of the car's drive: no
untimed out-lap, and one lap of driving for one lap of time. The line is the
plane through the spawn point at right angles to the centerline there — the
same shape of gate the SDK times training laps with, moved to where the car
starts.

The cars start at the **world origin (0, 0)**, facing along the track, wherever
your training happens to start its own — ``--spawn X Y`` and ``--spawn-yaw``
move that point. Every team is scored from the same place, which is what keeps
two teams' times comparable.

**Why ten, and why they are not identical.** The policy runs deterministically,
so ten cars from the exact same pose would drive ten identical laps to the
millisecond and tell you nothing you did not already know from one. So each
starts within ``--spawn-jitter`` degrees of straight, five either way by
default — far too small to matter at the line, and far too small to make one
attempt easier than another, but enough to separate the ten runs. Driving a
track is chaotic: a couple of degrees at the start decides whether a car makes
a corner most of a lap later, and a policy that only survives one exact opening
has not really learned the track. Ten attempts sample that, and the best one
counts.

The jitter is seeded (``--seed``), so a rerun repeats the same ten starts.

**The lap is published, and so is the policy that set it.** When the run
produces a time it is sent to the official leaderboard — team name, lap time,
the SDK's fingerprint — together with a zip of the checkpoint that drove it,
your ``teamcode/`` and the run's config. The lap appears on the board straight
away, greyed out and worth nothing: *awaiting verification*. The organisers then
run that same policy themselves, on a clean official SDK, and the row turns
green and takes its position if it produces the same time.

That is the whole of what makes the board mean anything. The fingerprint is
computed by your copy of the SDK, so a team willing to edit ``_locked.py`` can
send any value at all; a lap re-driven on somebody else's machine cannot be
edited into existence. See :mod:`lituanicax_sdk.bundle` for what is in the zip
and :mod:`lituanicax_sdk.verify` for what the organisers run.

Set ``LITUANICAX_TEAM`` so the board knows whose lap it is — see
:mod:`lituanicax_sdk.submit` — and pass ``--no-submit`` for a run you keep to
yourself. Publishing cannot fail a run: a board that is unreachable costs you a
printed line and nothing else.

The one thing this does not take from ``play.py`` is your
``compute_terminations``: the benchmark runs under
:mod:`lituanicax_sdk.rules` instead, so that a team with an aggressive stall
rule and a team with none are not running different sessions. Your observations
obviously still are yours — the policy could not run otherwise.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Isaac Sim has to be launched before anything else can be imported, so the
#  command-line arguments are parsed first and the rest of the imports follow.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import os
import sys
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

from lituanicax_sdk.runs import HYDRA_QUIET

TASK = "LituanicaX-Race-Team-v0"

#: How many attempts a submission is the best of.
AGENTS = 10

#: How long the simulator gets to shut down before the process stops waiting
#: for it. Generous — a shutdown that is merely slow finishes well inside this;
#: one that is stuck never finishes at all. See :func:`close_or_bail`.
SHUTDOWN_GRACE_S = 60.0

#: How much the starting heading varies between them, in degrees either way.
#: Small enough that every car starts on the same piece of track pointing the
#: same way; large enough that they do not all drive the identical lap.
SPAWN_JITTER_DEG = 5.0

parser = argparse.ArgumentParser(description="Officially score a trained policy.")
parser.add_argument(
    "--agents",
    type=int,
    default=AGENTS,
    help=f"How many attempts to run at once (default: {AGENTS}).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Policy to score (default: the most recent in logs/).",
)
parser.add_argument(
    "--spawn",
    type=float,
    nargs=2,
    metavar=("X", "Y"),
    default=None,
    help="Start every agent here instead of at the world origin, in metres.",
)
parser.add_argument(
    "--spawn-yaw",
    type=float,
    default=None,
    metavar="DEG",
    help="Heading to start at (default: follow the track at the spawn point).",
)
parser.add_argument(
    "--spawn-jitter",
    type=float,
    default=SPAWN_JITTER_DEG,
    metavar="DEG",
    help="How far the starting heading varies between agents, either way "
    f"(default: {SPAWN_JITTER_DEG:g}; 0 starts them all identically).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for the spawn jitter, so a rerun repeats the same ten starts.",
)
parser.add_argument(
    "--out", type=str, default=None, help="Where to write submission.json."
)
parser.add_argument(
    "--team",
    type=str,
    default=None,
    help="Team to publish under (default: $LITUANICAX_TEAM, or .lituanicax.json).",
)
parser.add_argument(
    "--no-submit",
    action="store_true",
    help="Score the policy without publishing the result, or the policy, at all.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args + HYDRA_QUIET

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────────────────────────────────────────────────────────
#  Everything below runs with Isaac Sim already up.
# ─────────────────────────────────────────────────────────────────────────────

from typing import cast  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import teamcode  # noqa: F401, E402 — importing registers the task
from lituanicax_sdk import (  # noqa: E402
    RaceEnv,
    RaceEnvCfg,
    sdk_fingerprint,
    tracks,
    verify_integrity,
)
from lituanicax_sdk._locked import runtime_fingerprint  # noqa: E402
from lituanicax_sdk.bundle import BundleError, build_bundle  # noqa: E402
from lituanicax_sdk.runs import find_checkpoint  # noqa: E402
from lituanicax_sdk.spawn import SpawnManager  # noqa: E402
from lituanicax_sdk.submit import print_outcome, submit  # noqa: E402
from lituanicax_sdk.timing import AttemptTimer  # noqa: E402

#: A failed attempt has no lap time.
NO_LAP = float("inf")


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Load a checkpoint, drive one timed lap, print the time."""
    # Isaac Sim exits the process hard, so anything still sitting in Python's
    # block buffer when the app closes is lost — which is the whole report the
    # moment you redirect a run to a file.
    sys.stdout.reconfigure(line_buffering=True)
    modified = verify_integrity()

    # ── The benchmark conditions, identical for every team ───────────────
    # Set here, after Hydra has had the config, so that a command-line override
    # cannot quietly change what is being measured.
    spawn = build_spawn_manager()
    # Hydra hands back whatever the task registered, typed as the Isaac Lab
    # base; everything the benchmark can score is a RaceEnv, so say so.
    race_cfg = cast(RaceEnvCfg, env_cfg)
    race_cfg.scene.num_envs = args_cli.agents
    race_cfg.track = tracks.OFFICIAL  # your own tracks are for training
    race_cfg.spawn_manager = spawn
    race_cfg.enforce_official_rules = True  # the SDK's crash rules, not yours
    race_cfg.official_stall_rule = True  # a wedged car fails, it does not wait
    # The attempt ends on the *spawn* line, not the track's start/finish line,
    # so the clock below decides when a car is finished rather than the SDK's
    # lap timer. Leaving terminate_on_lap off keeps the two from disagreeing.
    race_cfg.terminate_on_lap = False
    race_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    checkpoint = find_checkpoint(args_cli.checkpoint)
    print(f"\n[benchmark] policy : {checkpoint}")
    print(f"[benchmark] track  : {race_cfg.track.name}")
    print(f"[benchmark] spawn  : {spawn.describe()}")
    print(f"[benchmark] agents : {args_cli.agents}, one attempt each")
    print(f"[benchmark] sdk    : {sdk_fingerprint()}")

    env = RslRlVecEnvWrapper(
        cast(DirectRLEnv, gym.make(TASK, cfg=env_cfg, render_mode=None)),
        clip_actions=agent_cfg.clip_actions,
    )
    race = cast(RaceEnv, env.unwrapped)

    # ── Load the trained policy ───────────────────────────────────────────
    runner = OnPolicyRunner(
        env, class_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=race.device)
    network = runner.alg.policy

    # ── Drive ─────────────────────────────────────────────────────────────
    print(
        f"[benchmark] lap    : {race.track.track_length:.1f} m back to the line it "
        "starts on, timed from the moment the car is put down"
    )
    if race.num_envs > 1:
        print(
            f"[benchmark] the {race.num_envs} cars start on the same point, so the "
            "viewer shows them as one until they drive apart."
        )

    # Read before driving: the cars are still on the grid at this point.
    offsets = spawn_yaw_offsets(race, spawn)
    lap_times, outcomes = run_attempts(env, race, policy, network)

    report = build_report(
        lap_times, outcomes, offsets, race, spawn, checkpoint, modified
    )
    print_report(report)
    write_report(report, checkpoint)
    publish(report, checkpoint)

    close_or_bail(env)


def run_attempts(
    env: RslRlVecEnvWrapper, race: RaceEnv, policy, network
) -> tuple[list[float], list[str]]:
    """Drive every agent's single attempt to its end.

    Returns a lap time and an outcome per agent, in agent order, with
    :data:`NO_LAP` where the attempt produced no valid lap.

    Once an agent's attempt is over its result is banked, it is retired from
    the session, and it takes no further part: it stops being driven, it is not
    put back on the grid, and it is hidden, so what is left on the track is
    exactly the field still racing. One attempt per agent is the whole point of
    the format, and it is enforced in the simulation rather than only in the
    arithmetic — Isaac Lab would otherwise respawn a terminated car and let it
    drive the track again alongside the agents still on their attempt.
    """
    device, num_envs = race.device, race.num_envs

    lap_time = torch.full((num_envs,), NO_LAP, device=device)
    settled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    crashed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    timed_out = torch.zeros(num_envs, dtype=torch.bool, device=device)

    obs = env.get_observations()
    # The gate is where the cars are standing right now, so the clock is built
    # from the grid rather than from anything the spawn manager was asked for.
    clock = AttemptTimer(
        race.track, race.robot.data.root_pos_w[:, :2], race.step_dt, device
    )

    # An attempt cannot outlast the episode it runs in: the environment times
    # out at cfg.episode_length_s, which ends any attempt still going.
    for _ in range(int(race.max_episode_length) + 1):
        if not simulation_app.is_running() or bool(settled.all()):
            break

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # Recurrent policies carry state between steps; a car that has been
            # respawned must not drive on the memory of its last life.
            network.reset(dones)

        # Anything that ended the episode ended the attempt with it. Read this
        # first: a car that has just been terminated was teleported back to its
        # spawn point inside step(), and that jump crosses its own gate.
        ending = dones.bool() & ~settled

        position = race.robot.data.root_pos_w[:, :2]
        nearest, _ = race.track.nearest(position)
        finished, elapsed = clock.update(position, nearest, race.episode_length_buf)

        lapped = finished & ~settled & ~ending
        if lapped.any():
            lap_time = torch.where(lapped, elapsed, lap_time)
            # With one car the report below says this a moment later; with
            # several it is the only sign of who got round and when.
            if num_envs > 1:
                for agent in lapped.nonzero().flatten().tolist():
                    print(f"  agent {agent:2d}   lap {float(lap_time[agent]):7.3f} s")

        crashed |= ending & race.reset_terminated
        timed_out |= ending & race.reset_time_outs
        settled |= ending | lapped
        # One attempt per agent, in the simulation and not just in the score: a
        # settled car stops being driven and is not put back on the grid. The
        # environment already knows about the ones that failed; `lapped` is the
        # half only the clock above can see.
        race.retire(settled)

    outcomes = [
        "lap"
        if lap_time[i] != NO_LAP
        else "crashed"
        if bool(crashed[i])
        else "out of time"
        if bool(timed_out[i])
        else "unfinished"
        for i in range(num_envs)
    ]
    return [float(t) for t in lap_time], outcomes


def build_spawn_manager() -> SpawnManager:
    """Where the attempts start.

    One point — the world origin unless you move it — and a few degrees of
    heading either side of straight. The team's own spawn manager is
    deliberately not used: a curriculum that starts its cars on the easiest
    corner would be scoring a different track.
    """
    point = (0.0, 0.0) if args_cli.spawn is None else tuple(args_cli.spawn)
    return SpawnManager(
        xy=cast("tuple[float, float]", point),
        yaw_deg=args_cli.spawn_yaw,
        jitter_rad=math.radians(args_cli.spawn_jitter),
    )


def spawn_yaw_offsets(race: RaceEnv, spawn: SpawnManager) -> list[float]:
    """How far each car actually ended up from the nominal heading, in degrees.

    Read off the cars rather than off the spawn manager, so it is what the
    simulation did and not what it was asked for.
    """
    w, x, y, z = race.robot.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    nominal = spawn.pose()[2]
    offset = torch.remainder(yaw - nominal + math.pi, 2 * math.pi) - math.pi
    return [math.degrees(float(o)) for o in offset]


def build_report(
    lap_times: list[float],
    outcomes: list[str],
    yaw_offsets: list[float],
    race: RaceEnv,
    spawn: SpawnManager,
    checkpoint,
    modified: list[str],
) -> dict:
    finished = sorted(t for t in lap_times if t != NO_LAP)
    attempts = len(lap_times)
    return {
        # The submission.
        "best_lap_time_s": finished[0] if finished else None,
        # Context for it.
        "attempts": attempts,
        "laps_completed": len(finished),
        "median_lap_time_s": finished[len(finished) // 2] if finished else None,
        "slowest_lap_time_s": finished[-1] if finished else None,
        "lap_times_s": [None if t == NO_LAP else t for t in lap_times],
        "outcomes": outcomes,
        "spawn_yaw_offsets_deg": yaw_offsets,
        "spawn_jitter_deg": args_cli.spawn_jitter,
        "seed": args_cli.seed,
        "track": race.track.cfg.name,
        "track_length_m": race.track.track_length,
        "spawn": spawn.describe(),
        "checkpoint": str(checkpoint),
        "sdk_fingerprint": sdk_fingerprint(),
        # Read after the driving rather than before it, and after `teamcode` has
        # had every chance to touch the SDK: this is what the code the lap was
        # actually set under hashes to in memory. The organisers compare it with
        # a clean process of their own. See _locked.runtime_fingerprint.
        "runtime_fingerprint": runtime_fingerprint(),
        "sdk_modified": modified,
    }


def print_report(report: dict) -> None:
    line = "═" * 56
    attempts = report["attempts"]
    best = report["best_lap_time_s"]

    print("\n" + line)

    # Several cars are worth listing one by one; one car is the report.
    if attempts > 1:
        rows = zip(
            report["lap_times_s"],
            report["outcomes"],
            report["spawn_yaw_offsets_deg"],
        )
        for agent, (time_s, outcome, offset) in enumerate(rows):
            result = f"{time_s:8.3f} s" if time_s is not None else "        —"
            print(f"  agent {agent:2d}   {offset:+5.1f}°   {result}   {outcome}")
        print(line)

    if best is None:
        how = report["outcomes"][0] if attempts == 1 else "did not finish"
        print(f"  NO SUBMISSION — {how}")
        print("  A lap has to get all the way round, back over the line it")
        print("  started on, without touching a wall.")
        print(line)
        print(f"  spawn         {report['spawn']}")
        print(f"  sdk           {report['sdk_fingerprint']}")
        _print_modified(report, line)
        return

    fastest = f"      ← fastest of {attempts}" if attempts > 1 else ""
    print(f"  SUBMISSION    {best:8.3f} s{fastest}")
    print(line)
    print(
        f"  average speed {report['track_length_m'] / best:8.2f} m/s   over "
        f"{report['track_length_m']:.1f} m"
    )
    if attempts > 1:
        share = report["laps_completed"] / attempts
        print(
            f"  completed     {report['laps_completed']:5d}/{attempts}"
            f"       {share:.0%} of attempts"
        )
        print(f"  median lap    {report['median_lap_time_s']:8.3f} s")
        print(f"  slowest lap   {report['slowest_lap_time_s']:8.3f} s")
    print(f"  spawn         {report['spawn']}")
    print(f"  sdk           {report['sdk_fingerprint']}")
    _print_modified(report, line)


def _print_modified(report: dict, line: str) -> None:
    if report["sdk_modified"]:
        print("  WARNING: the SDK was modified — this result is not comparable.")
        for name in report["sdk_modified"]:
            print(f"           {name}")
    print(line)


def write_report(report: dict, checkpoint) -> None:
    out = args_cli.out or str(checkpoint.parent / "submission.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\n[benchmark] written to {out}")


def close_or_bail(env) -> None:
    """Shut the simulator down, but do not let it hold the process open.

    Isaac Sim's teardown does not reliably return — on a supported driver and a
    stock scene it can sit there indefinitely, long after the last line of
    output. A benchmark that has already printed its score, written
    ``submission.json`` and published the lap has nothing left to do at that
    point, and the process it is keeping alive is holding the GPU, which is the
    one thing the next run needs.

    So the shutdown gets a deadline. Everything worth keeping is on disk and
    already sent before the timer is armed, which is what makes exiting hard
    safe rather than merely convenient.
    """

    def bail() -> None:
        time.sleep(SHUTDOWN_GRACE_S)
        print(
            f"\n[benchmark] the simulator has not shut down after "
            f"{SHUTDOWN_GRACE_S:g}s, so this exits without waiting for it. "
            "Your result is unaffected: it is printed above, written to "
            "submission.json, and already published."
        )
        sys.stdout.flush()
        # Not sys.exit(): that unwinds through the same teardown that is stuck.
        os._exit(0)

    threading.Thread(target=bail, daemon=True).start()
    env.close()


def publish(report: dict, checkpoint) -> None:
    """Send the lap, and the policy that drove it, to the official leaderboard.

    The bundle is the half that matters: a lap time is a claim anybody can type,
    and the board no longer takes claims. What it ranks is a lap the organisers
    have re-run from the team's own checkpoint and ``teamcode/`` on a clean SDK.
    So the zip goes up with the time, and the row stays grey until it has been
    reproduced.

    Best-effort by design, both halves: :func:`lituanicax_sdk.submit.submit`
    reports a failure rather than raising one, so a board that is down, a laptop
    with no network or a team that has not set its name all still get their
    score printed and written to ``submission.json``.
    """
    if args_cli.no_submit:
        print("[submit] skipped (--no-submit) — nothing published, nothing to verify")
        return

    bundle = None
    if report["best_lap_time_s"] is not None:
        try:
            bundle = build_bundle(report, checkpoint, team=args_cli.team or "team")
            print(
                f"[submit] policy bundle {bundle.megabytes:.1f} MB, "
                f"{len(bundle.included)} files"
            )
            for left_out in bundle.skipped:
                print(f"[submit]   not included: {left_out}")
        except BundleError as exc:
            # A lap with no bundle is still worth publishing — it goes on the
            # board as a claim nobody can check, which is the honest thing for
            # it to be, and the team can see why and rerun.
            print(f"[submit] the policy bundle could not be built — {exc}")

    outcome = submit(report, team=args_cli.team, bundle=bundle)
    print_outcome(outcome)


if __name__ == "__main__":
    # Hydra fills in env_cfg and agent_cfg; the decorator hides that from static
    # analysis, so main() genuinely does take no arguments here.
    main()  # type: ignore[call-arg]
    simulation_app.close()
