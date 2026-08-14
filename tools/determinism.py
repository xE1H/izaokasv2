"""Phase 0.4 — how much of what we measure is real.

    # intra-batch: do identical cars in one simulation stay identical?
    python -m tools.determinism --headless --envs 64 --out artifacts/det-64.npz

    # cross-batch: does a car reproduce when the population size changes?
    python -m tools.determinism --headless --envs 1 --out artifacts/det-1.npz
    python -m tools.determinism --compare artifacts/det-64.npz artifacts/det-1.npz

    # can a recorded mid-corner state be restored and continued?
    python -m tools.determinism --headless --restore

**Needs Isaac Sim** except for ``--compare``, which only reads files.

Neither probe blocks Phase 1, and both change how the results are read.

**Why cross-batch determinism has teeth.** CMA-ES scores a candidate inside a
population of hundreds and then the winner is re-measured on its own. If GPU PhysX
does not reproduce across batch sizes — and it often does not, because the solver's
work is partitioned differently — then those two numbers are not the same
measurement, ``T_teacher`` carries that difference as error, and a candidate can
win a generation on noise. The fix if it fails is to average each candidate over
repeats, which costs throughput, so it is worth knowing before spending a day of
GPU time rather than after.

**Why state restorability is worth an hour.** If a recorded state can be written
back into a fresh environment and continued, then sampling-based trajectory
optimization becomes available later — you can branch from a corner entry and try
a hundred throttle traces from exactly the same state. If it cannot, that tool is
off the table and the search stays evolutionary.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

parser = argparse.ArgumentParser(
    description="Measure how reproducible the simulator is."
)
parser.add_argument("--envs", type=int, default=64, help="Cars to run.")
parser.add_argument("--steps", type=int, default=200, help="Steps to record.")
parser.add_argument("--out", default="artifacts/determinism.npz")
parser.add_argument(
    "--compare",
    nargs=2,
    metavar=("A", "B"),
    default=None,
    help="Compare two recordings and exit. Needs no simulator.",
)
parser.add_argument(
    "--restore", action="store_true", help="Run the state-restorability probe."
)
parser.add_argument(
    "--restore-at", type=int, default=90, help="Step to snapshot for --restore."
)
parser.add_argument("--restore-steps", type=int, default=30)
parser.add_argument("--allow-cpu", action="store_true")


def compare(first: str, second: str) -> int:
    """Report how far two recordings drift apart. Reads files only."""
    a, b = np.load(first), np.load(second)
    steps = min(len(a["position"]), len(b["position"]))
    position = np.linalg.norm(a["position"][:steps] - b["position"][:steps], axis=-1)
    speed = np.abs(a["speed"][:steps] - b["speed"][:steps])

    print(f"[determinism] {first} vs {second}, {steps} common steps")
    print(
        f"  position: max {position.max() * 1000:.3f} mm, "
        f"final {position[-1] * 1000:.3f} mm"
    )
    print(f"  speed:    max {speed.max() * 1000:.3f} mm/s")

    if position.max() < 1e-6:
        print("\n  BITWISE IDENTICAL. A candidate scored in a population reproduces")
        print("  exactly when re-run alone, so no averaging over repeats is needed.")
        return 0
    if position.max() < 0.01:
        print(
            f"\n  CLOSE ({position.max() * 1000:.2f} mm). Trajectories diverge but stay"
        )
        print("  within a centimetre over this window — small against the 0.15 m")
        print("  crash radius. Single evaluations are usable; expect a little noise")
        print("  in the last digit of a lap time.")
        return 0
    print(
        f"\n  DIVERGES ({position.max() * 1000:.0f} mm). "
        "The same actions do not produce"
    )
    print("  the same trajectory across these two runs, so a candidate's score")
    print("  carries real evaluation noise. Average each candidate over repeats in")
    print("  teacher.optimize, and treat differences smaller than this as nothing.")
    return 1


def script(step: int, count: int):
    """A deterministic manoeuvre that actually loads the car.

    Steady throttle with a slow steering sweep, so the run spends its time
    cornering — where the solver has contact and friction to get wrong — rather
    than rolling in a straight line where almost anything reproduces.
    """
    import torch

    actions = torch.zeros(count, 2)
    actions[:, 0] = 0.6
    actions[:, 1] = 0.7 * math.sin(step / 25.0)
    return actions


def main() -> int:
    # Not parse_args: AppLauncher's flags (--headless above all) arrive on the
    # same command line and belong to its parser, not this one.
    args, _ = parser.parse_known_args()

    if args.compare:
        return compare(*args.compare)

    # ── Isaac Sim from here on ────────────────────────────────────────────
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    app = AppLauncher(launcher_args).app

    import torch

    from tools.harness import make_env

    status = 0
    try:
        env = make_env(
            num_envs=args.envs,
            official_rules=False,
            stall_rule=False,
            episode_length_s=(args.steps + args.restore_steps + 20) / 30.0,
            allow_cpu=args.allow_cpu,
        )

        if args.restore:
            status = restorability(env, args, torch)
        else:
            status = reproducibility(env, args, torch)
    finally:
        try:
            app.close()
        except Exception as error:  # pragma: no cover
            print(f"[determinism] the simulator did not shut down cleanly: {error}")
    return status


def reproducibility(env, args, torch) -> int:
    """Record a trajectory, and check that identical cars stayed identical."""
    env.reset()
    positions, speeds = [], []
    for step in range(args.steps):
        env.step(script(step, env.num_envs).to(env.device))
        car = env.latest_car
        positions.append(car.pos_xy.detach().cpu().numpy().copy())
        speeds.append(car.speed_forward.detach().cpu().numpy().copy())

    position = np.stack(positions)  # [steps, envs, 2]
    speed = np.stack(speeds)

    print(f"\n[determinism] {env.num_envs} cars, {args.steps} steps, identical actions")
    if env.num_envs > 1:
        # Every car started at the same pose and got the same commands, so any
        # spread between them is the simulator, not the manoeuvre.
        spread = np.linalg.norm(position - position[:, :1, :], axis=-1)
        print(f"  spread between identical cars: max {spread.max() * 1000:.3f} mm")
        if spread.max() < 1e-6:
            print("  -> environments in one batch are bitwise independent.")
        elif spread.max() < 0.01:
            print("  -> environments drift by under a centimetre. Usable, but a")
            print("     population's scores carry that much noise.")
        else:
            print(
                f"  -> environments DIVERGE by {spread.max() * 1000:.0f} mm. Cars that"
            )
            print("     should be identical are not, so any single evaluation is a")
            print("     sample rather than a measurement. Average over repeats.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        position=position[:, 0, :],
        speed=speed[:, 0],
        num_envs=np.array(env.num_envs),
        steps=np.array(args.steps),
    )
    print(f"\n[determinism] car 0 written to {args.out}")
    print("  Run again with a different --envs and compare, to find out whether a")
    print("  candidate scored in a population reproduces when re-run on its own:")
    print(f"    python -m tools.determinism --compare {args.out} <other.npz>")
    return 0


def restorability(env, args, torch) -> int:
    """Snapshot a mid-corner state, write it back, and see if it continues."""
    env.reset()
    for step in range(args.restore_at):
        env.step(script(step, env.num_envs).to(env.device))

    robot = env.robot
    snapshot = {
        "root_pose": robot.data.root_state_w[:, :7].clone(),
        "root_velocity": robot.data.root_state_w[:, 7:].clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
    }

    original = []
    for step in range(args.restore_at, args.restore_at + args.restore_steps):
        env.step(script(step, env.num_envs).to(env.device))
        original.append(env.latest_car.pos_xy.detach().cpu().numpy().copy())

    # Put it back exactly where it was and drive the same commands again.
    robot.write_root_pose_to_sim(snapshot["root_pose"])
    robot.write_root_velocity_to_sim(snapshot["root_velocity"])
    robot.write_joint_state_to_sim(snapshot["joint_pos"], snapshot["joint_vel"])
    env.sim.step(render=False)

    restored = []
    for step in range(args.restore_at, args.restore_at + args.restore_steps):
        env.step(script(step, env.num_envs).to(env.device))
        restored.append(env.latest_car.pos_xy.detach().cpu().numpy().copy())

    drift = np.linalg.norm(np.stack(original) - np.stack(restored), axis=-1)
    print(
        f"\n[determinism] restored a state from step {args.restore_at}, "
        f"replayed {args.restore_steps} steps"
    )
    print(
        f"  divergence: max {drift.max() * 1000:.2f} mm, "
        f"final {drift[-1].max() * 1000:.2f} mm"
    )

    if drift.max() < 0.01:
        print("\n  RESTORABLE. Sampling-based trajectory optimization is available:")
        print("  a state can be branched from and continued, so a hundred throttle")
        print("  traces can be tried from the identical corner entry.")
        return 0
    print("\n  NOT RESTORABLE. Writing a state back does not reproduce what followed,")
    print("  so branching from a recorded corner is off the table and the search")
    print("  stays evolutionary. Not a blocker — just one fewer tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
