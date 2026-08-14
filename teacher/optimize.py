"""Phase 2 — optimize the controller against the simulator with CMA-ES.

    # search (needs a CUDA GPU and Isaac Sim)
    python -m teacher.optimize --headless --population 64 --generations 100

    # measure the winner at the window a real attempt gets
    python -m teacher.optimize --headless --measure artifacts/teacher.json

Requires ``cma``::

    uv pip install cma

That is a deliberate dependency. CMA-ES with a working step-size adaptation is not
a few lines of numpy, and a hand-rolled version would be the sort of subtle,
plausible-looking code that quietly searches badly and takes the blame with it.
pycma is small, pure Python, and the reference implementation.

**What is being optimized.** Lap time, directly, with no reward shaping anywhere
in the loop. Each candidate is evaluated on all ``--starts`` official start
jitters at once, so a generation is ``population x starts`` cars in one
simulation, and the score is the min over that candidate's starts — which is
exactly what the leaderboard takes.

**Two things that are not the plan, and why.**

*Fixed population instead of IPOP restarts.* Doubling the population on stagnation
means a different number of environments, and the number of environments is fixed
when the Isaac scene is built — so a true IPOP restart costs a scene rebuild,
which is minutes. Restarts here re-inflate ``sigma`` from the incumbent best at the
same population size. Running a genuinely larger population means invoking the
script again with a bigger ``--population``, which is cheap to do and easy to
compare.

*The search runs a short attempt window.* A valid lap is 12-16 s, so
``--episode 25`` is generous while killing stragglers early; at the official 60 s
a generation spends most of its time simulating cars that already crashed.
``T_teacher`` is then re-measured at 60 s in ``--measure`` mode, because that is
the window the benchmark will actually give a car and the SDK does not override it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Optimize the teacher with CMA-ES.")
parser.add_argument(
    "--population", type=int, default=64, help="Candidates per generation."
)
parser.add_argument(
    "--starts", type=int, default=10, help="Start jitters per candidate."
)
parser.add_argument("--generations", type=int, default=100)
parser.add_argument("--seeds", type=int, default=1, help="Independent CMA-ES runs.")
parser.add_argument(
    "--sigma", type=float, default=0.15, help="Initial step, normalized."
)
parser.add_argument(
    "--episode", type=float, default=25.0, help="Search attempt window, s."
)
parser.add_argument(
    "--min-completions",
    type=int,
    default=1,
    help=(
        "Starts a candidate must finish to be scored on time. Default 1, "
        "matching the leaderboard, which takes the fastest of ten attempts and "
        "does not care about the other nine."
    ),
)
parser.add_argument("--warmstart", default="artifacts/teacher-warmstart.json")
parser.add_argument("--out", default="artifacts/teacher.json")
parser.add_argument("--reference", default="teamcode/reference.npz")
parser.add_argument("--dynamics", default="artifacts/dynamics.json")
parser.add_argument("--spacing", type=float, default=0.02)
parser.add_argument(
    "--restart-after",
    type=int,
    default=25,
    help="Generations without improvement before re-inflating sigma.",
)
parser.add_argument(
    "--measure",
    default=None,
    metavar="PARAMS",
    help="Skip the search: measure this parameter file at the official window.",
)
parser.add_argument(
    "--measure-episode",
    type=float,
    default=None,
    help=(
        "Attempt window for --measure, seconds. Defaults to the official 60. "
        "Set it to the search's window to check whether a candidate that scored "
        "well in the search is genuinely fragile or merely being measured "
        "somewhere different."
    ),
)
parser.add_argument("--allow-cpu", action="store_true")
parser.add_argument(
    "--trace",
    default=None,
    metavar="PATH",
    help="With --measure: record the control law's inputs and outputs per step.",
)
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
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from lituanicax_sdk.vehicle import VEHICLE  # noqa: E402
from teacher.controller import (  # noqa: E402
    Controller,
    build_reference,
    stack_references,
)
from teacher.params import ControllerParams  # noqa: E402
from teacher.warmstart import build as build_warmstart  # noqa: E402
from tools.evaluate import (  # noqa: E402
    RepeatedStarts,
    evaluate,
    objective,
    official_start_offsets,
    summarize,
)
from tools.geometry import TrackGeometry  # noqa: E402
from tools.harness import OFFICIAL_EPISODE_S, make_env  # noqa: E402
from tools.measured import Measured  # noqa: E402


def require_cma():
    try:
        import cma  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover — environment problem
        raise SystemExit(
            "teacher.optimize needs the `cma` package:\n\n    uv pip install cma\n\n"
            "It is the reference CMA-ES implementation; hand-rolling one is the "
            "kind of subtle code that searches badly without ever looking broken."
        ) from error
    return cma


def make_driver(geometry, candidates, rows, car: Measured):
    """One controller driving a whole generation."""
    references = [build_reference(geometry, p, v_max=car.v_max_m_s) for p in candidates]
    return Controller(
        geometry,
        stack_references(references),
        candidates,
        wheelbase_m=car.wheelbase_m,
        max_steer_rad=VEHICLE.max_steer_rad,
        rows=rows,
    )


def score_generation(env, geometry, candidates, rows, car, starts, min_completions):
    """Drive one generation and return ``(losses, details)``."""
    driver = make_driver(geometry, candidates, rows, car)
    attempts = evaluate(env, driver)

    losses, details = [], []
    for index in range(len(candidates)):
        block = attempts[index * starts : (index + 1) * starts]
        loss, detail = objective(
            block, min_completions=min_completions, num_starts=starts
        )
        losses.append(loss)
        details.append(detail)
    return losses, details


# ═══════════════════════════════════════════════════════════════════════════
#  Search
# ═══════════════════════════════════════════════════════════════════════════


def checkpoint(params: ControllerParams, report: dict) -> None:
    """Write the incumbent to disk, now, before the next generation.

    A hundred generations is hours of GPU time, and until this existed the only
    write was after the last one — so a crash, a preemption, or a decision to
    stop early at generation 80 threw all of it away. Written every time the best
    improves, which is rare enough to cost nothing and often enough that the most
    that can be lost is the generations since the last improvement.
    """
    params.save(args_cli.out)
    Path("artifacts/optimize-history.json").write_text(json.dumps(report, indent=2))


def search(
    env, geometry, car: Measured, warm: ControllerParams
) -> tuple[ControllerParams, dict]:
    cma = require_cma()
    starts = args_cli.starts
    population = args_cli.population
    rows = torch.arange(population * starts, device=env.device) // starts

    best_params, best_loss = warm, float("inf")
    history: list[dict] = []

    for seed in range(args_cli.seeds):
        print(f"\n[optimize] ── CMA-ES seed {seed + 1}/{args_cli.seeds} ──")
        # The generation budget is spent across however many restarts stagnation
        # calls for, rather than being abandoned at the first one. Before this
        # loop existed, hitting --restart-after simply broke out and — with a
        # single seed — ended the entire run, while printing that it was
        # restarting. A 250-generation search would have stopped at about 40.
        spent, attempt = 0, 0
        while spent < args_cli.generations:
            # Widen the step each time. A search that has stopped improving at
            # this scale needs to look further out, and the incumbent it starts
            # from is kept regardless, so a bad restart costs generations rather
            # than the result.
            sigma = args_cli.sigma * (1.5**attempt)
            if attempt:
                print(
                    f"[optimize] restart {attempt} from the incumbent, "
                    f"sigma {sigma:.3f}"
                )
            strategy = cma.CMAEvolutionStrategy(
                list(best_params.to_normalized()),
                sigma,
                {
                    "bounds": [0.0, 1.0],
                    "popsize": population,
                    "seed": seed * 1000 + attempt + 1,
                    "verbose": -9,
                },
            )
            stale, seed_best = 0, float("inf")
            attempt += 1

            for generation in range(spent, args_cli.generations):
                spent += 1
                if strategy.stop():
                    print(f"[optimize] stopped: {strategy.stop()}")
                    break

                started = time.time()
                solutions = strategy.ask()
                candidates = [ControllerParams.from_normalized(x) for x in solutions]
                losses, details = score_generation(
                    env,
                    geometry,
                    candidates,
                    rows,
                    car,
                    starts,
                    args_cli.min_completions,
                )
                strategy.tell(solutions, losses)

                order = int(np.argmin(losses))
                loss = losses[order]
                laps = sum(1 for d in details if d["branch"] == "lap")
                elapsed = time.time() - started

                improved = loss < best_loss
                if improved:
                    best_loss, best_params = loss, candidates[order]
                    marker = "  <- best so far, saved"
                else:
                    marker = ""
                if loss < seed_best:
                    seed_best, stale = loss, 0
                else:
                    stale += 1

                detail = details[order]
                shown = (
                    f"{detail['best_lap_time_s']:.3f} s"
                    if detail["best_lap_time_s"] is not None
                    else f"no lap ({detail['best_progress']:.0%} round)"
                )
                print(
                    f"[optimize] gen {generation + 1:3d}  J {loss:7.3f}  {shown:>22}  "
                    f"{detail['completions']}/{starts} done  "
                    f"{laps}/{population} candidates lapped  {elapsed:.0f}s{marker}"
                )
                history.append(
                    {
                        "seed": seed,
                        "generation": generation,
                        "loss": loss,
                        "candidates_lapping": laps,
                        **{
                            k: detail[k]
                            for k in ("completions", "best_lap_time_s", "best_progress")
                        },
                    }
                )

                if improved:
                    checkpoint(
                        best_params,
                        {"best_loss": best_loss, "history": history},
                    )

                if stale >= args_cli.restart_after:
                    # Not IPOP: the environment fixes the population size, so this
                    # re-inflates sigma from the incumbent instead of growing lambda.
                    print(
                        f"[optimize] {stale} generations without improvement — "
                        "restarting from the incumbent with a wider sigma"
                    )
                    break

    return best_params, {"best_loss": best_loss, "history": history}


# ═══════════════════════════════════════════════════════════════════════════
#  The reportable measurement
# ═══════════════════════════════════════════════════════════════════════════


class Tracing:
    """Wraps a driver and records what it saw and what it did, every step.

    Gate 1 is a yes/no — the car either completed a lap or it did not — and when
    it says no there is nothing in the answer to act on. This records the inputs
    to the control law alongside its outputs, so the difference between "the
    speed profile asked for more than the car can do" and "the steering is
    tracking the wrong line" is one plot rather than a guess.
    """

    def __init__(self, driver, geometry, reference):
        self.driver = driver
        self.geometry = geometry
        self.reference = reference
        self.frames: list[dict] = []

    def __call__(self, car):
        actions = self.driver(car)
        s, offset = self.geometry.project(car.pos_xy)
        self.frames.append(
            {
                "s": s.detach().cpu().numpy().copy(),
                "n": offset.detach().cpu().numpy().copy(),
                "speed": car.speed_forward.detach().cpu().numpy().copy(),
                "throttle": actions[:, 0].detach().cpu().numpy().copy(),
                "steer_cmd": actions[:, 1].detach().cpu().numpy().copy(),
                "steer_angle": car.steer_angle.detach().cpu().numpy().copy(),
                "up_axis": car.up_axis.detach().cpu().numpy().copy(),
                "pos": car.pos_xy.detach().cpu().numpy().copy(),
                "ref_n": self.geometry.interp(s, self.reference.offset)
                .detach()
                .cpu()
                .numpy()
                .copy(),
                "ref_speed": self.geometry.interp(s, self.reference.speed)
                .detach()
                .cpu()
                .numpy()
                .copy(),
                "ref_kappa": self.geometry.interp(s, self.reference.kappa)
                .detach()
                .cpu()
                .numpy()
                .copy(),
            }
        )
        return actions

    def save(self, path) -> None:
        keys = self.frames[0].keys()
        np.savez(
            path,
            **{k: np.stack([f[k] for f in self.frames]) for k in keys},
            track_length_m=np.array(self.geometry.length),
        )


def measure(
    env, geometry, car: Measured, params: ControllerParams, trace: str | None = None
) -> dict:
    """Score one candidate at the window a real attempt gets."""
    starts = env.num_envs
    rows = torch.zeros(starts, dtype=torch.long, device=env.device)
    driver = make_driver(geometry, [params] * 1, rows, car)
    if trace:
        reference = build_reference(geometry, params, v_max=car.v_max_m_s)
        driver = Tracing(driver, geometry, reference)
    attempts = evaluate(env, driver, verbose=True)
    if trace:
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        driver.save(trace)
        print(f"[optimize] controller trace -> {trace}")

    valid = sorted(a.lap_time_s for a in attempts if a.valid)
    return {
        "T_teacher": valid[0] if valid else None,
        "median_lap_time_s": valid[len(valid) // 2] if valid else None,
        "slowest_lap_time_s": valid[-1] if valid else None,
        "completions": len(valid),
        "attempts": len(attempts),
        "lap_times_s": [a.lap_time_s for a in attempts],
        "outcomes": [a.outcome for a in attempts],
        "episode_length_s": float(env.cfg.episode_length_s),
        "summary": summarize(attempts),
    }


def main() -> int:
    car = Measured.load(args_cli.dynamics)
    print(car.describe())
    if not car.measured:
        print(
            "[optimize] WARNING: no probe results, so the warm start and the "
            "effective limits are guesses. Run `python -m tools.probe` first."
        )

    geometry = TrackGeometry.from_track(
        Track(OFFICIAL, device="cpu"), spacing_m=args_cli.spacing
    )
    print(geometry.describe())

    offsets = official_start_offsets(count=args_cli.starts)
    spawn = RepeatedStarts(offsets)

    # ── Measure only ──────────────────────────────────────────────────────
    if args_cli.measure:
        params = ControllerParams.load(args_cli.measure)
        print(f"\n[optimize] measuring {args_cli.measure} at the official window")
        print(params.describe())
        # The official window unless deliberately overridden. --measure-episode
        # exists for one reason: a candidate that scores well in the search and
        # then fails here is either a real fragility or a difference between the
        # two environments, and the only way to tell is to make them the same.
        window = (
            args_cli.measure_episode
            if args_cli.measure_episode is not None
            else OFFICIAL_EPISODE_S
        )
        env = make_env(
            num_envs=args_cli.starts,
            spawn=spawn,
            official_rules=True,
            episode_length_s=window,
            allow_cpu=args_cli.allow_cpu,
        )
        # The controller's geometry must live where the cars do.
        geometry_on_device = TrackGeometry.from_track(
            env.track, spacing_m=args_cli.spacing
        )
        result = measure(env, geometry_on_device, car, params, trace=args_cli.trace)
        print()
        print("=" * 60)
        if result["T_teacher"] is None:
            print(f"  NO LAP — {result['summary']}")
        else:
            print(f"  T_teacher   {result['T_teacher']:.3f} s")
            print(f"  {result['summary']}")
            print(f"  window      {result['episode_length_s']:.0f} s")
        print("=" * 60)
        Path("artifacts/T_teacher.json").write_text(json.dumps(result, indent=2))
        print("[optimize] written to artifacts/T_teacher.json")
        return 0 if result["T_teacher"] is not None else 1

    # ── Search ────────────────────────────────────────────────────────────
    warm_path = Path(args_cli.warmstart)
    if warm_path.is_file():
        warm = ControllerParams.load(warm_path)
        print(f"[optimize] warm start from {warm_path}")
    else:
        print(f"[optimize] no {warm_path}, building a warm start now")
        warm, _ = build_warmstart(geometry, car)
    print(warm.describe())

    total = args_cli.population * args_cli.starts
    print(
        f"\n[optimize] {args_cli.population} candidates x {args_cli.starts} starts "
        f"= {total} cars, {args_cli.episode:.0f} s window"
    )
    env = make_env(
        num_envs=total,
        spawn=spawn,
        official_rules=True,
        episode_length_s=args_cli.episode,
        allow_cpu=args_cli.allow_cpu,
    )
    geometry_on_device = TrackGeometry.from_track(env.track, spacing_m=args_cli.spacing)

    best, report = search(env, geometry_on_device, car, warm)

    path = best.save(args_cli.out)
    print(f"\n[optimize] best parameters -> {path}")
    print(best.describe())

    # reference.npz goes into teamcode/, because that is the only directory the
    # submission bundle carries (lituanicax_sdk/bundle.py:227) — a Phase 3 student
    # that reads it from anywhere else would fail verification.
    reference = build_reference(geometry_on_device, best, v_max=car.v_max_m_s)
    reference_path = Path(args_cli.reference)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        reference_path,
        s=geometry_on_device.s.cpu().numpy(),
        x=geometry_on_device.pos[:, 0].cpu().numpy(),
        y=geometry_on_device.pos[:, 1].cpu().numpy(),
        psi=geometry_on_device.heading_at(geometry_on_device.s).cpu().numpy(),
        track_length_m=np.array(geometry_on_device.length),
        **reference.as_arrays(),
    )
    print(f"[optimize] reference -> {reference_path}")

    Path("artifacts/optimize-history.json").write_text(json.dumps(report, indent=2))
    print(
        f"[optimize] best J {report['best_loss']:.3f}. Now measure it at the "
        f"official window:\n           python -m teacher.optimize --headless "
        f"--measure {args_cli.out}"
    )
    return 0


if __name__ == "__main__":
    status = main()
    # Give the simulator half a minute to shut down and then leave regardless.
    # It has wedged in teardown here for over ten minutes after a completed run,
    # and a search that has just spent hours producing a result must not lose it
    # to a shutdown — everything is written to disk by this point. os._exit
    # because a process stuck in a C++ lock does not unwind or take a signal.
    import threading  # noqa: PLC0415

    closing = threading.Thread(target=simulation_app.close, daemon=True)
    closing.start()
    closing.join(timeout=30.0)
    if closing.is_alive():
        print("[optimize] the simulator will not shut down; leaving without it.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
