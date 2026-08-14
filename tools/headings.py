"""The ten headings the benchmark will actually use.

    python -m tools.headings --headless

**Needs Isaac Sim.** Writes ``artifacts/headings.json``.

The benchmark spawns ten cars at the world origin and gives each a random yaw
within ±5°, drawn once from a seeded RNG (``benchmark.py:393-397``). Its own
docstring promises the draw repeats: *"The jitter is seeded (--seed), so a rerun
repeats the same ten starts."* So there is a specific set of ten headings that
will be scored, and optimizing against any other set is optimizing against
starts the car will never see.

That mattered more than it sounds. The search had been scoring candidates on
headings evenly spaced across the range, which is a sensible proxy and is not
the thing: candidates kept winning on a heading of −1.7° that the benchmark
never draws, then failing on the ten it does. Every apparent improvement below
15.067 s came from that mismatch.

The draw depends on the RNG state at the moment ``SpawnManager.sample`` runs, so
this reads the headings out of a *running* environment rather than trying to
recompute them, and checks the two things that would make them not the
benchmark's:

* **stability across resets and runs** — if the same environment gives different
  headings twice, nothing here is reproducible and the whole idea is dead;
* **independence from the environment's own configuration** — this harness is
  not byte-for-byte the team's environment, so if the draw shifts when the
  observation width changes, then it also shifts between here and the benchmark
  and the numbers cannot be trusted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Read the benchmark's actual starts.")
parser.add_argument("--out", default="artifacts/headings.json")
parser.add_argument("--agents", type=int, default=10, help="Cars, as the benchmark.")
parser.add_argument("--seed", type=int, default=0, help="Spawn seed, as the benchmark.")
parser.add_argument("--jitter-deg", type=float, default=5.0)
parser.add_argument("--allow-cpu", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────────────────────────────────────────────────────────
#  Isaac Sim is up from here on.
# ─────────────────────────────────────────────────────────────────────────────

import torch  # noqa: E402

from lituanicax_sdk.spawn import SpawnManager  # noqa: E402
from tools.harness import make_env  # noqa: E402


def read_offsets(env, spawn) -> list[float]:
    """Degrees each car actually ended up from the nominal heading.

    Read off the cars, not off the spawner — the same way ``benchmark.py:400``
    does it, so this is what the simulation did rather than what it was asked
    for.
    """
    w, x, y, z = env.robot.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    nominal = spawn.pose()[2]
    offset = torch.remainder(yaw - nominal + math.pi, 2 * math.pi) - math.pi
    return [math.degrees(float(o)) for o in offset]


def headings(observation_space: int | None = None) -> list[float]:
    """Build the benchmark's spawn, reset once, and read what came out."""
    spawn = SpawnManager(xy=(0.0, 0.0), jitter_rad=math.radians(args_cli.jitter_deg))
    env = make_env(
        num_envs=args_cli.agents,
        spawn=spawn,
        official_rules=True,
        seed=args_cli.seed,
        allow_cpu=args_cli.allow_cpu,
    )
    if observation_space is not None:
        env.cfg.observation_space = observation_space
    env.reset()
    offsets = read_offsets(env, spawn)
    env.close()
    return offsets


def main() -> int:
    first = headings()
    second = headings()

    print("\n  first  read: " + ", ".join(f"{o:+.4f}" for o in first))
    print("  second read: " + ", ".join(f"{o:+.4f}" for o in second))

    stable = all(abs(a - b) < 1e-6 for a, b in zip(first, second))
    print(f"\n  stable across runs: {stable}")
    if not stable:
        print(
            "  The draw does not repeat, so there is no fixed set of ten to aim at\n"
            "  and evenly spaced headings remain the honest proxy."
        )
        Path(args_cli.out).write_text(json.dumps({"stable": False}, indent=2))
        return 1

    print(f"  spread: {min(first):+.3f}° to {max(first):+.3f}°")
    payload = {
        "stable": True,
        "seed": args_cli.seed,
        "agents": args_cli.agents,
        "jitter_deg": args_cli.jitter_deg,
        "offsets_deg": first,
        "offsets_rad": [math.radians(o) for o in first],
    }
    path = Path(args_cli.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"\n[headings] written to {path}")
    return 0


if __name__ == "__main__":
    status = main()
    import threading

    closing = threading.Thread(target=simulation_app.close, daemon=True)
    closing.start()
    closing.join(timeout=30.0)
    sys.stdout.flush()
    os._exit(status)
