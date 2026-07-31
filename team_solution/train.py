"""Train the driving policy with PPO.

    train --num_envs 200            # after ./install.sh
    train --num_envs 200 --headless # much faster: no on-screen rendering
    train --resume                  # continue from the newest checkpoint

    python -m team_solution.train --help   # without the helper script

Three flags, and that is all:

    --num_envs N   how many cars to simulate in parallel
    --headless     no on-screen rendering
    --resume       continue from the most recent checkpoint

How *long* it trains, the network size and every other learning setting live in
``team_solution/ppo_cfg.py``. Any of them can be overridden for a single run
without editing the file — ``train --num_envs 200 agent.max_iterations=200``.

Each run writes everything it produces — checkpoints, TensorBoard data, the
exact config used — into one folder, ``logs/<timestamp>/``. TensorBoard starts
automatically and prints its URL.

What the policy sees and what it is paid for is ``team_solution/env.py``. The
car, the physics and the lap clock come from ``lituanicax_sdk/``.

This script lives in ``team_solution/`` rather than at the repository root
because it is yours: how you train is not something the competition fixes.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Isaac Sim has to be launched before anything else can be imported, so the
#  command-line arguments are parsed first and the rest of the imports follow
#  underneath.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

from lituanicax_sdk.runs import HYDRA_QUIET

TASK = "LituanicaX-Race-Team-v0"

parser = argparse.ArgumentParser(description="Train the driving policy with PPO.")
parser.add_argument(
    "--num_envs", type=int, default=None, help="How many cars to simulate at once."
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Continue from the most recent checkpoint.",
)
AppLauncher.add_app_launcher_args(parser)  # --headless, --device, --enable_cameras
args_cli, hydra_args = parser.parse_known_args()

# Hydra parses whatever is left over, so hide our own arguments from it.
sys.argv = [sys.argv[0]] + hydra_args + HYDRA_QUIET

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────────────────────────────────────────────────────────
#  Everything below runs with Isaac Sim already up.
# ─────────────────────────────────────────────────────────────────────────────

import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import webbrowser  # noqa: E402
from typing import cast  # noqa: E402

import gymnasium as gym  # noqa: E402
import rsl_rl.runners.on_policy_runner as on_policy_runner_module  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.utils import store_code_state  # noqa: E402

import team_solution  # noqa: F401, E402 — importing registers the task
from lituanicax_sdk.runs import find_checkpoint, logs_dir, new_run_dir  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _store_code_state_safe(logdir, repositories):
    """RSL-RL saves a git diff alongside each run; ours contains binary files
    (Track.usdc, Track.blend, …) which the plain-text writer chokes on."""
    try:
        return store_code_state(logdir, repositories)
    except (UnicodeEncodeError, UnicodeDecodeError):
        print("[train] Git diff contains binary files — skipping code snapshot.")
        return []


on_policy_runner_module.store_code_state = _store_code_state_safe


def launch_tensorboard(root: str) -> None:
    """Start TensorBoard in the background, then print and open its URL."""
    port = 6006
    for candidate in range(6006, 6020):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("localhost", candidate)) != 0:
                port = candidate  # nothing listening here, so this port is free
                break

    tensorboard_exe = os.path.join(os.path.dirname(sys.executable), "tensorboard")
    subprocess.Popen(
        [tensorboard_exe, "--logdir", root, "--port", str(port), "--host", "0.0.0.0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    url = f"http://localhost:{port}"
    print(f"\n[TensorBoard] Watching : {root}")
    print(f"[TensorBoard] Open at  : {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass  # no browser available (e.g. over SSH) — the URL is printed anyway


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train a PPO agent.

    Hydra hands us two configs: ``env_cfg`` (from ``TeamRaceEnvCfg``) and
    ``agent_cfg`` (from ``TeamPPORunnerCfg``). Overrides that would change a
    locked value — the car, the physics rate — are rejected when the
    environment is built, not silently applied.
    """
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    # Resolve --resume first, while logs/ still holds only older runs.
    resume_path = find_checkpoint() if args_cli.resume else None

    log_dir = new_run_dir()
    env_cfg.log_dir = str(log_dir)
    print(f"[train] logging this run to: {log_dir}")

    launch_tensorboard(str(logs_dir()))

    env = RslRlVecEnvWrapper(
        cast(DirectRLEnv, gym.make(TASK, cfg=env_cfg, render_mode=None)),
        clip_actions=agent_cfg.clip_actions,
    )

    runner = OnPolicyRunner(
        env, class_to_dict(agent_cfg), log_dir=str(log_dir), device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[train] resuming from: {resume_path}")
        runner.load(str(resume_path))

    # Keep a copy of both configs next to the checkpoints, so any run can be
    # reproduced later even if the code has moved on.
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)

    start_time = time.time()
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )
    print(f"[train] training time: {round(time.time() - start_time, 2)} s")

    env.close()


if __name__ == "__main__":
    # Hydra fills in env_cfg and agent_cfg; the decorator hides that from static
    # analysis, so main() genuinely does take no arguments here.
    main()  # type: ignore[call-arg]
    simulation_app.close()
