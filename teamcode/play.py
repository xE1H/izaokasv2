"""Watch a trained policy drive, and export it for the real car.

    play                                          # newest checkpoint
    play --num_envs 1                             # one car
    play --checkpoint logs/<run>/model_1000.pt    # a specific one
    play --video                                  # record a clip instead

    python -m teamcode.play --help           # without the helper script

Three flags:

    --num_envs N    how many cars to show at once
    --checkpoint P  which policy to load (default: the newest in logs/)
    --video         record a clip to the run folder

Rendering is on, so ``--enable_cameras`` is set for you — without it Isaac Sim
refuses to produce frames and video recording fails.

As well as driving, this writes deployment-ready copies of the network into the
run folder as ``exported/policy.pt`` (TorchScript) and ``exported/policy.onnx``.

For a lap time, use ``benchmark`` — that is the official measurement.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Isaac Sim has to be launched before anything else can be imported, so the
#  command-line arguments are parsed first and the rest of the imports follow.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

from lituanicax_sdk.runs import HYDRA_QUIET

TASK = "LituanicaX-Race-Team-v0"
VIDEO_LENGTH_STEPS = 1000  # about 33 s at 30 Hz

parser = argparse.ArgumentParser(description="Watch a trained policy drive.")
parser.add_argument(
    "--num_envs", type=int, default=None, help="How many cars to show at once."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Policy to load (default: the most recent in logs/).",
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record a clip of the run."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Playing back always renders, and Isaac Sim will not produce frames without
# this — which shows up as an error the moment you pass --video.
args_cli.enable_cameras = True

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
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import teamcode  # noqa: F401, E402 — importing registers the task
from lituanicax_sdk import RaceEnv  # noqa: E402
from lituanicax_sdk.runs import find_checkpoint  # noqa: E402


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Load a checkpoint, export it, then drive with it until you close the window."""
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = agent_cfg.seed if agent_cfg.seed is not None else -1

    checkpoint = find_checkpoint(args_cli.checkpoint)
    # Everything this script writes goes back into the run the checkpoint came
    # from, so a run folder stays self-contained.
    log_dir = checkpoint.parent
    env_cfg.log_dir = str(log_dir)

    env = gym.make(
        TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    if args_cli.video:
        folder = str(log_dir / "videos" / "play")
        print(f"[play] recording {VIDEO_LENGTH_STEPS} steps to {folder}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=folder,
            step_trigger=lambda step: step == 0,
            video_length=VIDEO_LENGTH_STEPS,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(
        cast(DirectRLEnv, env), clip_actions=agent_cfg.clip_actions
    )
    race = cast(RaceEnv, env.unwrapped)

    # ── Load the trained policy ───────────────────────────────────────────
    print(f"[play] loading {checkpoint}")
    runner = OnPolicyRunner(
        env, class_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device
    )
    runner.load(str(checkpoint))

    policy = runner.get_inference_policy(device=race.device)
    network = runner.alg.policy

    # The observation normaliser has to travel with the exported network, or
    # the exported policy would see raw, unscaled numbers.
    normalizer = getattr(network, "actor_obs_normalizer", None)

    # ── Export for deployment (TorchScript + ONNX) ────────────────────────
    export_dir = str(log_dir / "exported")
    export_policy_as_jit(
        network, normalizer=normalizer, path=export_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        network, normalizer=normalizer, path=export_dir, filename="policy.onnx"
    )
    print(f"[play] exported to {export_dir}")

    # ── Drive ─────────────────────────────────────────────────────────────
    step_dt = race.step_dt
    lap_timer = race.lap_timer
    obs = env.get_observations()
    timestep = 0
    laps_seen = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            network.reset(dones)

        # Report laps as they are set. `benchmark` is the tool for an official
        # time; this is just so you can see what you are watching.
        total = lap_timer.summary()["laps_completed"]
        if total > laps_seen:
            laps_seen = total
            print(
                f"[lap] {float(lap_timer.finished_laps_s.min()):6.3f} s   "
                f"(best {float(lap_timer.best_lap_time_s):.3f} s, {total} laps)"
            )

        if args_cli.video:
            timestep += 1
            if timestep >= VIDEO_LENGTH_STEPS:
                break

        # Play back at wall-clock speed — this is for watching, so running
        # faster than real time would only make it harder to see.
        remaining = step_dt - (time.time() - start_time)
        if remaining > 0:
            time.sleep(remaining)

    env.close()


if __name__ == "__main__":
    # Hydra fills in env_cfg and agent_cfg; the decorator hides that from static
    # analysis, so main() genuinely does take no arguments here.
    main()  # type: ignore[call-arg]
    simulation_app.close()
