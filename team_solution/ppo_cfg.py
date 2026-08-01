"""PPO learning settings. Entirely yours.

These control *how* the car learns, not what it drives — for that see
``env_cfg.py``. The RL side is unconstrained: change any of this, swap RSL-RL
for another library, or write your own trainer. The SDK only fixes the car and
the clock.

Every value here is a knob you can turn; the comments say what turning it does.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TeamPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Settings for the RSL-RL PPO training loop."""

    # One "iteration" = collect `num_steps_per_env` steps in every environment,
    # then run a few passes of gradient descent over that batch of experience.
    num_steps_per_env: int = 800
    max_iterations: int = 500  # can be overridden with --max_iterations
    save_interval: int = 10  # save a checkpoint every N iterations

    # A label for the experiment.  Checkpoints go to logs/<timestamp>/ either
    # way; this only names the run in wandb / neptune, if you enable them.
    experiment_name: str = "team_solution"

    # Which observation group feeds which network.  This environment publishes a
    # single group named "policy", and both networks read from it.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    # ── The neural networks ────────────────────────────────────────────────
    # The "actor" picks the actions; the "critic" estimates how good a
    # situation is, which is used to judge whether an action was better or
    # worse than expected.
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # how randomly the car explores at the start
        actor_obs_normalization=True,  # keep network inputs on a common scale
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256],  # two hidden layers, 512 then 256 neurons
        critic_hidden_dims=[512, 256],
        activation="mish",
    )

    # ── The PPO algorithm itself ───────────────────────────────────────────
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=1e-4,
        schedule="adaptive",  # auto-tune the learning rate to hit `desired_kl`
        num_learning_epochs=6,  # passes over each batch of collected experience
        num_mini_batches=6,
        clip_param=0.2,  # how far the policy may move in a single update
        entropy_coef=0.004,  # higher = keeps exploring for longer
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        gamma=0.9965,  # discount: how much future reward counts right now
        lam=0.95,  # GAE lambda: bias/variance trade-off
        desired_kl=0.01,  # target amount of policy change per update
        max_grad_norm=0.75,
    )
