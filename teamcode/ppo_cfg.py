"""PPO learning settings. Entirely yours.

These control *how* the car learns, not what it drives — for that see
``env.py``. The RL side is unconstrained: change any of this, swap RSL-RL
for another library, or write your own trainer. The SDK only fixes the car and
the clock.

Every value here is a knob you can turn; the comments say what turning it does.
None of them can be deleted — RSL-RL requires the lot — so unlike
``teamcode/env.py`` this file is as short as it gets.
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
    max_iterations: int = 50  # can be overridden with --max_iterations
    save_interval: int = 5  # save a checkpoint every N iterations

    # A label for the experiment.  Checkpoints go to logs/<timestamp>/ either
    # way; this only names the run in wandb / neptune, if you enable them.
    experiment_name: str = "teamcode"

    # Which observation group feeds which network.  This environment publishes a
    # single group named "policy", and both networks read from it.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    # ── The neural networks ────────────────────────────────────────────────
    # The "actor" picks the actions; the "critic" estimates how good a
    # situation is, which is used to judge whether an action was better or
    # worse than expected.
    #
    # The networks are small on purpose. Three observations do not need more,
    # and 64 units is about enough to learn "steer towards the line, and go a
    # speed you can correct from" and not enough for a policy that treats each
    # corner differently. Widen them as you widen the observation.
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # how randomly the car explores at the start
        actor_obs_normalization=True,  # keep network inputs on a common scale
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 64],  # two hidden layers, 64 neurons each
        critic_hidden_dims=[64, 64],
        activation="elu",
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
        # Discount: how much future reward counts right now. At 30 Hz this is a
        # horizon of about 3 s — enough to see a wall coming and slow for it,
        # nowhere near the ~10 s of a whole lap, so the policy cannot learn to
        # give up speed *here* for a better exit *there*. That is most of why
        # the baseline drives corner by corner instead of driving a lap. 0.9965
        # is the value that puts a full lap inside the horizon.
        gamma=0.99,
        lam=0.95,  # GAE lambda: bias/variance trade-off
        desired_kl=0.01,  # target amount of policy change per update
        max_grad_norm=0.75,
    )
