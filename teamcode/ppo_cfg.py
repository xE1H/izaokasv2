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
    #: 50 was a smoke-test budget. A lap is ~15 s and the policy has to find one
    #: before it can start shaving it, which takes tens of iterations on its own.
    #: At roughly 2 min/iteration on 3072 environments this is about ten hours.
    max_iterations: int = 300
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
        # Explore harder than the baseline. The target is the *fastest* lap out
        # of many attempts, not a dependable one, so the policy has to find a
        # committed line rather than settle into the safe middle of the corridor.
        init_noise_std=1.0,
        actor_obs_normalization=True,  # keep network inputs on a common scale
        critic_obs_normalization=True,
        # 64 units was sized for three observations. There are now 34, twenty-one
        # of which are a curvature preview of the track ahead, and the policy is
        # meant to treat each corner differently rather than learn one rule for
        # all of them.
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    # ── The PPO algorithm itself ───────────────────────────────────────────
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=1e-4,
        schedule="adaptive",  # auto-tune the learning rate to hit `desired_kl`
        num_learning_epochs=6,  # passes over each batch of collected experience
        num_mini_batches=6,
        clip_param=0.2,  # how far the policy may move in a single update
        entropy_coef=0.006,  # higher = keeps exploring for longer
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        # Discount: how much future reward counts right now. At 0.99 and 30 Hz
        # the horizon is about 3 s — enough to see a wall coming and slow for
        # it, nowhere near the ~15 s of a lap, so the policy cannot learn to give
        # up speed *here* for a better exit *there*, which is what driving a lap
        # rather than a sequence of corners actually means.
        #
        # 0.9965 is the value this file already identified as putting a whole lap
        # inside the horizon, and it matters twice over now: the lap-time bonus
        # in `compute_reward` is paid once, at the end of a lap, and at a 3 s
        # horizon a car most of a lap away from that payout cannot see it at all.
        gamma=0.9965,
        lam=0.95,  # GAE lambda: bias/variance trade-off
        desired_kl=0.01,  # target amount of policy change per update
        max_grad_norm=0.75,
    )
