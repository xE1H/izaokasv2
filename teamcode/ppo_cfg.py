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
    #: 800 steps is 26.7 s of driving per environment per iteration, and it buys
    #: nothing: RSL-RL truncates GAE at the rollout boundary and bootstraps with
    #: ``V(s_T)``, and because ``is_finite_horizon`` is False the timeout
    #: bootstrap is correct too, so there is no requirement that a lap fit inside
    #: one rollout. What it costs is enormous — at 3072 environments it is 2.5 M
    #: samples fed to 30 gradient steps, so most of an expensive simulation is
    #: spent generating data the optimiser barely reads. 128 steps collects 393 k
    #: samples and takes 20 gradient steps on them, which is roughly six times
    #: more policy improvement per GPU-hour at the same sample count.
    num_steps_per_env: int = 128
    #: 50 was a smoke-test budget. With the shorter rollout an iteration is a few
    #: seconds of simulation rather than two minutes, so this is the same order
    #: of wall-clock as before and many more updates.
    max_iterations: int = 1000
    save_interval: int = 25  # save a checkpoint every N iterations

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
        # 64 units was sized for three observations. There are now 40, of which
        # 24 are a curvature preview of the track ahead and two say where on the
        # lap the car is, and the policy is meant to treat each of the thirteen
        # corners differently rather than learn one rule for all of them. Locked
        # alongside the observation width by --resume, so this takes the larger
        # of the two sizes considered.
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # ── The PPO algorithm itself ───────────────────────────────────────────
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        # A seed for the adaptive schedule, not a fixed rate: it moves the rate
        # by 1.5x per update towards `desired_kl` and converges from either
        # side, so starting an order of magnitude low just wastes the first
        # updates finding its way back up.
        learning_rate=5e-4,
        schedule="adaptive",  # auto-tune the learning rate to hit `desired_kl`
        num_learning_epochs=5,  # passes over each batch of collected experience
        num_mini_batches=4,
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
