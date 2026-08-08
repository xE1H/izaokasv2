"""Your environment. This is the file you work in.

Three methods, all yours:

* :meth:`compute_observations` — what the policy sees
* :meth:`compute_reward` — what it is paid for
* :meth:`compute_terminations` — when an episode is cut short

Each is handed a ``car`` (a :class:`~lituanicax_sdk.state.CarState`) holding
every measurable property of the vehicle and where it is on the track, in SI
units. Nothing below is imposed by the SDK.

**What is here is deliberately the least that works.** Three observations, one
reward term, two terminations, every car starting from the same point. It gets
round the track and it is slow — that is the point. It is a floor to beat, not
a solution to tune, and the fastest way to improve it is to add to it rather
than to adjust it.

``TeamRaceEnvCfg`` below says where the cars drive and for how long. It lives
in this file rather than a separate one because it is three settings, and they
are only meaningful next to the code they configure.

What you *cannot* change is the car itself: the mass, the motor, the brakes,
the steering, the tyres and the simulation rate are the same for every team, so
that a lap time says something about the policy rather than the vehicle. You
can read all of them (``car.max_speed_m_s``, ``car.drive_torque_nm``, …); see
``README.md``.
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils import configclass

from lituanicax_sdk import CarState, RaceEnv, RaceEnvCfg
from lituanicax_sdk.tracks import OFFICIAL

#: How many numbers :meth:`TeamEnv.compute_observations` returns. It has to be
#: declared, because it sizes the policy's input before the simulation starts.
OBSERVATION_SPACE = 3


# ══════════════════════════════════════════════════════════════════════════
#  Where the cars drive
# ══════════════════════════════════════════════════════════════════════════


@configclass
class TeamRaceEnvCfg(RaceEnvCfg):
    """Scene and run settings for the baseline solution."""

    observation_space: int = OBSERVATION_SPACE

    #: 60 s at 30 Hz = 1800 policy steps. This is also the window a scored
    #: attempt gets, so shortening it can cut off a slow lap.
    #:
    #: **There is a floor on this, and it is higher than it looks.** A lap is
    #: measured from the *track's* start/finish line, not from where a car
    #: spawned, and the first crossing after a spawn only starts the clock — it
    #: is never itself a timed lap. So before a car can record anything it must
    #: drive the out-lap *and* a full lap: from the origin that is 14.7 m + 50 m
    #: = 65 m, which at the speed this baseline manages is 30-45 s. Anything
    #: shorter and the episode ends mid-lap, the timer resets with it, and
    #: ``Lap/best_lap_time_s`` never appears in TensorBoard however well the
    #: cars are driving. ``benchmark`` is unaffected: it times an attempt from
    #: the spawn point, so it has no out-lap to pay for.
    episode_length_s: float = 60.0

    #: Lap times are only compared on official tracks. Train wherever you like
    #: — register your own in ``teamcode/tracks/``.
    track = OFFICIAL

    #: Where cars start is not set here, so the SDK's default applies: every
    #: car on the world origin, facing along the track, which is also where a
    #: scored attempt starts. That means the policy only ever practises the
    #: track from one place and in one direction.
    #:
    #: Spawning is a curriculum, and it is yours.
    #: :class:`~lituanicax_sdk.spawn.PresetSpawnManager` spreads cars over a
    #: list of poses you choose, each usable facing either way; set
    #: ``spawn_manager`` to one to train on the corners the car keeps losing.


# ══════════════════════════════════════════════════════════════════════════
#  What the policy does
# ══════════════════════════════════════════════════════════════════════════


class TeamEnv(RaceEnv):
    """The baseline solution: 3 observations, 1 reward term, 2 terminations."""

    # ══════════════════════════════════════════════════════════════════════
    #  What the policy sees
    # ══════════════════════════════════════════════════════════════════════

    def compute_observations(self, car: CarState) -> torch.Tensor:
        """Build the observation vector, ``[num_envs, OBSERVATION_SPACE]``.

        =======  ====================================================
        Index    Meaning
        =======  ====================================================
        0        Forward speed of the body
        1        Distance from the centerline, signed left-positive
        2        Heading error relative to the track direction
        =======  ====================================================

        Indices 1 and 2 flip automatically when the car drives the track the
        other way round, because ``car`` works out the direction of travel from
        the velocity.

        **This is the reason the baseline is slow.** Three numbers describing
        where the car is *now*, and nothing at all about where the track goes
        next: it cannot see a corner until it is already off the line in one.
        A policy with no preview has one safe strategy — go slowly enough to
        correct whatever appears — and that is what it learns.

        ``car.lookahead([10, 20, 40, 70, 100])`` returns the centerline ahead
        in the car's own frame, with the curvature at each point, and is the
        single biggest thing missing here. After that: how close the walls are
        (``car.dist_to_wall``), what the next corner is like
        (``car.dist_to_next_corner``, ``car.next_corner_curvature``), whether
        the car is sliding (``car.speed_lateral``, ``car.yaw_rate``,
        ``car.slip``) and what it just did (``car.steer_cmd``,
        ``car.throttle_cmd``). Widen ``OBSERVATION_SPACE`` to match.

        Scale whatever you add so the numbers sit roughly in ``[-1, 1]`` — the
        divisors below are that and nothing more.
        """
        speed = (car.speed_forward / car.max_speed_m_s).clamp(-1.0, 1.0)
        cross_track = (car.signed_cross_track_error / 0.3).clamp(-1.0, 1.0)
        heading_error = (car.heading_error / (0.5 * math.pi)).clamp(-1.0, 1.0)

        return torch.stack([speed, cross_track, heading_error], dim=-1)

    # ══════════════════════════════════════════════════════════════════════
    #  What it is paid for
    # ══════════════════════════════════════════════════════════════════════

    def compute_reward(self, car: CarState) -> torch.Tensor:
        """One number per car per step: how far it got, and nothing else.

        The principle worth keeping even if you change everything else: **only
        covering ground quickly earns real reward.** Anything else belongs as a
        small penalty shaping *how*, because a term that pays for something
        other than progress tends to get farmed — the classic failure is an
        alive bonus big enough that stopping in a safe corner beats racing.

        Note what this already is: distance per step, summed over a fixed
        episode, *is* average speed. This one line pays for full throttle, and
        the baseline is slow in spite of it — because of what it cannot see and
        how little of the future it is trained to care about, not because
        anything here holds it back.

        There are no penalties at all, and it shows: the car saws at the
        steering, spins its wheels off the line and leans into corners hard
        enough to lift a wheel. Small penalties on ``car.slip``, ``car.roll``
        and on how much the commands change from step to step (``car.steer_cmd
        - car.steer_cmd_prev``) are each worth a try — keep them an order of
        magnitude below the distance term, and log them separately with
        :meth:`log` so you can see what each one bought.

        Lap time is deliberately not in here, and cannot be: the SDK measures
        it and you cannot influence it, which is what makes it worth comparing.
        Reward progress with ``car.speed_forward`` or ``car.progress_m``.
        """
        # Distance covered forwards this step, as a fraction of the furthest
        # the car could have gone. Distance rather than speed, so it sums to
        # the same total however the car gets round.
        reward = car.speed_forward / car.max_speed_m_s * car.step_dt

        self.log("Rewards/distance", reward)
        return reward

    # ══════════════════════════════════════════════════════════════════════
    #  When an episode ends early
    # ══════════════════════════════════════════════════════════════════════

    def compute_terminations(self, car: CarState) -> torch.Tensor:
        """End the episode early, ``[num_envs]`` boolean.

        This is a training decision, not a rule, which is why the SDK leaves it
        here. Below: crashing, and giving up.

        Ending on a wall touch is a *choice*. A lap during which the car
        touched a wall never counts either way — the SDK sees to that — so you
        could let a crashed car keep driving and learn to recover. The baseline
        does not, because a car scraping a wall for the rest of the episode is
        a minute of samples teaching it nothing.

        The stall rule is there for the same reason, and it matters more than it
        looks: a car that shuffles back and forth on the spot earns almost
        nothing and never crashes, so without it that is a *stable* thing for
        the policy to settle into and a whole episode of samples that teach
        nothing. Terminating gives the state a value of zero, which makes
        standing still strictly worse than driving.

        Nothing here ends the episode of a car that has rolled onto its roof
        (``car.up_axis < 0.3``), so it sits out the rest of its 60 seconds.
        Cutting that short too is close to free training speed.

        During a measured run the SDK's own crash rules are applied on top of
        whatever this returns, so a recorded lap always means the same thing.
        """
        # Hit a wall. Sticky: once true it stays true for the episode. The
        # grace period covers the car settling onto its suspension at a start.
        crashed = car.wall_touched & (car.episode_step >= 8)

        # Gave up: barely moving, well after the start. 0.04 of top speed is
        # 0.27 m/s, and 45 steps is 1.5 s of grace to get going.
        #
        # Read instantaneously, unlike the SDK's own rule during a measured run,
        # which needs 45 *consecutive* slow steps. What has to be caught here is
        # a car jittering back and forth on the spot: it is above the threshold
        # half the time, so a consecutive count keeps resetting and never fires.
        # The cost is that braking hard mid-lap can also end an episode — if
        # that starts to cost more than the jitter did, count slow steps instead.
        stalled = (car.speed_forward < 0.04 * car.max_speed_m_s) & (
            car.episode_step > 45
        )

        self.log("Terminations/crashed", crashed.float())
        self.log("Terminations/stalled", stalled.float())
        return crashed | stalled
