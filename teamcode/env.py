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

from lituanicax_sdk import CarState, RaceEnv, RaceEnvCfg, rules
from lituanicax_sdk.spawn import SpawnManager
from lituanicax_sdk.tracks import OFFICIAL

#: How far ahead the policy looks, in centerline points. The official track's
#: points are about 5 cm apart, so this is 0.3 m to 5 m.
#:
#: Spaced geometrically rather than evenly, because what the car needs to know
#: is not uniform in distance. The near end exists because the steering servo
#: takes ten control steps to place the wheels — a third of a second, a metre
#: and a half at racing speed — so the policy has to act on a corner before it
#: is in it. The far end exists because braking for a corner starts long before
#: the corner: at 5 m/s and the measured 8.3 m/s² of braking, shedding 2 m/s
#: takes 1.7 m. Even spacing would spend most of its resolution on the straights,
#: where nothing is decided.
LOOKAHEAD_POINTS = [6, 12, 20, 32, 50, 72, 100]

#: How many numbers :meth:`TeamEnv.compute_observations` returns. It has to be
#: declared, because it sizes the policy's input before the simulation starts.
#:
#: **Fixed once and then left alone.** ``train --resume`` cannot load a checkpoint
#: whose observation width differs, so changing this discards every hour already
#: spent. 13 scalars plus (x, y, curvature) at each lookahead point.
OBSERVATION_SPACE = 13 + 3 * len(LOOKAHEAD_POINTS)

#: Lap time the bonus pays out against, seconds. A lap slower than this earns
#: nothing extra. Set above the deterministic controller's verified 14.900 s so
#: every lap worth submitting sits inside the paying part of the curve.
LAP_BONUS_REFERENCE_S = 18.0

#: How far under the reference still pays, seconds. 6 s reaches 12.0 s, which is
#: below anything this car has been shown to do — the ceiling must not be inside
#: the range being competed over, or the reward stops distinguishing the laps
#: that matter most.
LAP_BONUS_MAX_UNDER_S = 6.0

#: Scale on the *squared* seconds under the reference.
#:
#: **Squared on purpose, and this is the whole design of the reward.** The
#: competition scores the single fastest of however many attempts are run, so
#: the median is worth nothing and survival is worth nothing — only the floor
#: counts. PPO maximises *expected* return, which by default produces a careful
#: policy that is good on average, and that is the wrong animal entirely.
#:
#: A reward convex in pace is what makes an expected-return maximiser prefer a
#: gamble to a sure thing: with the square, a policy that laps at 14 s half the
#: time and crashes the other half beats one that laps at 15 s every time.
#: Linear scoring would rank those the other way round. So the curvature here is
#: not tuning — it is the difference between optimising the median and
#: optimising the floor.
#:
#:     lap 16 s -> 4 x 2 =   8      lap 14 s -> 16 x 2 =  32
#:     lap 15 s -> 9 x 2 =  18      lap 13 s -> 25 x 2 =  50
#:
#: Against a distance term that totals about 28 over a full 60 s episode, and a
#: lap that is collected two or three times inside one, pace on a completed lap
#: is decisively the larger prize.
LAP_BONUS_SCALE = 2.0


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

    #: The scored pose, with a narrow band of heading noise around it.
    #:
    #: The benchmark spawns every attempt at the world origin facing the track
    #: tangent and jitters the heading by up to ±5°, and that jitter is the
    #: **only** thing separating one attempt from another — the policy itself
    #: runs deterministically when it is scored. So the shape of this
    #: distribution decides what is being optimised.
    #:
    #: The SDK's default is ``jitter_rad = 0``, which would train every one of
    #: thousands of cars on a single identical pose and then score them across
    #: ±5° — the wrong distribution, and knife-edge besides.
    #:
    #: Training on the *full* ±5° is the obvious correction and it is also
    #: wrong for this competition. It asks PPO to be good on average across the
    #: band, and the average is worth nothing: only the single fastest attempt
    #: is scored. ±1.15° instead, which is wide enough that the policy is not
    #: balanced on a knife edge and narrow enough that it specialises hard on
    #: the pose it is actually judged from. With a large ``--agents`` at scoring
    #: time some attempt lands within a hundredth of a degree of centre, and
    #: that is the one that counts.
    #:
    #: A spread curriculum (:class:`~lituanicax_sdk.spawn.PresetSpawnManager`,
    #: poses all round the lap) was considered and rejected: it teaches corners
    #: the car would otherwise reach only by surviving everything before them,
    #: but it spends capacity on entry conditions that never occur in a scored
    #: run. Revisit it only if the policy stops making progress mid-lap.
    spawn_manager: SpawnManager = SpawnManager(jitter_rad=0.02)


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

        ==========  =================================================================
        Index       Meaning
        ==========  =================================================================
        0           Forward speed
        1           Lateral speed — how much the car is going sideways
        2           Yaw rate
        3           Wheel slip
        4           Distance from the centerline, signed left-positive
        5           Heading error against the track direction
        6           Distance to the nearest wall
        7           Up-axis — 1 upright, 0 on its side
        8           Roll angle
        9, 10       Throttle and steering asked for on the previous step
        11, 12      Distance to the next corner, and how tight it is
        13..33      ``(x, y, curvature)`` of the centerline at each lookahead point
        ==========  =================================================================

        Indices 4 and 5, and the lookahead, all flip automatically when the car
        drives the track the other way round, because ``car`` works out the
        direction of travel from the velocity.

        **The preview is the point.** The baseline's three numbers describe where
        the car is *now* and say nothing about where the track goes next, so it
        cannot see a corner until it is already in one, and the only safe policy
        is to go slowly enough to correct whatever appears. Seven lookahead
        points carrying curvature is what lets a policy brake for a corner it
        cannot yet feel.

        The previous commands are here for a specific measured reason: the
        steering is an effort-limited servo that needs about ten control steps to
        reach what it was asked for, so the wheels are never where the last
        command said. A policy that cannot see what it already asked for is
        guessing at its own actuator state.

        Everything is scaled to roughly ``[-1, 1]``. The four-wheel quantities
        (``wheel_slips``, ``suspension_travel``, ``applied_wheel_torque``) are
        left out deliberately: they quadruple the width for detail that cannot be
        acted on through two actions. So is ``progress_m``, which would invite
        memorising the track by arc length rather than reading what is ahead.
        """
        max_speed = car.max_speed_m_s

        scalars = torch.stack(
            [
                (car.speed_forward / max_speed).clamp(-1.0, 1.0),
                (car.speed_lateral / max_speed).clamp(-1.0, 1.0),
                (car.yaw_rate / 6.0).clamp(-1.0, 1.0),
                car.slip.clamp(-1.0, 1.0),
                (car.signed_cross_track_error / 0.35).clamp(-1.0, 1.0),
                (car.heading_error / (0.5 * math.pi)).clamp(-1.0, 1.0),
                (car.dist_to_wall / 0.35).clamp(0.0, 1.0),
                car.up_axis.clamp(-1.0, 1.0),
                (car.roll / 0.5).clamp(-1.0, 1.0),
                car.throttle_cmd_prev.clamp(-1.0, 1.0),
                car.steer_cmd_prev.clamp(-1.0, 1.0),
                (car.dist_to_next_corner / 5.0).clamp(0.0, 1.0),
                (car.next_corner_curvature / 3.0).clamp(-1.0, 1.0),
            ],
            dim=-1,
        )

        # [N, len(LOOKAHEAD_POINTS), 3] of (forward, left, curvature) in the
        # car's own frame. The furthest point is 5 m away, so metres divide by 5;
        # curvature is up to about 3 1/m on the tightest corner here.
        preview = car.lookahead(LOOKAHEAD_POINTS)
        preview = torch.stack(
            [
                (preview[..., 0] / 5.0).clamp(-1.0, 1.0),
                (preview[..., 1] / 5.0).clamp(-1.0, 1.0),
                (preview[..., 2] / 3.0).clamp(-1.0, 1.0),
            ],
            dim=-1,
        )

        return torch.cat([scalars, preview.flatten(start_dim=1)], dim=-1)

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
        distance = car.speed_forward / car.max_speed_m_s * car.step_dt

        # What is actually being scored, paid once per clean lap, and squared in
        # pace so that going faster is worth disproportionately more.
        #
        # The distance term above pays for average speed, which is not the same
        # thing as a fast lap — and worse, it pays a car that survives without
        # ever completing one. It is here because it is the only dense signal
        # and nothing would ever reach the first corner without it, not because
        # covering ground is the goal.
        #
        # `just_finished` is true only on the step a *valid* lap closed, and the
        # SDK has already invalidated any lap with a wall touch in it, so this
        # cannot be collected by cutting a corner into the barrier.
        under = (LAP_BONUS_REFERENCE_S - car.lap.last_time_s).clamp(
            0.0, LAP_BONUS_MAX_UNDER_S
        )
        lap_bonus = car.lap.just_finished.float() * under**2 * LAP_BONUS_SCALE

        # Shaping, deliberately small. These say *how* to drive and must never be
        # worth earning on their own — the classic failure is a term a car can
        # farm by not racing. Kept lighter than they would be for a policy meant
        # to look tidy: a car being asked to find the fastest lap it can should
        # be allowed to saw at the wheel and slide if that is what is quick.
        steer_rate = (car.steer_cmd - car.steer_cmd_prev) ** 2
        penalty = 0.01 * steer_rate + 0.005 * car.slip**2

        reward = distance + lap_bonus - penalty

        # Logged apart so TensorBoard says which term is moving the policy. If
        # Rewards/lap_bonus stays flat at zero, no car is completing a lap and
        # the sparse term is doing nothing — that is the signal to look at.
        self.log("Rewards/distance", distance)
        self.log("Rewards/lap_bonus", lap_bonus)
        self.log("Rewards/penalty_steer_rate", 0.02 * steer_rate)
        self.log("Rewards/penalty_slip", 0.01 * car.slip**2)
        return reward

    # ══════════════════════════════════════════════════════════════════════
    #  When an episode ends early
    # ══════════════════════════════════════════════════════════════════════

    def compute_terminations(self, car: CarState) -> torch.Tensor:
        """End the episode early, ``[num_envs]`` boolean.

        This is a training decision, not a rule, which is why the SDK leaves it
        here — and the decision taken is to **train on precisely the rules that
        score the car**, by calling the SDK's own predicates rather than
        restating them.

        That is not just tidiness. A policy trained against a private copy of
        the crash rules is a policy trained for a slightly different
        competition, and the divergences are easy to introduce and hard to see:
        the baseline gated wall contact behind the 8-step spawn grace but would
        have ended a rolled-over car's episode during it, and used an
        instantaneous stall test where the real rule needs 45 *consecutive* slow
        steps. The second one bites exactly the behaviour being trained for — a
        car braking hard into a hairpin dips under the threshold for a few steps,
        and an instantaneous test kills the episode and teaches it not to brake.

        So: :func:`~lituanicax_sdk.rules.official_terminations` covers wall
        contact and rollover with the grace period applied to both, and
        :func:`~lituanicax_sdk.rules.stalled` counts consecutive slow steps.
        Ending the episode on these is still a choice — a lap with a wall touch
        never counts either way, so a crashed car could be left to drive and
        learn recovery — but a car scraping along a barrier for the rest of a
        minute is samples that teach nothing.

        During a measured run the SDK applies these same rules itself and this
        method is not called at all, so training and scoring cannot drift apart.
        """
        # Wall contact and rollover, on exactly the terms a scored attempt uses:
        # `rules.official_terminations` is `(wall_touched | flipped) & past
        # grace`, with the grace period covering the car settling onto its
        # suspension after a respawn.
        #
        # Called rather than restated. The thresholds — 0.15 m to a wall, an
        # up-axis of 0.3, 8 steps of grace — are the SDK's numbers, and a policy
        # trained against a private copy of them is a policy trained for a
        # slightly different competition. This also fixes a real trap: gating
        # the wall on the grace period but not the rollover would end episodes
        # during the settle, where a scored attempt would keep driving.
        crashed = rules.official_terminations(car)

        # Gave up. The official rule needs 45 *consecutive* slow steps, and the
        # count is worth matching rather than approximating with an
        # instantaneous test: a policy being pushed for the fastest possible lap
        # brakes hard into the tight corners, and an instantaneous threshold
        # would end those episodes and teach it not to. `_stall_steps` is the
        # SDK's own buffer — allocated always, cleared on every respawn, and
        # advanced by the SDK only under official rules, which is the one mode
        # where this method is not called. So there is no double counting.
        self._stall_steps, stalled = rules.stalled(car, self._stall_steps)

        self.log("Terminations/crashed", crashed.float())
        self.log("Terminations/stalled", stalled.float())
        return crashed | stalled
