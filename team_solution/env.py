"""Your environment. This is the file you work in.

Three methods, all yours:

* :meth:`compute_observations` — what the policy sees
* :meth:`compute_reward` — what it is paid for
* :meth:`compute_terminations` — when an episode is cut short

Each is handed a ``car`` (a :class:`~lituanicax_sdk.state.CarState`) holding
every measurable property of the vehicle and where it is on the track, in SI
units. Nothing below is imposed by the SDK — it is a worked starting point that
trains, and every line of it is yours to change or delete.

``TeamRaceEnvCfg`` below says where the cars drive and for how long. It lives
in this file rather than a separate one because it is four settings, and they
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
from lituanicax_sdk.spawn import PresetSpawnManager
from lituanicax_sdk.tracks import OFFICIAL

#: How far ahead the lookahead points sit, counted in centerline *points*, not
#: metres. On the official track those are 1.7 to 11.5 cm apart (median 3.7),
#: so these offsets average roughly 0.4, 0.7, 1.5, 2.6 and 3.7 m ahead — and
#: shift as the car moves between the sparse straights and the dense corners.
#: This is the thing that lets the policy brake *before* a corner rather than
#: in it, and is usually the first thing worth changing.
LOOKAHEAD_OFFSETS = [10, 20, 40, 70, 100]

#: Scales, chosen so the numbers the network sees sit roughly in [-1, 1].
MAX_YAW_RATE = 10.0  # rad/s
MAX_CROSS_TRACK = 0.3  # m
MAX_LOOKAHEAD_DIST = 5.0  # m
MAX_CURVATURE = 10.0  # 1/m

#: Reward weights. Only distance earns real reward; the rest are small
#: penalties that shape *how* the car covers it.
W_DISTANCE = 4.0
W_ALIVE = 0.02
W_STEER_DEADZONE = 0.003
W_STEER_RATE = 0.003
W_THROTTLE_RATE = 0.002
W_SLIP = 0.03
W_ROLL = 0.1

STEER_DEADZONE = 0.05  # steering below this much is free
ROLL_REFERENCE_DEG = 15.0  # the lean angle that costs the full roll penalty

#: Terminations.
STALL_SPEED_FRACTION = 0.04  # "too slow" as a fraction of top speed
STALL_AFTER_STEPS = 45  # but only once the car has had time to get going

#: Where cars start, as ``(x, y, heading_deg)`` in world metres. Yours: this is
#: the curriculum, and the SDK has no opinion about it — its own default is the
#: world origin and nothing else. Five points spread round the official track,
#: each used facing either way, so one policy learns it in both directions.
#:
#: The scored attempt always starts at (0, 0), so it is worth keeping a start
#: there, or near it, among whatever else you train on.
SPAWN_POINTS = [
    (0.00, 0.00, 0.0),  # the origin, where `evaluate` starts
    (-0.74, -4.89, 20.8),
    (-3.82, -3.24, -156.0),
    (-3.54, -1.93, -154.5),
    (5.34, -3.62, 0.0),
    (3.52, -4.59, -217.9),
]

#: How many numbers compute_observations returns. Derived from the constants
#: above, so adding a lookahead point does not leave a number to update by hand
#: — but it has to be declared, because it sizes the policy's input before the
#: simulation starts.
OBSERVATION_SPACE = 8 + 3 * len(LOOKAHEAD_OFFSETS)


# ══════════════════════════════════════════════════════════════════════════
#  Where the cars drive
# ══════════════════════════════════════════════════════════════════════════


@configclass
class TeamRaceEnvCfg(RaceEnvCfg):
    """Scene and run settings for the baseline solution."""

    observation_space: int = OBSERVATION_SPACE

    #: 90 s at 30 Hz = 2700 policy steps, enough for several laps.
    episode_length_s: float = 90.0

    #: Lap times are only compared on official tracks. Train wherever you like
    #: — register your own in ``team_solution/tracks/``.
    track = OFFICIAL

    #: Where cars start, from the list above. Each point is used facing either
    #: way, so one policy learns the track in both directions, and the split
    #: adapts: whichever direction is doing worse gets more cars, and the two
    #: converge back towards 50/50.
    #:
    #: The SDK's default is a single car on the world origin. Everything past
    #: that — how many points, where, which way, how much jitter — is yours.
    spawn_manager = PresetSpawnManager(
        points=SPAWN_POINTS, jitter_rad=0.1, balance_directions=True
    )


# ══════════════════════════════════════════════════════════════════════════
#  What the policy does
# ══════════════════════════════════════════════════════════════════════════


class TeamEnv(RaceEnv):
    """The baseline solution: 23 observations, 7 reward terms, one termination."""

    # ══════════════════════════════════════════════════════════════════════
    #  What the policy sees
    # ══════════════════════════════════════════════════════════════════════

    def compute_observations(self, car: CarState) -> torch.Tensor:
        """Build the observation vector, ``[num_envs, OBSERVATION_SPACE]``.

        =======  ====================================================
        Index    Meaning
        =======  ====================================================
        0        Wheel speed
        1        Forward speed of the body
        2        Sideways speed of the body (drift)
        3        Yaw rate
        4        Distance from the centerline
        5        Heading error relative to the track direction
        6        Distance to the next corner
        7        Sharpness of that corner
        8-22     5 lookahead points: (x, y) in the car's own frame,
                 plus the track curvature there
        =======  ====================================================

        Everything from index 4 onwards flips automatically when the car drives
        the track the other way round, because ``car`` works out the direction
        of travel from the velocity. That is what lets one policy handle both
        directions.

        Things worth trying: ``car.dist_to_wall`` (the baseline is blind to how
        close the walls are), and the last action — the reward below penalises
        jerky control without ever telling the policy what it just did.
        """
        # ── 0-3: how the car is moving ─────────────────────────────────────
        wheel_speed = (car.wheel_speed / car.max_speed_m_s).clamp(0.0, 1.0)
        forward_speed = (car.speed_forward / car.max_speed_m_s).clamp(0.0, 1.0)
        lateral_speed = (car.speed_lateral / car.max_speed_m_s).clamp(-1.0, 1.0)
        yaw_rate = (car.yaw_rate / MAX_YAW_RATE).clamp(-1.0, 1.0)

        # ── 4-5: where the car sits on the track ───────────────────────────
        cross_track = (car.cross_track_error / MAX_CROSS_TRACK).clamp(0.0, 1.0)
        heading_error = (car.heading_error / math.pi).clamp(-1.0, 1.0)

        # ── 6-7: the next corner ───────────────────────────────────────────
        corner_dist = (car.dist_to_next_corner / MAX_LOOKAHEAD_DIST).clamp(0.0, 1.0)
        corner_sharpness = (car.next_corner_curvature / MAX_CURVATURE).clamp(0.0, 1.0)

        # ── 8-22: the shape of the track just ahead ────────────────────────
        # [N, k, 3] of (x, y, curvature), x forward and y left in the car frame.
        ahead = car.lookahead(LOOKAHEAD_OFFSETS)
        ahead_xy = (ahead[..., :2] / MAX_LOOKAHEAD_DIST).clamp(-1.0, 1.0)
        ahead_curvature = (ahead[..., 2:] / MAX_CURVATURE).clamp(0.0, 1.0)
        lookahead = torch.cat([ahead_xy, ahead_curvature], dim=-1)

        return torch.cat(
            [
                torch.stack(
                    [
                        wheel_speed,
                        forward_speed,
                        lateral_speed,
                        yaw_rate,
                        cross_track,
                        heading_error,
                        corner_dist,
                        corner_sharpness,
                    ],
                    dim=-1,
                ),
                lookahead.reshape(car.num_envs, -1),
            ],
            dim=-1,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  What it is paid for
    # ══════════════════════════════════════════════════════════════════════

    def compute_reward(self, car: CarState) -> torch.Tensor:
        """One number per car per step: mostly speed, minus a few bad habits.

        The principle worth keeping even if you change everything else: **only
        covering ground quickly earns real reward**, and everything else is a
        small penalty shaping *how*. A term that pays for anything other than
        progress tends to get farmed — the classic failure is an alive bonus
        big enough that stopping in a safe corner beats racing.

        Lap time is deliberately not in here. The SDK measures it and you
        cannot influence it, which is what makes it worth comparing.
        """
        # The main reward: distance covered forwards this step. Distance rather
        # than speed, so it sums to the same total however the car gets round.
        r_distance = W_DISTANCE * car.speed_forward / car.max_speed_m_s * car.step_dt

        # A constant trickle, so simply surviving is always worth something.
        r_alive = torch.full((car.num_envs,), W_ALIVE, device=car.device)

        # Penalty: steering beyond a small free band, which favours committing
        # to a line over constant small corrections.
        excess = (car.steer_cmd.abs().clamp(0.0, 1.0) - STEER_DEADZONE).clamp(min=0.0)
        r_steer = -W_STEER_DEADZONE * excess / max(1.0 - STEER_DEADZONE, 1e-6)

        # Penalty: jerky controls, which a real servo and motor could not follow.
        r_steer_rate = -W_STEER_RATE * (car.steer_cmd - car.steer_cmd_prev).abs()
        r_throttle_rate = (
            -W_THROTTLE_RATE * (car.throttle_cmd - car.throttle_cmd_prev).abs()
        )

        # Penalty: wheels turning faster (or slower) than the car is moving.
        r_slip = -W_SLIP * car.slip.clamp(0.0, 1.0)

        # Penalty: leaning over, which is what precedes a roll-over.
        r_roll = -W_ROLL * torch.rad2deg(car.roll).abs() / ROLL_REFERENCE_DEG

        # Log the parts separately. When the policy does something strange,
        # this is the fastest way to see which term paid for it.
        self.log("Rewards/distance", r_distance)
        self.log("Rewards/alive", r_alive)
        self.log("Rewards/steer_deadzone", r_steer)
        self.log("Rewards/steer_rate", r_steer_rate)
        self.log("Rewards/throttle_rate", r_throttle_rate)
        self.log("Rewards/slip", r_slip)
        self.log("Rewards/roll", r_roll)

        return (
            r_distance
            + r_alive
            + r_steer
            + r_steer_rate
            + r_throttle_rate
            + r_slip
            + r_roll
        )

    # ══════════════════════════════════════════════════════════════════════
    #  When an episode ends early
    # ══════════════════════════════════════════════════════════════════════

    def compute_terminations(self, car: CarState) -> torch.Tensor:
        """End the episode early, ``[num_envs]`` boolean.

        This is a training decision, not a rule, which is why the SDK leaves it
        here. Below: crashing, flipping over, and giving up.

        Note that ending on a wall touch is a *choice*. A lap during which the
        car touched a wall never counts either way — the SDK sees to that — so
        you could let a crashed car keep driving and learn to recover. The
        baseline does not, because a car scraping a wall for 80 seconds is 80
        seconds of samples teaching it nothing.

        During a measured run the SDK's own crash rules are applied on top of
        whatever this returns, so a recorded lap always means the same thing.
        """
        past_grace = car.episode_step >= 8

        # Hit a wall. Sticky: once true it stays true for the episode.
        crashed = car.wall_touched

        # Rolled over: the car's up-axis no longer points up.
        flipped = car.up_axis < 0.3

        # Gave up: barely moving, well after the start.
        stalled = (car.speed_forward < STALL_SPEED_FRACTION * car.max_speed_m_s) & (
            car.episode_step > STALL_AFTER_STEPS
        )

        self.log("Terminations/crashed", crashed.float())
        self.log("Terminations/flipped", flipped.float())
        self.log("Terminations/stalled", stalled.float())

        return (crashed | flipped | stalled) & past_grace
