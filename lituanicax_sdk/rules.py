"""The official rules — what counts as a crash when a lap is being measured.

These are **not** applied while you train. During training you decide entirely
what ends an episode: see
:meth:`~lituanicax_sdk.env.RaceEnv.compute_terminations`. Ending an episode
early is a training decision and the SDK has no business making it for you.
Cutting a car off when it stops making progress saves samples; cutting it off
when it drifts wide teaches a tighter line, and may also teach timidity. Your
call.

When a lap time is being *measured* — the benchmark, or any config with
``enforce_official_rules`` set — these replace whatever you defined, so that
every team's cars fail for exactly the same reasons:

* **Wall contact** — the car's centre comes within
  :data:`WALL_COLLISION_RADIUS_M` of a wall. The car freezes on the spot and
  the episode ends, so a crashed car cannot scrape its way to a better time.
* **Roll-over** — the car is no longer upright.
* **Stall** — the car stays barely moving for too long. This is how a car
  pinned against a wall ends its attempt: its centre can sit just outside the
  crash radius while the body is wedged, so the wall never registers as a
  crash yet the car cannot move. The benchmark turns this on separately (see
  ``RaceEnvCfg.official_stall_rule``), since whether a stalled car is worth
  terminating is a choice some teams make differently during training.

One rule applies at all times, training and measurement alike: **a lap during
which the car touched a wall is not a valid lap.** That is a property of the
lap rather than a termination, so it holds however you have chosen to end your
episodes.
"""

from __future__ import annotations

import torch

from .state import CarState

#: Car centre closer than this to a wall counts as a crash, in metres.
WALL_COLLISION_RADIUS_M = 0.15

#: Below this the car's up-axis no longer points up — roughly 73° of roll.
FLIPPED_UP_AXIS_THRESHOLD = 0.3

#: Crashes are ignored for this many steps after a respawn, while the car
#: settles onto its suspension.
SPAWN_GRACE_STEPS = 8

#: A car is stalled when it stays below this forward speed for too long —
#: a fraction of top speed, so the rule scales with the locked vehicle.
STALL_SPEED_FRACTION = 0.04

#: How many consecutive slow steps count as a stall. 45 steps is 1.5 s at the
#: 30 Hz policy rate: longer than any braking zone, shorter than anything a
#: working car needs to get going.
STALL_AFTER_STEPS = 45


def touching_wall(car: CarState) -> torch.Tensor:
    """True where the car is touching a wall, ``[N]``.

    Measured as the distance from the car to every wall segment rather than
    through physics contact reports: one small tensor operation, and it fires
    the instant the car gets too close instead of a step later.

    Evaluated on every step, because a lap with a wall touch in it is invalid
    whether or not you terminate on it. Use it in your own terminations if you
    want to — ``car.dist_to_wall`` gives you the raw distance.
    """
    return car.dist_to_wall < WALL_COLLISION_RADIUS_M


def flipped(car: CarState) -> torch.Tensor:
    """True where the car has rolled over, ``[N]``."""
    return car.up_axis < FLIPPED_UP_AXIS_THRESHOLD


def official_terminations(car: CarState) -> torch.Tensor:
    """The crash rules as applied during a measured run, ``[N]``."""
    past_grace = car.episode_step >= SPAWN_GRACE_STEPS
    return (car.wall_touched | flipped(car)) & past_grace


def stalled(
    car: CarState, slow_steps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the stall clock and say whether each car has stalled, ``[N]``.

    Args:
        car: the cars being measured.
        slow_steps: each car's count of consecutive steps spent barely moving,
            ``[N]`` long. The caller owns this buffer and zeroes it on respawn.

    Returns:
        ``(slow_steps, stalled)`` — the advanced counts, and ``True`` where a
        count has reached :data:`STALL_AFTER_STEPS`.

    A single slow step resets the count, so braking hard in a corner never
    ends an episode; only *staying* slow does. This is how a measured run
    catches a car that is pinned against a wall without ever being detected as
    a crash — see :mod:`lituanicax_sdk.rules` for why that matters.
    """
    too_slow = car.speed_forward < STALL_SPEED_FRACTION * car.max_speed_m_s
    next_steps = torch.where(too_slow, slow_steps + 1, torch.zeros_like(slow_steps))
    return next_steps, next_steps >= STALL_AFTER_STEPS
