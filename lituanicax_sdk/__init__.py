"""LituanicaX racing SDK.

The SDK gives you a car, a ground plane, a track and physics — and tells you
everything measurable about the car at every step. What the policy sees, what
it is paid for and when an episode ends is code you write.

**Locked** (this package): the MuSHR nano v2 and its motor, brakes, steering
and tyres; the simulation rate; the official track; and lap timing. You can
read every one of those values — :class:`~lituanicax_sdk.state.CarState`
exposes them — you just cannot change them, because a car that differs between
teams makes lap times meaningless.

**Yours** (``teamcode/``): observations, reward, terminations, spawning,
your own tracks and objects, and the whole RL side. The SDK ships no rewards
and no observations, deliberately: one it wrote would be one every team shared.

Getting started::

    from lituanicax_sdk import RaceEnv

    class TeamEnv(RaceEnv):
        def compute_observations(self, car):
            return torch.stack([
                car.speed_forward / car.max_speed_m_s,
                car.yaw_rate / 10.0,
                car.cross_track_error / 0.3,
            ], dim=-1).clamp(-1.0, 1.0)

        def compute_reward(self, car):
            return car.speed_forward / car.max_speed_m_s * car.step_dt

See ``README.md`` for what ``car`` gives you, and ``teamcode/env.py``
for a complete worked example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

from . import rules, submit, tracks
from ._locked import LockedParameterError, sdk_fingerprint, verify_integrity
from .spawn import SpawnManager
from .state import CarState
from .submit import SubmissionOutcome
from .timing import LapTimer
from .track import Track, TrackCfg
from .vehicle import TIMING, VEHICLE

if TYPE_CHECKING:
    # Only for type checkers and editors: at runtime these come from the lazy
    # __getattr__ below, because importing them needs a running simulator.
    from .env import RaceEnv, RaceEnvCfg

__version__ = "1.0.0"


def __getattr__(name: str):
    """Expose the Isaac Lab-dependent pieces lazily.

    ``env`` pulls in Isaac Lab, which cannot be imported before the simulator
    has been launched. Keeping it behind a lazy attribute means the vehicle
    spec, the track maths and the lap timer stay importable — and testable — on
    their own.
    """
    if name in ("RaceEnv", "RaceEnvCfg"):
        from . import env

        return getattr(env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


gym.register(
    id="LituanicaX-Race-v0",
    entry_point="lituanicax_sdk.env:RaceEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "lituanicax_sdk.env:RaceEnvCfg"},
)

__all__ = [
    "CarState",
    "LapTimer",
    "LockedParameterError",
    "RaceEnv",
    "RaceEnvCfg",
    "SpawnManager",
    "SubmissionOutcome",
    "TIMING",
    "Track",
    "TrackCfg",
    "VEHICLE",
    "__version__",
    "rules",
    "sdk_fingerprint",
    "submit",
    "tracks",
    "verify_integrity",
]
