"""The simulation harness: a minimal ``RaceEnv`` that lets an arbitrary driver drive.

**Needs Isaac Sim.** Import this only after ``AppLauncher`` has started the
simulator — see the entry points in :mod:`tools.probe` and
:mod:`teacher.optimize` for the pattern the SDK's own scripts use.

The SDK's environment is built to be driven by a policy: it hands
``compute_observations`` a :class:`~lituanicax_sdk.state.CarState` and expects an
observation vector back. Phase 0-2 has no policy, so :class:`HarnessEnv` publishes
a one-element dummy observation and *keeps the car state*, which the driving loop
then feeds to the controller.

That is not a workaround, it is the enforcement mechanism. The controller sees
exactly what the SDK is willing to show a policy and nothing else, which is the
property that makes its demonstrations learnable in Phase 3.

Two things here are load-bearing and easy to get wrong:

* **Collision filtering is only automatic on the GPU.** ``clone_environments``
  passes ``enable_env_ids=filter_collisions if device != "cpu" else False``
  (``IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py:236``), and
  nothing in the SDK calls ``scene.filter_collisions()``. On CPU, every car in the
  population shares one 0.7 m corridor *and collides with the others*, which
  silently invalidates every lap time. :func:`make_env` refuses unless told.
* **The episode length is the attempt window.** The benchmark does not override
  it, so ``T_teacher`` has to be measured at whatever the team config ships (60 s)
  even though the search runs a shorter window to go faster.
"""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from lituanicax_sdk import RaceEnv, RaceEnvCfg
from lituanicax_sdk.spawn import SpawnManager
from lituanicax_sdk.state import CarState
from lituanicax_sdk.track import TrackCfg
from lituanicax_sdk.tracks import OFFICIAL

#: The dummy observation width. It has to be at least 1 — the SDK rejects 0,
#: because the number sizes a policy's input before the simulation starts.
OBSERVATION_SPACE = 1

#: Attempt window for the *search*. A valid lap on the official track is 12-16 s,
#: so 25 s is generous for a lap and kills stragglers early. Shortening this is
#: the single biggest lever on generation time: at 60 s a generation is minutes
#: of simulating cars that already crashed.
SEARCH_EPISODE_S = 25.0

#: Attempt window the *score* uses — what ``TeamRaceEnvCfg`` ships and therefore
#: what the benchmark will actually give a car. ``T_teacher`` is measured here.
OFFICIAL_EPISODE_S = 60.0


@configclass
class HarnessEnvCfg(RaceEnvCfg):
    """A ``RaceEnvCfg`` with the observation width the harness needs."""

    observation_space: int = OBSERVATION_SPACE
    episode_length_s: float = SEARCH_EPISODE_S


class HarnessEnv(RaceEnv):
    """A ``RaceEnv`` that publishes no observations and keeps the car state.

    ``compute_observations`` is the SDK's hook that runs *after* the post-step
    respawn (``DirectRLEnv.step`` calls ``_reset_idx`` at line 378 and
    ``_get_observations`` at 389), so the state stashed here is the state a real
    policy would have acted on — freshly reset cars included. Reading the state
    from ``_get_dones`` instead would hand a just-crashed car its pre-crash
    position.
    """

    #: The most recent snapshot. Set during ``reset()`` and on every ``step()``.
    latest_car: CarState

    def compute_observations(self, car: CarState) -> torch.Tensor:
        self.latest_car = car
        return torch.zeros(self.num_envs, OBSERVATION_SPACE, device=self.device)

    def compute_reward(self, car: CarState) -> torch.Tensor:
        # Nothing is learning here; lap time is measured by the clock, not paid for.
        return torch.zeros(self.num_envs, device=self.device)

    def compute_terminations(self, car: CarState) -> torch.Tensor:
        # Only consulted when official rules are off — the probes want a car that
        # keeps going whatever it does, so nothing ends an episode but the clock.
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


class FixedPoses(SpawnManager):
    """Put each environment on its own pose, deterministically.

    The SDK's spawners either put every car on one point or sample from a list at
    random. The probes need neither: each measurement is a different manoeuvre
    that wants its own starting place and heading, and all of them in one
    simulation — because Isaac Sim takes minutes to start and the alternative is
    paying that once per measurement.

    Args:
        poses: ``[(x, y, yaw_rad), ...]``, one per environment. Tiled if there are
            more environments than poses.
    """

    _poses: torch.Tensor

    def __init__(self, poses, *, height_m: float = 0.002):
        super().__init__(jitter_rad=0.0, height_m=height_m)
        self.poses = [tuple(float(v) for v in pose) for pose in poses]
        if not self.poses:
            raise ValueError("FixedPoses needs at least one pose.")

    def setup(self, track, num_envs, device) -> None:
        self._track = track
        self._poses = torch.tensor(self.poses, dtype=torch.float32, device=device)
        self._xy = self._poses[0, :2]
        self._yaw = self._poses[0, 2]

    def sample(self, env_ids: torch.Tensor):
        chosen = self._poses[env_ids % len(self.poses)]
        return chosen[:, :2].clone(), chosen[:, 2].clone()

    def describe(self) -> str:
        return f"{len(self.poses)} fixed per-environment poses"


def make_env(
    *,
    num_envs: int,
    track: TrackCfg = OFFICIAL,
    spawn: SpawnManager | None = None,
    official_rules: bool = False,
    stall_rule: bool | None = None,
    episode_length_s: float = SEARCH_EPISODE_S,
    seed: int = 0,
    device: str | None = None,
    allow_cpu: bool = False,
) -> HarnessEnv:
    """Build a harness environment.

    Args:
        num_envs: how many cars. For a search this is candidates x starts.
        track: which track. ``OFFICIAL`` for anything being scored.
        spawn: where cars start. Defaults to the SDK's own — the world origin,
            facing along the track, which is also where a scored attempt starts.
        official_rules: apply :mod:`lituanicax_sdk.rules` instead of the (empty)
            ``compute_terminations``, and freeze a crashed car on the spot. On for
            anything being scored; off for the probes, which need a car that
            keeps driving.
        stall_rule: cut off a car that stops making progress. Defaults to
            ``official_rules``, matching what the benchmark does.
        episode_length_s: the attempt window.
        device: ``None`` leaves Isaac Sim's default, which is ``cuda:0``.
        allow_cpu: run on CPU anyway, accepting that cars will collide with each
            other. Only ever right for a single-car debug run.

    Returns:
        A constructed :class:`HarnessEnv`, already reset is *not* implied — call
        :meth:`HarnessEnv.reset` before driving.
    """
    cfg = HarnessEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.track = track
    cfg.episode_length_s = float(episode_length_s)
    cfg.seed = int(seed)
    cfg.enforce_official_rules = bool(official_rules)
    cfg.official_stall_rule = (
        bool(official_rules) if stall_rule is None else bool(stall_rule)
    )
    # The attempt is timed against the spawn point by AttemptTimer, not against
    # the track's start/finish line, so the SDK's own lap timer must not also be
    # ending episodes — the two would disagree. Matches benchmark.py:250.
    cfg.terminate_on_lap = False
    if spawn is not None:
        cfg.spawn_manager = spawn
    if device is not None:
        cfg.sim.device = device

    resolved = str(cfg.sim.device)
    if resolved.startswith("cpu") and not allow_cpu:
        raise RuntimeError(
            f"sim.device is {resolved!r}, and on CPU Isaac Lab does not filter "
            "collisions between environments — every car would share one 0.7 m "
            "corridor and crash into the others, so every lap time would be "
            "meaningless. Use a CUDA device, or pass allow_cpu=True if you are "
            "deliberately debugging one car."
        )
    if num_envs > 1 and resolved.startswith("cpu"):
        print(
            f"[harness] WARNING: {num_envs} cars on CPU will collide with each "
            "other. Lap times from this run are not comparable."
        )

    return HarnessEnv(cfg)


def drive(
    env: HarnessEnv,
    driver,
    *,
    steps: int | None = None,
    on_step=None,
) -> int:
    """Reset, then drive to the end of the episode.

    Args:
        env: a harness environment.
        driver: called with the current ``CarState``, returns ``[N, 2]`` actions.
        steps: how many policy steps. Defaults to one full episode.
        on_step: optional ``(step, car, actions, dones) -> bool``. Return True to
            stop early — the probes use this to detect a finished measurement, and
            the evaluator to stop once every attempt has settled.

    Returns:
        How many steps were taken.
    """
    env.reset()
    car = env.latest_car
    limit = int(env.max_episode_length) if steps is None else int(steps)

    for step in range(limit):
        actions = driver(car)
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        car = env.latest_car
        if on_step is not None and on_step(step, car, actions, dones):
            return step + 1
    return limit
