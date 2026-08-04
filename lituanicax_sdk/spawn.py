"""Where cars start each episode.

Open. Spawning is a training decision — it is your curriculum — and where your
cars start is yours to decide in ``teamcode``.

The SDK's own default is deliberately the dullest one there is:
:class:`SpawnManager` puts every car on the **world origin**, facing along the
track. That is one hardcoded point, the same one the benchmark scores from, and
it ships no opinion about which corners are worth practising.

Two ways to do better, both from your side of the fence:

* :class:`PresetSpawnManager` — a list of poses you choose, each usable in
  either direction, with the two directions balanced adaptively: whichever is
  doing *worse* gets more cars. Because each point is used both ways, one
  direction otherwise ends up easier and dominates the learning signal. The
  baseline solution uses this, and the points live in ``teamcode/env.py``.
* Subclass either one and set ``RaceEnvCfg.spawn_manager``. Randomising along
  the centerline, or starting cars at speed rather than from rest, are both
  reasonable things to try.

What you cannot change is where the lap clock starts and stops — see
:mod:`lituanicax_sdk.timing`.
"""

from __future__ import annotations

import math

import torch

from .track import Track

#: A spawn point further than this from the centerline is probably a mistake,
#: so :class:`SpawnManager` says so rather than dropping cars into a wall in
#: silence.
OFF_TRACK_WARNING_M = 1.0


class SpawnManager:
    """Puts every car on one point, by default the world origin (0, 0).

    The SDK's default and the benchmark's: one fixed pose, so a run measures
    the policy rather than the spawn distribution. The point has to be on the
    track — the origin is, on the official one, right on the centerline.

    Unless a heading is given, :meth:`setup` works one out from the track's
    tangent at that point, so this is not tied to one track's coordinates.

    Args:
        xy: where to put the cars, in world metres.
        yaw_deg: which way to face. ``None`` follows the track.
        jitter_rad: random heading noise, ``0`` for an identical pose every time.
        height_m: how far above the ground to drop the car.
    """

    # Created in setup(), once the track and the device are known — declared
    # here rather than assigned None so the rest of the class can read them as
    # the tensors they always are by the time anything calls it.
    _track: Track
    _xy: torch.Tensor
    _yaw: torch.Tensor

    def __init__(
        self,
        xy: tuple[float, float] = (0.0, 0.0),
        yaw_deg: float | None = None,
        jitter_rad: float = 0.0,
        height_m: float = 0.002,
    ):
        self.xy = xy
        self.yaw_deg = yaw_deg
        self.jitter_rad = jitter_rad
        self.height_m = height_m

    def setup(self, track: Track, num_envs: int, device: torch.device | str) -> None:
        """Called once by the environment, after the track is loaded."""
        self._track = track
        self._xy = torch.tensor(self.xy, dtype=torch.float32, device=device)

        nearest_idx, distance = track.nearest(self._xy.unsqueeze(0))
        if float(distance[0]) > OFF_TRACK_WARNING_M:
            print(
                f"[spawn] warning: {self.describe()} is {float(distance[0]):.2f} m "
                f"from the centerline of '{track.cfg.name}'. Cars start where they "
                "are put, walls and all."
            )

        if self.yaw_deg is None:
            tangent = track.tangents[nearest_idx[0]]
            self._yaw = torch.atan2(tangent[1], tangent[0])
        else:
            self._yaw = torch.deg2rad(torch.tensor(self.yaw_deg, device=device))

    def sample(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose a pose for each respawning car.

        Returns:
            ``(xy, yaw)`` — ``[n, 2]`` world positions and ``[n]`` headings.
        """
        n = len(env_ids)
        device = self._track.device
        xy = self._xy.unsqueeze(0).expand(n, 2).clone()
        jitter = (torch.rand(n, device=device) - 0.5) * (2.0 * self.jitter_rad)
        yaw = torch.full((n,), float(self._yaw), device=device) + jitter
        return xy, yaw

    def observe_episode_lengths(
        self, env_ids: torch.Tensor, lengths: torch.Tensor
    ) -> None:
        """Record how long the finishing episodes lasted.

        Nothing to learn from them here — every car starts the same — but a
        curriculum built on top of this will want them, which is why it is a
        method rather than something the environment decides to call.
        """

    def log_dict(self) -> dict[str, torch.Tensor]:
        """Anything worth charting in TensorBoard."""
        return {}

    def pose(self) -> tuple[float, float, float]:
        """The pose every car starts from: ``(x, y, yaw_rad)``.

        Only meaningful after :meth:`setup`, which is where the heading is
        worked out when one was not given.
        """
        return float(self._xy[0]), float(self._xy[1]), float(self._yaw)

    def describe(self) -> str:
        facing = "along the track" if self.yaw_deg is None else f"{self.yaw_deg:g}°"
        if tuple(self.xy) == (0.0, 0.0):
            where = "world origin (0, 0)"
        else:
            where = f"({self.xy[0]:g}, {self.xy[1]:g})"
        return f"{where}, facing {facing}"


class PresetSpawnManager(SpawnManager):
    """Spreads cars over a list of poses you choose, facing either way.

    Yours to configure: the poses are an argument, not a property of the track,
    because which parts of a track are worth practising is a training decision.
    The baseline solution's list lives in ``teamcode/env.py``.

    Each point is used in both directions, and the split between them adapts:
    whichever direction is currently surviving less well gets more cars, so one
    direction cannot quietly dominate the learning signal. As the two even out
    the split converges back towards 50/50.

    Args:
        points: ``(x, y, heading_deg)`` poses in world metres.
        jitter_rad: random heading noise, so every start is slightly different.
        height_m: how far above the ground to drop the car.
        balance_directions: bias spawns towards the direction doing worse.
            Turn off for a fixed 50/50 split.
        ema_alpha: how quickly the per-direction difficulty estimate moves.
            Small: this is measured over thousands of episodes.
    """

    _presets: torch.Tensor
    _flipped: torch.Tensor
    _mean_len_normal: torch.Tensor
    _mean_len_flipped: torch.Tensor

    def __init__(
        self,
        points: list[tuple[float, float, float]],
        jitter_rad: float = 0.1,
        height_m: float = 0.002,
        balance_directions: bool = True,
        ema_alpha: float = 0.005,
    ):
        if not points:
            raise ValueError(
                "PresetSpawnManager needs at least one (x, y, heading_deg) point. "
                "For a single fixed start, SpawnManager() is simpler."
            )
        super().__init__(jitter_rad=jitter_rad, height_m=height_m)
        self.points = points
        self.balance_directions = balance_directions
        self.ema_alpha = ema_alpha

    def setup(self, track: Track, num_envs: int, device: torch.device | str) -> None:
        self._track = track
        self._presets = torch.tensor(self.points, dtype=torch.float32, device=device)
        self._flipped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._mean_len_normal = torch.tensor(100.0, device=device)
        self._mean_len_flipped = torch.tensor(100.0, device=device)

        _, distance = track.nearest(self._presets[:, :2])
        stray = (distance > OFF_TRACK_WARNING_M).nonzero().flatten().tolist()
        for index in stray:
            print(
                f"[spawn] warning: spawn point {index} {self.points[index]} is "
                f"{float(distance[index]):.2f} m from the centerline of "
                f"'{track.cfg.name}'."
            )

    def observe_episode_lengths(
        self, env_ids: torch.Tensor, lengths: torch.Tensor
    ) -> None:
        """Record how long the finishing episodes lasted, per direction.

        Must be called *before* the episode counters are cleared.
        """
        if not self.balance_directions or lengths.numel() == 0:
            return
        was_flipped = self._flipped[env_ids]
        self._mean_len_normal = self._blend(
            self._mean_len_normal, lengths[~was_flipped]
        )
        self._mean_len_flipped = self._blend(
            self._mean_len_flipped, lengths[was_flipped]
        )

    def _blend(self, current: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        if samples.numel() == 0:
            return current
        return (1.0 - self.ema_alpha) * current + self.ema_alpha * samples.mean()

    def sample(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._presets.device
        n = len(env_ids)

        if self.balance_directions:
            # More cars into whichever direction is surviving less well.
            p_flipped = (
                self._mean_len_normal
                / (self._mean_len_normal + self._mean_len_flipped + 1e-6)
            ).clamp(0.05, 0.95)
        else:
            p_flipped = torch.tensor(0.5, device=device)

        is_flipped = torch.rand(n, device=device) < p_flipped
        self._flipped[env_ids] = is_flipped

        chosen = self._presets[torch.randint(len(self._presets), (n,), device=device)]
        xy = chosen[:, :2].clone()

        jitter = (torch.rand(n, device=device) - 0.5) * (2.0 * self.jitter_rad)
        yaw = torch.deg2rad(chosen[:, 2]) + is_flipped * math.pi + jitter
        return xy, yaw

    def log_dict(self) -> dict[str, torch.Tensor]:
        if not self.balance_directions:
            return {}
        return {
            "Spawn/mean_ep_len_forward": self._mean_len_normal.unsqueeze(0).clone(),
            "Spawn/mean_ep_len_reversed": self._mean_len_flipped.unsqueeze(0).clone(),
        }

    def describe(self) -> str:
        return f"{len(self.points)} preset points, either direction"
