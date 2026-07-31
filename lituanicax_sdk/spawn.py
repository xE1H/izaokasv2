"""Where cars start each episode.

Open. Spawning is a training decision — it is your curriculum — and the SDK
only ships a default worth starting from.

:class:`SpawnManager` places cars at the track's preset points, each usable in
either direction, and balances the two directions adaptively: whichever
direction is currently doing *worse* gets more cars. Because each start point
is used both ways, one direction can otherwise end up easier and dominate the
learning signal. As the two even out the split converges back to roughly 50/50.

To do something else, subclass this and set ``RaceEnvCfg.spawn_manager``.
Randomising the position along the centerline instead of using presets, or
starting cars at a speed rather than from rest, are both reasonable things to
try. What you cannot change is where the lap clock starts and stops — that is
the track's start/finish line, not your spawn point.
"""

from __future__ import annotations

import math

import torch

from .track import Track


class SpawnManager:
    """Places respawning cars on the track.

    Args:
        jitter_rad: random heading noise, so every start is slightly different.
        height_m: how far above the ground to drop the car.
        balance_directions: adaptively bias spawns towards the direction that
            is currently doing worse. Turn off for a fixed 50/50 split.
        ema_alpha: how quickly the per-direction difficulty estimate moves.
            Small: this is measured over thousands of episodes.
    """

    # All of this is created in setup(), once the track and the device are
    # known — declared here rather than assigned None in __init__ so that the
    # rest of the class can read them as the tensors they always are by the
    # time anything calls it.
    _track: Track
    _presets: torch.Tensor
    _flipped: torch.Tensor
    _mean_len_normal: torch.Tensor
    _mean_len_flipped: torch.Tensor

    def __init__(
        self,
        jitter_rad: float = 0.1,
        height_m: float = 0.002,
        balance_directions: bool = True,
        ema_alpha: float = 0.005,
    ):
        self.jitter_rad = jitter_rad
        self.height_m = height_m
        self.balance_directions = balance_directions
        self.ema_alpha = ema_alpha

    def setup(self, track: Track, num_envs: int, device: torch.device | str) -> None:
        """Called once by the environment, after the track is loaded."""
        if not track.cfg.spawn_points:
            raise ValueError(
                f"Track '{track.cfg.name}' has no spawn_points, so cars have "
                "nowhere to start."
            )
        self._track = track
        self._presets = torch.tensor(
            track.cfg.spawn_points, dtype=torch.float32, device=device
        )
        self._flipped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._mean_len_normal = torch.tensor(100.0, device=device)
        self._mean_len_flipped = torch.tensor(100.0, device=device)

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
        """Choose a pose for each respawning car.

        Returns:
            ``(xy, yaw)`` — ``[n, 2]`` world positions and ``[n]`` headings.
        """
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
        scale_x, scale_y = self._track.cfg.centerline_scale
        xy = torch.stack([chosen[:, 0] * scale_x, chosen[:, 1] * scale_y], dim=-1)

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


class OriginSpawnManager(SpawnManager):
    """Put every car at the world origin, facing along the track.

    The evaluator uses this so all attempts start from the same point: a fair
    sample of the policy, not of the spawn distribution. Pass ``--spawn-presets``
    to ``evaluate`` to use the track's preset start points instead.

    The origin has to be on the track — it is on the official one, right on
    the centerline. ``setup`` works out the heading from the track's tangent
    there, so the manager is not tied to one track's coordinates. There is no
    positional jitter and, by default, no heading jitter either: every agent
    starts from the exact same pose.
    """

    _yaw: torch.Tensor

    def __init__(self, jitter_rad: float = 0.0):
        super().__init__(jitter_rad=jitter_rad, balance_directions=False)

    def setup(self, track: Track, num_envs: int, device: torch.device | str) -> None:
        self._track = track
        nearest_idx, _ = track.nearest(torch.zeros(1, 2, device=device))
        tangent = track.tangents[nearest_idx[0]]
        self._yaw = torch.atan2(tangent[1], tangent[0])

    def sample(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(env_ids)
        device = self._track.device
        xy = torch.zeros(n, 2, device=device)
        jitter = (torch.rand(n, device=device) - 0.5) * (2.0 * self.jitter_rad)
        yaw = torch.full((n,), float(self._yaw), device=device) + jitter
        return xy, yaw

    def observe_episode_lengths(
        self, env_ids: torch.Tensor, lengths: torch.Tensor
    ) -> None:
        """Nothing to learn from episode lengths: every car starts the same."""

    def log_dict(self) -> dict[str, torch.Tensor]:
        return {}
