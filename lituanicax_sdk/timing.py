"""Lap timing — locked, so that every team's lap times mean the same thing.

This is the number the competition is about, so it is measured by the SDK and
not by the teams.  It is also *only* measured: lap time is never part of the
reward.  Shape your reward however you like; the clock is separate.

A lap runs from the start/finish line back to itself.  That line is a plane
through one point of the track's centerline, at right angles to the track
there — a property of the *track*, not of where a team happens to spawn its
cars.  This is the substantive change from the original implementation, which
measured laps from each car's own spawn point and so could not be compared
between teams that spawn differently.

The line spans the width of the track: "at the line" means being within
``lap_gate_window_m`` of it *along the centerline*, not within some radius of
the centerline point.  Cars race a line, not the centerline, and a gate that
only catches cars passing close to the centerline catches almost nothing.

Three rules keep the numbers honest:

* **Arming** — the car must leave the gate window before a crossing counts, so
  sitting on the line does not tick over laps.
* **Travel** — it must reach the far side of the loop, so nudging over the line
  and reversing back is not a lap.
* **Out-lap** — the first crossing after a spawn only starts the clock.  Cars
  start wherever they start, so that first partial loop is never timed.

Both directions round the track work: crossing is a sign change in the distance
to the plane, and travel is measured the short way round, whichever way the car
went.

Pure tensor maths against a :class:`~lituanicax_sdk.track.Track`, with no Isaac
Lab dependency, so it is tested directly (see ``tests/test_timing.py``).
"""

from __future__ import annotations

import torch

from .track import Track


class LapTimer:
    """Times laps for ``num_envs`` cars on one shared track.

    Call :meth:`update` once per policy step and :meth:`reset` for the cars
    that respawn.  Everything it reports is per environment except
    :attr:`best_lap_time_s`, which is the best seen anywhere in the run.
    """

    #: A lap must reach at least this fraction of the way round the loop.
    MIN_TRAVEL_FRACTION = 0.45

    def __init__(
        self,
        track: Track,
        num_envs: int,
        step_dt: float,
        device: torch.device | str,
        grace_steps: int = 0,
    ):
        self.track = track
        self.num_envs = num_envs
        self.step_dt = step_dt
        self.device = torch.device(device)
        self.grace_steps = grace_steps

        z_bool = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        z_long = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        z_float = torch.zeros(num_envs, device=self.device)

        self._armed = z_bool.clone()  # has left the gate since the last crossing
        self._started = z_bool.clone()  # the out-lap is over, the clock is real
        self._invalid = z_bool.clone()  # something spoiled the lap in progress
        self._start_step = z_long.clone()
        self._prev_offset = z_float.clone()
        self._max_travel = z_float.clone()

        #: Valid laps completed in the current stint, per car. Cleared on
        #: respawn — see :meth:`reset`.
        self.lap_count = z_long.clone()
        #: Time of the last completed lap, per car (0 = none yet). Cleared on
        #: respawn.
        self.last_lap_time_s = z_float.clone()
        #: Best lap this car has driven in the whole run (0 = none yet).
        #: Survives respawns.
        self.best_lap_time_s_per_env = z_float.clone()
        #: Best lap anywhere in the run so far (0 = none yet).
        self.best_lap_time_s = torch.tensor(0.0, device=self.device)

        # Filled in by update(), for the current step.
        self.finished_laps_s = torch.zeros(0, device=self.device)
        self.just_finished = z_bool.clone()

        # Running totals over the whole run, for logging without NaNs.
        self._laps_timed = 0
        self._total_lap_time_s = 0.0

    # ──────────────────────────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear the lap *in progress* for respawning cars.

        The half-driven lap has to go — carrying it across a respawn would time
        a lap the car never drove — and so does ``lap_count``, which counts laps
        in the current stint.

        What survives is everything that describes the *run*: the best lap per
        car, the best anywhere, and the running totals behind
        :meth:`summary`. Otherwise a report at the end of a run would only see
        whichever cars happened not to have crashed recently, and would count a
        fraction of the laps actually driven.
        """
        self._armed[env_ids] = False
        self._started[env_ids] = False
        self._invalid[env_ids] = False
        self._start_step[env_ids] = 0
        self._max_travel[env_ids] = 0.0
        # Zero, not the spawn offset, so the first step after a respawn can
        # never look like a crossing.
        self._prev_offset[env_ids] = 0.0
        self.lap_count[env_ids] = 0
        self.last_lap_time_s[env_ids] = 0.0

    def invalidate(self, mask: torch.Tensor) -> None:
        """Mark the lap in progress as not counting, for the given cars.

        Called when a car touches a wall.  Today that also ends the episode, so
        it rarely changes the outcome, but it keeps lap validity a rule of its
        own rather than an accident of the termination conditions.
        """
        self._invalid |= mask

    def update(
        self,
        pos_xy: torch.Tensor,
        nearest_idx: torch.Tensor,
        episode_step: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the clock one policy step.

        Args:
            pos_xy: ``[N, 2]`` world positions of the cars.
            nearest_idx: ``[N]`` index of the nearest centerline point.
            episode_step: ``[N]`` steps elapsed in the current episode.

        Returns:
            ``[N]`` boolean, true where a valid lap completed on this step.
        """
        track = self.track

        to_gate = pos_xy - track.gate_point
        # Signed distance to the start/finish plane; the sign flips on crossing.
        offset = to_gate @ track.gate_normal
        in_grace = episode_step < self.grace_steps

        # How far round the loop the car is from the line. This is also what
        # decides whether it is *at* the line: a real timing gate spans the
        # track, so being at it means being at that point along the centerline,
        # not within some radius of the centerline point itself. A car taking a
        # wide racing line through the start/finish straight is still crossing
        # the line, and measuring the distance round the loop says so.
        travel = track.loop_distance_from_gate(nearest_idx)
        self._max_travel = torch.maximum(self._max_travel, travel)
        at_gate = travel < track.cfg.lap_gate_window_m

        self._armed |= ~at_gate & ~in_grace
        crossed = (self._prev_offset * offset) < 0.0
        went_far_enough = (
            self._max_travel > self.MIN_TRAVEL_FRACTION * track.track_length
        )
        completing = self._armed & crossed & at_gate & went_far_enough
        # Keep the last *non-zero* offset as the reference. A sample landing
        # exactly on the plane has no sign, and overwriting with zero would
        # swallow the sign change on the following step.
        self._prev_offset = torch.where(
            offset != 0.0, offset.detach(), self._prev_offset
        )

        lap_time_s = (episode_step - self._start_step).float() * self.step_dt
        # The first crossing ends the out-lap: it starts the clock, but is not
        # itself a timed lap.
        timed = completing & self._started & ~self._invalid

        # Every crossing restarts the lap, whether or not it was timed.
        self._armed &= ~completing
        self._max_travel = torch.where(
            completing, torch.zeros_like(self._max_travel), self._max_travel
        )
        self._start_step = torch.where(completing, episode_step, self._start_step)
        self._invalid &= ~completing
        self._started |= completing

        # ── Records ────────────────────────────────────────────────────────
        self.lap_count += timed.long()
        self.last_lap_time_s = torch.where(timed, lap_time_s, self.last_lap_time_s)

        first_here = timed & (self.best_lap_time_s_per_env <= 0.0)
        improved = timed & (lap_time_s < self.best_lap_time_s_per_env)
        self.best_lap_time_s_per_env = torch.where(
            first_here | improved, lap_time_s, self.best_lap_time_s_per_env
        )

        self.finished_laps_s = lap_time_s[timed].detach()
        self.just_finished = timed
        if self.finished_laps_s.numel() > 0:
            self._laps_timed += int(self.finished_laps_s.numel())
            self._total_lap_time_s += float(self.finished_laps_s.sum())
            fastest = self.finished_laps_s.min()
            self.best_lap_time_s = (
                torch.minimum(self.best_lap_time_s, fastest)
                if float(self.best_lap_time_s) > 0.0
                else fastest
            )

        return timed

    # ──────────────────────────────────────────────────────────────────────

    def log_dict(self) -> dict[str, torch.Tensor]:
        """Metrics for this step, in the form RSL-RL expects.

        RSL-RL takes the set of keys from the *first* step of each iteration and
        averages each one over the whole iteration. Two consequences shape what
        is logged here:

        * Logging the per-step tensor of finished laps would average to ``nan``
          on any iteration where no car completed one — which is most of early
          training. So the lap statistics are *running* values, accumulated by
          :meth:`update`, and always finite.
        * The lap keys are withheld until the first lap has ever been timed,
          because a flat zero line before that makes the chart useless. The
          iteration in which the first lap falls will not show them, since they
          were absent on its first step; every iteration after does.
        """
        laps_per_min = (
            self.just_finished.float().sum() / self.num_envs * (60.0 / self.step_dt)
        )
        log = {"Lap/laps_per_min": laps_per_min.unsqueeze(0)}

        # Only meaningful once something has been timed; before that they would
        # be a flat zero line that makes the chart unreadable.
        if self._laps_timed > 0:
            log["Lap/best_lap_time_s"] = self.best_lap_time_s.unsqueeze(0).clone()
            log["Lap/mean_lap_time_s"] = torch.tensor(
                [self.mean_lap_time_s], device=self.device
            )
        return log

    @property
    def mean_lap_time_s(self) -> float:
        """Mean of every valid lap timed so far in the run. Zero if none yet."""
        if self._laps_timed == 0:
            return 0.0
        return self._total_lap_time_s / self._laps_timed

    def summary(self) -> dict[str, float]:
        """A plain-Python snapshot of the whole run, for reports.

        Everything here is a run total, not a snapshot of the current stint:
        ``lap_count`` is cleared whenever a car respawns, so summing it would
        report only the laps of whichever cars happened to be mid-episode at
        the moment the report was taken.
        """
        valid = self.best_lap_time_s_per_env[self.best_lap_time_s_per_env > 0]
        return {
            "laps_completed": self._laps_timed,
            "best_lap_time_s": float(self.best_lap_time_s),
            "mean_lap_time_s": self.mean_lap_time_s,
            "mean_best_lap_time_s": float(valid.mean()) if valid.numel() else 0.0,
            "cars_with_a_lap": int(valid.numel()),
            "track_length_m": self.track.track_length,
        }
