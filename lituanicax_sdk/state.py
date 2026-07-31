"""``CarState`` — everything you are allowed to know about the car.

This is the SDK's main surface for teams.  Observation terms, reward terms and
termination terms all receive a :class:`CarState` and nothing else.  It is
deliberately a *view*: it hands out numbers, never the underlying
:class:`~isaaclab.assets.Articulation`, so a term can read the simulation but
cannot reach in and change it.

Everything is a tensor with one row per car.  Values are raw SI units — metres,
metres per second, radians — and are *not* normalised, because how you scale
your inputs is your business.  The locked constants you would scale by
(:attr:`max_speed_m_s`, :attr:`wheel_radius_m`, :attr:`max_steer_rad`) are here
too, so you never need to hardcode them.

Everything is computed once per policy step and cached, so reading the same
quantity in five observation terms and a reward costs no more than reading it
once.  The expensive one is :attr:`nearest_idx`, a search over every centerline
point; the old code ran it twice per step and this runs it once.
"""

from __future__ import annotations

from functools import cached_property

import torch

from .track import Track
from .vehicle import VehicleCfg


class CarState:
    """A read-only view of every car in the scene, for one policy step."""

    def __init__(
        self,
        robot,
        track: Track,
        vehicle: VehicleCfg,
        step_dt: float,
        throttle_ids: torch.Tensor,
        steer_ids: torch.Tensor,
        episode_step: torch.Tensor,
        commands: "ControlHistory",
        suspension_ids: torch.Tensor | None = None,
        applied_wheel_torque: torch.Tensor | None = None,
        sensors: dict | None = None,
        wall_touched: torch.Tensor | None = None,
        lap: "LapView | None" = None,
    ):
        self._robot = robot
        self.track = track
        self.vehicle = vehicle
        self.step_dt = step_dt
        self._throttle_ids = throttle_ids
        self._steer_ids = steer_ids
        self._suspension_ids = suspension_ids
        self._applied_wheel_torque = applied_wheel_torque
        self._episode_step = episode_step
        self._commands = commands
        #: Anything you added through ``RaceEnvCfg.extra_scene_entities`` —
        #: ray-casters, cameras, contact sensors — keyed by the name you gave it.
        self.sensors = sensors or {}

        self._wall_touched = wall_touched
        self._lap = lap

    # ══════════════════════════════════════════════════════════════════════
    #  Locked constants — scale your observations by these
    # ══════════════════════════════════════════════════════════════════════

    @property
    def max_speed_m_s(self) -> float:
        """The car's top speed on the flat. A sensible speed normaliser."""
        return self.vehicle.motor_no_load_speed_m_s

    @property
    def wheel_radius_m(self) -> float:
        return self.vehicle.wheel_radius_m

    @property
    def max_steer_rad(self) -> float:
        return self.vehicle.max_steer_rad

    @property
    def mass_kg(self) -> float:
        return self.vehicle.mass_kg

    @property
    def max_wheel_omega(self) -> float:
        """Wheel speed at which the motor stops producing torque, rad/s."""
        return self.vehicle.max_wheel_omega

    @property
    def drive_torque_nm(self) -> float:
        return self.vehicle.drive_torque_nm

    @property
    def wheelbase_names(self) -> tuple[str, ...]:
        """Wheel order used by every per-wheel array here."""
        return self.vehicle.throttle_joints

    @property
    def num_envs(self) -> int:
        return self._robot.data.root_pos_w.shape[0]

    @property
    def device(self) -> torch.device:
        return self._robot.data.root_pos_w.device

    # ══════════════════════════════════════════════════════════════════════
    #  Where the car is
    # ══════════════════════════════════════════════════════════════════════

    @cached_property
    def pos_w(self) -> torch.Tensor:
        """World position, ``[N, 3]``."""
        return self._robot.data.root_pos_w.detach().clone()

    @cached_property
    def pos_xy(self) -> torch.Tensor:
        """World position on the ground plane, ``[N, 2]``."""
        return self.pos_w[:, :2]

    @cached_property
    def quat_w(self) -> torch.Tensor:
        """World orientation as ``(w, x, y, z)``, ``[N, 4]``."""
        return self._robot.data.root_quat_w.detach().clone()

    @cached_property
    def yaw(self) -> torch.Tensor:
        """Heading in the world frame, radians, ``[N]``."""
        w, x, y, z = self.quat_w.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @cached_property
    def roll(self) -> torch.Tensor:
        """Sideways lean, radians, ``[N]``. Leaning hard precedes a roll-over."""
        w, x, y, z = self.quat_w.unbind(dim=-1)
        return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

    @cached_property
    def pitch(self) -> torch.Tensor:
        """Nose-up angle, radians, ``[N]``."""
        w, x, y, z = self.quat_w.unbind(dim=-1)
        return torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))

    @cached_property
    def up_axis(self) -> torch.Tensor:
        """How upright the car is: 1 level, 0 on its side, -1 upside down, ``[N]``."""
        _, x, y, _ = self.quat_w.unbind(dim=-1)
        return 1.0 - 2.0 * (x * x + y * y)

    @cached_property
    def forward_xy(self) -> torch.Tensor:
        """Unit vector the car points along, ``[N, 2]``."""
        return torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], dim=-1)

    # ══════════════════════════════════════════════════════════════════════
    #  How the car is moving
    # ══════════════════════════════════════════════════════════════════════

    @cached_property
    def lin_vel_b(self) -> torch.Tensor:
        """Velocity in the car's own frame (x forward, y left), ``[N, 3]``."""
        return self._robot.data.root_lin_vel_b.detach().clone()

    @cached_property
    def lin_vel_w(self) -> torch.Tensor:
        """Velocity in the world frame, ``[N, 3]``."""
        return self._robot.data.root_lin_vel_w.detach().clone()

    @cached_property
    def ang_vel_b(self) -> torch.Tensor:
        """Angular velocity in the car's own frame, ``[N, 3]``."""
        return self._robot.data.root_ang_vel_b.detach().clone()

    @property
    def speed_forward(self) -> torch.Tensor:
        """Forward speed, m/s, ``[N]``. Negative when reversing."""
        return self.lin_vel_b[:, 0]

    @property
    def speed_lateral(self) -> torch.Tensor:
        """Sideways speed, m/s, ``[N]``. This is drift."""
        return self.lin_vel_b[:, 1]

    @cached_property
    def speed_ground(self) -> torch.Tensor:
        """Speed over the ground regardless of direction, m/s, ``[N]``."""
        return self.lin_vel_w[:, :2].norm(dim=-1)

    @property
    def yaw_rate(self) -> torch.Tensor:
        """How fast the car is turning, rad/s, ``[N]``."""
        return self.ang_vel_b[:, 2]

    # ══════════════════════════════════════════════════════════════════════
    #  Drivetrain
    # ══════════════════════════════════════════════════════════════════════

    @cached_property
    def wheel_omega(self) -> torch.Tensor:
        """Speed of each wheel, rad/s, ``[N, 4]``."""
        return self._robot.data.joint_vel[:, self._throttle_ids].detach().clone()

    @cached_property
    def wheel_speed(self) -> torch.Tensor:
        """Mean wheel rim speed, m/s, ``[N]``.

        Equal to :attr:`speed_forward` when the tyres are gripping; higher when
        they are spinning, lower when they are locked.
        """
        return self.wheel_omega.abs().mean(dim=-1) * self.vehicle.wheel_radius_m

    @cached_property
    def slip(self) -> torch.Tensor:
        """Fractional difference between wheel speed and ground speed, ``[N]``.

        Zero when the tyres grip. Large under wheelspin or lock-up.
        """
        ground = self.speed_forward.clamp(min=0.0)
        return (self.wheel_speed - ground).abs() / ground.clamp(min=1e-3)

    @cached_property
    def wheel_speeds(self) -> torch.Tensor:
        """Rim speed of each wheel, m/s, ``[N, 4]``.

        Ordered as :attr:`wheelbase_names`: rear left, rear right, front left,
        front right. Comparing the four tells you which end is losing grip.
        """
        return self.wheel_omega * self.vehicle.wheel_radius_m

    @cached_property
    def wheel_slips(self) -> torch.Tensor:
        """Slip of each wheel against the car's forward speed, ``[N, 4]``.

        Positive means the wheel is turning faster than the car is moving
        (wheelspin), negative means slower (locking up).
        """
        ground = self.speed_forward.clamp(min=0.0).unsqueeze(-1)
        return (self.wheel_speeds - ground) / ground.clamp(min=1e-3)

    @cached_property
    def applied_wheel_torque(self) -> torch.Tensor:
        """Torque the drive model asked of each wheel this step, Nm, ``[N, 4]``.

        What the motor and brakes are actually doing, after the torque-speed
        curve and the brake fade — not the raw throttle command.
        """
        if self._applied_wheel_torque is None:
            return torch.zeros(self.num_envs, 4, device=self.device)
        return self._applied_wheel_torque.detach().clone()

    @cached_property
    def suspension_travel(self) -> torch.Tensor:
        """Compression of each suspension joint, metres, ``[N, 4]``.

        Loads up under braking and in corners, so it says something about
        weight transfer that the body angles alone do not.
        """
        if self._suspension_ids is None:
            return torch.zeros(self.num_envs, 4, device=self.device)
        return self._robot.data.joint_pos[:, self._suspension_ids].detach().clone()

    @cached_property
    def steer_angle(self) -> torch.Tensor:
        """Actual steering joint angle, radians, ``[N]``.

        The joints are driven in ``tan`` space, so this undoes that: it is the
        angle the wheels are really at, which lags the command.
        """
        joint = self._robot.data.joint_pos[:, self._steer_ids].detach()
        return torch.atan(joint.mean(dim=-1))

    # ── What the policy asked for ─────────────────────────────────────────

    @property
    def throttle_cmd(self) -> torch.Tensor:
        """Throttle the policy commanded this step, ``[-1, 1]``, ``[N]``."""
        return self._commands.throttle

    @property
    def steer_cmd(self) -> torch.Tensor:
        """Steering the policy commanded this step, ``[-1, 1]``, ``[N]``."""
        return self._commands.steer

    @property
    def throttle_cmd_prev(self) -> torch.Tensor:
        return self._commands.throttle_prev

    @property
    def steer_cmd_prev(self) -> torch.Tensor:
        return self._commands.steer_prev

    # ══════════════════════════════════════════════════════════════════════
    #  Raw joints — everything, if the shaped readouts above are not enough
    # ══════════════════════════════════════════════════════════════════════

    @cached_property
    def joint_pos(self) -> torch.Tensor:
        """Position of every joint in the articulation, ``[N, num_joints]``."""
        return self._robot.data.joint_pos.detach().clone()

    @cached_property
    def joint_vel(self) -> torch.Tensor:
        """Velocity of every joint in the articulation, ``[N, num_joints]``."""
        return self._robot.data.joint_vel.detach().clone()

    @property
    def joint_names(self) -> list[str]:
        """Names of the joints, in the order :attr:`joint_pos` uses."""
        return list(self._robot.data.joint_names)

    # ══════════════════════════════════════════════════════════════════════
    #  Where the car is on the track
    # ══════════════════════════════════════════════════════════════════════

    @cached_property
    def _nearest(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.track.nearest(self.pos_xy)

    @property
    def nearest_idx(self) -> torch.Tensor:
        """Index of the closest centerline point, ``[N]``."""
        return self._nearest[0]

    @property
    def cross_track_error(self) -> torch.Tensor:
        """Distance from the centerline, metres, ``[N]``. Always positive."""
        return self._nearest[1]

    @cached_property
    def track_tangent(self) -> torch.Tensor:
        """Direction the track runs at the car's position, ``[N, 2]``."""
        return self.track.tangents[self.nearest_idx]

    @cached_property
    def signed_cross_track_error(self) -> torch.Tensor:
        """Distance from the centerline, signed left-positive, ``[N]``.

        The unsigned :attr:`cross_track_error` cannot tell the policy which way
        to correct; this can.
        """
        offset = self.pos_xy - self.track.points[self.nearest_idx]
        tangent = self.track_tangent
        side = tangent[:, 0] * offset[:, 1] - tangent[:, 1] * offset[:, 0]
        return torch.sign(side) * self.cross_track_error

    @cached_property
    def going_forward(self) -> torch.Tensor:
        """True where the car is travelling along the stored centerline order, ``[N]``.

        Cars are spawned facing both ways, so which way round "forward" is
        depends on the car, not on the track. Every track-relative signal
        derived from this mirrors itself automatically, which is what lets one
        policy drive the track in both directions.
        """
        return (self.lin_vel_w[:, :2] * self.track_tangent).sum(dim=-1) > 0.0

    @cached_property
    def direction_sign(self) -> torch.Tensor:
        """``+1`` along the stored centerline order, ``-1`` against it, ``[N]``."""
        return torch.where(self.going_forward, 1, -1)

    @cached_property
    def heading_error(self) -> torch.Tensor:
        """Angle between where the car points and the track, radians, ``[N]``.

        Signed: positive means the car is aimed to the left of the track.
        Folded to ``±π/2`` so that driving the track backwards reads as
        well-aligned rather than as a maximal error.
        """
        forward, tangent = self.forward_xy, self.track_tangent
        angle = torch.acos((forward * tangent).sum(dim=-1).clamp(-1.0, 1.0).abs())
        side = forward[:, 0] * tangent[:, 1] - forward[:, 1] * tangent[:, 0]
        return angle * torch.sign(side)

    @cached_property
    def progress_m(self) -> torch.Tensor:
        """Distance round the loop from the start/finish line, metres, ``[N]``.

        Measured the short way, so it rises to half the track length at the far
        side and falls again. Free to use in your reward — unlike lap *timing*,
        which the SDK measures and you cannot influence.
        """
        return self.track.loop_distance_from_gate(self.nearest_idx)

    @cached_property
    def dist_to_next_corner(self) -> torch.Tensor:
        """Distance along the track to the next corner, metres, ``[N]``."""
        return self._corner[0]

    @cached_property
    def next_corner_curvature(self) -> torch.Tensor:
        """How sharp that corner is, 1/m, ``[N]``."""
        return self._corner[1]

    @cached_property
    def _corner(self) -> tuple[torch.Tensor, torch.Tensor]:
        track = self.track
        arc_here = track.arc_length[self.nearest_idx].unsqueeze(-1)
        arc_corners = track.arc_length[track.corner_idx].unsqueeze(0)
        ahead = torch.where(
            arc_corners >= arc_here,
            arc_corners - arc_here,
            track.track_length - arc_here + arc_corners,
        )
        # Driving the other way turns "ahead of me" into "behind me".
        ahead = torch.where(
            self.going_forward.unsqueeze(-1), ahead, track.track_length - ahead
        )
        distance, slot = ahead.min(dim=1)
        return distance, track.curvature[track.corner_idx][slot]

    def lookahead(self, offsets: list[int]) -> torch.Tensor:
        """Centerline points ahead of the car, in the car's own frame.

        Args:
            offsets: how far ahead, counted in centerline points. On the
                official track the points are about 5 cm apart, so
                ``[10, 20, 40, 70, 100]`` is roughly 0.5, 1, 2, 3.5 and 5 m.

        Returns:
            ``[N, len(offsets), 3]`` of ``(x, y, curvature)``, where ``x`` is
            forward and ``y`` is left, both in metres. This is what lets a
            policy anticipate a corner instead of only reacting to it.

        Indices step backwards along the stored centerline for cars driving the
        other way round, so "ahead" always means ahead of *this* car.
        """
        track = self.track
        cos_yaw, sin_yaw = torch.cos(self.yaw), torch.sin(self.yaw)
        out = []
        for offset in offsets:
            idx = (self.nearest_idx + self.direction_sign * offset) % track.num_points
            point = track.points[idx]
            dx = point[:, 0] - self.pos_xy[:, 0]
            dy = point[:, 1] - self.pos_xy[:, 1]
            out.append(
                torch.stack(
                    [
                        dx * cos_yaw + dy * sin_yaw,
                        -dx * sin_yaw + dy * cos_yaw,
                        track.curvature[idx],
                    ],
                    dim=-1,
                )
            )
        return torch.stack(out, dim=1)

    @cached_property
    def dist_to_wall(self) -> torch.Tensor:
        """Distance from the car's centre to the nearest wall, metres, ``[N]``."""
        return self.track.distance_to_walls(self.pos_xy)

    # ══════════════════════════════════════════════════════════════════════
    #  Episode and lap
    # ══════════════════════════════════════════════════════════════════════

    @property
    def episode_step(self) -> torch.Tensor:
        """Policy steps since this car last respawned, ``[N]``."""
        return self._episode_step

    @property
    def episode_time_s(self) -> torch.Tensor:
        """Seconds since this car last respawned, ``[N]``."""
        return self._episode_step.float() * self.step_dt

    @cached_property
    def wall_touched(self) -> torch.Tensor:
        """True where the car has hit a wall this episode, ``[N]``.

        Sticky: once a car crashes it stays true until it respawns. The
        environment passes the live buffer, so this reflects a touch the moment
        the physics step records one.
        """
        if self._wall_touched is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._wall_touched

    @property
    def lap(self) -> "LapView":
        """Lap timing for each car. Read-only — the SDK owns the clock."""
        if self._lap is None:
            raise RuntimeError(
                "Lap timing is only available on a CarState built by a running "
                "RaceEnv; this one was constructed without a lap timer."
            )
        return self._lap


class ControlHistory:
    """The last two commands from the policy, kept for smoothness penalties."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.throttle = torch.zeros(num_envs, device=device)
        self.steer = torch.zeros(num_envs, device=device)
        self.throttle_prev = torch.zeros(num_envs, device=device)
        self.steer_prev = torch.zeros(num_envs, device=device)

    def push(self, throttle: torch.Tensor, steer: torch.Tensor) -> None:
        self.throttle_prev, self.throttle = self.throttle, throttle
        self.steer_prev, self.steer = self.steer, steer

    def reset(self, env_ids: torch.Tensor) -> None:
        for buf in (self.throttle, self.steer, self.throttle_prev, self.steer_prev):
            buf[env_ids] = 0.0


class LapView:
    """What a term is allowed to see of the lap timer."""

    def __init__(self, timer):
        self._timer = timer

    @property
    def count(self) -> torch.Tensor:
        """Valid laps completed in this stint, ``[N]``. Cleared on respawn."""
        return self._timer.lap_count

    @property
    def last_time_s(self) -> torch.Tensor:
        """Time of the last completed lap, seconds, ``[N]``. Zero if none yet."""
        return self._timer.last_lap_time_s

    @property
    def best_time_s(self) -> torch.Tensor:
        """This car's best lap in the run, seconds, ``[N]``. Zero if none yet.

        Survives respawns, unlike :attr:`count` — so it is a fact about the run
        rather than about the current stint, and is for reporting rather than
        for rewarding.
        """
        return self._timer.best_lap_time_s_per_env

    @property
    def run_best_time_s(self) -> float:
        """The fastest lap by any car so far in this run. Zero if none yet."""
        return float(self._timer.best_lap_time_s)

    @property
    def just_finished(self) -> torch.Tensor:
        """True on the step a valid lap completed, ``[N]``."""
        return self._timer.just_finished
