"""The car: a MuSHR nano v2, identical for every team.

Everything in this file is locked.  The whole point of the competition is that
the same car is driven by different policies, so the mass, the motor, the
brakes, the steering limits, the tyres and the simulation rate are fixed and
the same for everyone.

``VEHICLE`` and ``TIMING`` are plain dataclasses with no Isaac Lab dependency,
so they can be read (and tested) without starting the simulator.  The Isaac Lab
:class:`ArticulationCfg` is built on demand by :func:`build_articulation_cfg`,
which does need the simulator's config types.

Teams read these numbers freely — ``CarState`` exposes the ones you need for
observations, so you never have to hardcode 6.7 or 0.037 in your own code.
"""

from __future__ import annotations

from pathlib import Path

from ._locked import LockedCfg, locked_dataclass

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CAR_USD = str(ASSETS_DIR / "mushr_nano_v2.usd")


# ═══════════════════════════════════════════════════════════════════════════
#  Locked: how fast the simulation runs
# ═══════════════════════════════════════════════════════════════════════════


@locked_dataclass
class TimingCfg(LockedCfg):
    """Simulation and control rates.

    Two numbers define the whole clock, and they are the two that mean
    something: how often the policy acts, and how many physics substeps it gets
    between actions.  Everything else — ``physics_dt``, ``step_dt``,
    ``physics_hz``, ``render_interval`` — is derived, so the rate can be
    changed in one place without any pair of values drifting out of step.

    Locked because lap times are counted in policy steps.  A team running the
    physics at a different rate would get a different car *and* a differently
    measured clock.
    """

    #: How often the policy acts, and so the rate the lap clock ticks at.
    policy_hz: float = 30.0
    #: Physics substeps per policy step.  Raising this buys contact and
    #: friction accuracy at a proportional cost in simulation time; it does not
    #: change how often the policy acts.
    decimation: int = 4

    @property
    def physics_hz(self) -> float:
        """Physics solver rate, ``policy_hz × decimation``."""
        return self.policy_hz * self.decimation

    @property
    def physics_dt(self) -> float:
        """Seconds of simulated time per physics substep."""
        return 1.0 / self.physics_hz

    @property
    def step_dt(self) -> float:
        """Seconds of simulated time per policy step."""
        return 1.0 / self.policy_hz

    @property
    def render_interval(self) -> int:
        """Physics substeps between rendered frames — one frame per policy step."""
        return self.decimation

    def describe(self) -> str:
        return (
            f"{self.policy_hz:.0f} Hz policy "
            f"({self.physics_hz:.0f} Hz physics / {self.decimation})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Locked: the car
# ═══════════════════════════════════════════════════════════════════════════


@locked_dataclass
class VehicleCfg(LockedCfg):
    """Physical parameters of the MuSHR nano v2.

    ``motor_no_load_speed_m_s`` is the speed at which the motor stops
    producing torque — the car's top speed on the flat.  It is deliberately a
    *motor* parameter here and nothing else: the old code used the same number
    to normalise observations and rewards, which meant a team could not rescale
    its inputs without also changing the drivetrain.  Scale your observations
    by ``CarState.max_speed_m_s`` instead, which reads this value.
    """

    name: str = "mushr_nano_v2"
    usd_path: str = CAR_USD
    usd_scale: float = 0.755

    # ── Chassis ────────────────────────────────────────────────────────────
    mass_kg: float = 1.27
    wheel_radius_m: float = 0.037  # 74 mm diameter wheels

    # ── Motor: a DC torque-speed curve, full torque at rest, none at top speed
    motor_no_load_speed_m_s: float = 6.7
    drive_torque_nm: float = 0.068

    # ── Brakes: constant plus throttle-proportional, fading out near standstill
    brake_torque_base_nm: float = 0.028
    brake_torque_gain_nm: float = 0.310
    brake_throttle_scale: float = 0.4
    brake_release_omega_rad_s: float = 2.0

    # ── Steering: an Ackermann rack driven in tan(angle) space ─────────────
    max_steer_rad: float = 0.488  # ±28°
    steer_stiffness: float = 100.0
    steer_damping: float = 10.0
    steer_velocity_limit: float = 6.51
    steer_effort_limit: float = 3.2

    # ── Suspension: a stiff spring that only absorbs ground unevenness ─────
    suspension_stiffness: float = 1e6

    # ── Contact ────────────────────────────────────────────────────────────
    ground_static_friction: float = 2.0
    ground_dynamic_friction: float = 1.6
    ground_restitution: float = 0.05
    tyre_static_friction: float = 1.0
    tyre_dynamic_friction: float = 1.0
    tyre_restitution: float = 0.05

    # ── Joint and link names inside the USD ────────────────────────────────
    throttle_joints: tuple[str, ...] = (
        "back_left_wheel_throttle",
        "back_right_wheel_throttle",
        "front_left_wheel_throttle",
        "front_right_wheel_throttle",
    )
    steer_joints: tuple[str, ...] = (
        "front_left_wheel_steer",
        "front_right_wheel_steer",
    )
    suspension_joints: tuple[str, ...] = (
        "front_left_wheel_suspension",
        "front_right_wheel_suspension",
        "back_left_wheel_suspension",
        "back_right_wheel_suspension",
    )
    wheel_links: tuple[str, ...] = (
        "front_left_wheel_link",
        "front_right_wheel_link",
        "back_left_wheel_link",
        "back_right_wheel_link",
    )
    articulation_subpath: str = "mushr_nano"

    # ── The policy's control interface — 2 numbers, and only ever 2 ────────
    action_dim: int = 2
    action_names: tuple[str, ...] = ("throttle", "steering")

    @property
    def max_wheel_omega(self) -> float:
        """No-load wheel speed, rad/s."""
        return self.motor_no_load_speed_m_s / self.wheel_radius_m

    def all_joints(self) -> tuple[str, ...]:
        return self.throttle_joints + self.steer_joints + self.suspension_joints


# The singletons.  These are the competition's vehicle and clock.
VEHICLE = VehicleCfg()
TIMING = TimingCfg()


# ═══════════════════════════════════════════════════════════════════════════
#  Isaac Lab plumbing
# ═══════════════════════════════════════════════════════════════════════════


def build_articulation_cfg(prim_path: str = "/World/envs/env_.*/Robot"):
    """Build the Isaac Lab :class:`ArticulationCfg` for the locked vehicle.

    Imported lazily so that the rest of this module stays usable — and
    testable — without Isaac Sim on the path.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    v = VEHICLE
    scale = (v.usd_scale, v.usd_scale, v.usd_scale)

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=v.usd_path,
            scale=scale,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10000.0,
                max_angular_velocity=100000.0,
                max_depenetration_velocity=1.0,
                max_contact_impulse=5.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=dict.fromkeys(v.all_joints(), 0.0),
        ),
        actuators={
            # The steering servo holds a commanded angle, so it needs a stiff
            # position controller.
            "steering": ImplicitActuatorCfg(
                joint_names_expr=list(v.steer_joints),
                stiffness=v.steer_stiffness,
                damping=v.steer_damping,
                velocity_limit_sim=v.steer_velocity_limit,
                effort_limit_sim=v.steer_effort_limit,
            ),
            # The wheels are driven with raw torque, so their actuators must
            # not fight the drive model with stiffness or damping.
            "throttle": ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_throttle"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                effort_limit_sim=5.0,
                velocity_limit_sim=1e9,
            ),
            "suspension": ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_suspension"],
                stiffness=v.suspension_stiffness,
                damping=0.0,
                friction=0.0,
                effort_limit_sim=1e9,
                velocity_limit_sim=1e9,
            ),
        },
    )


def build_sim_cfg():
    """The locked :class:`SimulationCfg`."""
    from isaaclab.sim import SimulationCfg

    return SimulationCfg(dt=TIMING.physics_dt, render_interval=TIMING.render_interval)


def build_ground_material_cfg():
    import isaaclab.sim as sim_utils

    return sim_utils.RigidBodyMaterialCfg(
        static_friction=VEHICLE.ground_static_friction,
        dynamic_friction=VEHICLE.ground_dynamic_friction,
        restitution=VEHICLE.ground_restitution,
        friction_combine_mode="multiply",
        restitution_combine_mode="min",
    )


__all__ = [
    "VEHICLE",
    "TIMING",
    "VehicleCfg",
    "TimingCfg",
    "build_articulation_cfg",
    "build_sim_cfg",
    "build_ground_material_cfg",
    "CAR_USD",
    "ASSETS_DIR",
]
