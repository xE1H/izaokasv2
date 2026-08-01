"""Tests for CarState — everything the SDK tells a team about the car.

Runs against the fake articulation in ``conftest.py``, so no Isaac Sim.

    .venv/bin/python -m pytest tests/test_car_state.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from lituanicax_sdk import dynamics, rules
from lituanicax_sdk.state import ControlHistory
from lituanicax_sdk.vehicle import TIMING, VEHICLE

RADIUS = 5.0


# ── The locked constants a team scales by ─────────────────────────────────


def test_locked_car_parameters_are_readable(robot, make_state):
    """Teams cannot change the car, but must be able to read all of it."""
    car = make_state(robot)
    assert car.max_speed_m_s == VEHICLE.motor_no_load_speed_m_s
    assert car.wheel_radius_m == VEHICLE.wheel_radius_m
    assert car.max_steer_rad == VEHICLE.max_steer_rad
    assert car.mass_kg == VEHICLE.mass_kg
    assert car.drive_torque_nm == VEHICLE.drive_torque_nm
    assert car.max_wheel_omega == pytest.approx(
        VEHICLE.motor_no_load_speed_m_s / VEHICLE.wheel_radius_m
    )
    assert car.step_dt == pytest.approx(1 / 30)
    # The whole spec is reachable, for anything not surfaced directly.
    assert car.vehicle.brake_torque_gain_nm == VEHICLE.brake_torque_gain_nm


def test_state_is_a_view_not_a_handle(robot, make_state):
    """A team gets numbers, not the articulation they came from."""
    car = make_state(robot)
    assert not hasattr(car, "robot")
    pos = car.pos_w
    pos[:] = 99.0
    assert float(robot.data.root_pos_w[0, 0]) == 0.0


# ── Pose and motion ───────────────────────────────────────────────────────


def test_pose_decomposition(robot, make_state):
    robot.set_yaw(math.pi / 3)
    car = make_state(robot)
    assert float(car.yaw) == pytest.approx(math.pi / 3, abs=1e-5)
    assert float(car.roll) == pytest.approx(0.0, abs=1e-5)
    assert float(car.up_axis) == pytest.approx(1.0, abs=1e-5)

    robot.set_roll(math.radians(30))
    assert math.degrees(float(make_state(robot).roll)) == pytest.approx(30.0, abs=1e-3)


def test_upside_down_reads_as_flipped(robot, make_state):
    robot.set_roll(math.pi)
    car = make_state(robot)
    assert float(car.up_axis) == pytest.approx(-1.0, abs=1e-5)
    assert bool(rules.flipped(car))


def test_body_frame_velocities(robot, make_state):
    robot.data.root_lin_vel_b[:, 0] = 4.0
    robot.data.root_lin_vel_b[:, 1] = -1.5
    robot.data.root_ang_vel_b[:, 2] = 2.0
    car = make_state(robot)
    assert float(car.speed_forward) == pytest.approx(4.0)
    assert float(car.speed_lateral) == pytest.approx(-1.5)
    assert float(car.yaw_rate) == pytest.approx(2.0)


# ── Drivetrain ────────────────────────────────────────────────────────────


def test_wheel_speed_and_slip(robot, make_state):
    speed = 3.0
    robot.data.joint_vel[:, :4] = speed / VEHICLE.wheel_radius_m
    robot.data.root_lin_vel_b[:, 0] = speed
    car = make_state(robot)
    assert float(car.wheel_speed) == pytest.approx(speed, rel=1e-4)
    assert float(car.slip) == pytest.approx(0.0, abs=1e-4)

    robot.data.joint_vel[:, :4] = 2 * speed / VEHICLE.wheel_radius_m
    assert float(make_state(robot).slip) == pytest.approx(1.0, rel=1e-3)


def test_per_wheel_data_is_exposed(robot, make_state):
    """A team needs to see which end is losing grip, not just an average."""
    robot.data.root_lin_vel_b[:, 0] = 2.0
    # Rear wheels spinning, fronts rolling with the car.
    robot.data.joint_vel[:, 0:2] = 4.0 / VEHICLE.wheel_radius_m
    robot.data.joint_vel[:, 2:4] = 2.0 / VEHICLE.wheel_radius_m
    car = make_state(robot)

    assert car.wheel_speeds.shape == (1, 4)
    assert float(car.wheel_speeds[0, 0]) == pytest.approx(4.0, rel=1e-3)
    assert float(car.wheel_speeds[0, 2]) == pytest.approx(2.0, rel=1e-3)

    slips = car.wheel_slips
    assert float(slips[0, 0]) > 0.5, "spinning rear wheel"
    assert float(slips[0, 2]) == pytest.approx(0.0, abs=1e-3), "gripping front wheel"
    assert car.wheelbase_names[0] == "back_left_wheel_throttle"


def test_raw_joints_are_available(robot, make_state):
    robot.data.joint_pos[:, 6] = 0.01
    car = make_state(robot)
    assert car.joint_pos.shape == (1, 10)
    assert car.joint_vel.shape == (1, 10)
    assert len(car.joint_names) == 10
    assert float(car.suspension_travel[0, 0]) == pytest.approx(0.01)


def test_steer_angle_undoes_the_tan_encoding(robot, make_state):
    robot.data.joint_pos[:, 4:6] = math.tan(0.3)
    assert float(make_state(robot).steer_angle) == pytest.approx(0.3, abs=1e-5)


# ── Track-relative ────────────────────────────────────────────────────────


def test_cross_track_error(robot, make_state):
    robot.data.root_pos_w[:, 0] = RADIUS + 0.2
    assert float(make_state(robot).cross_track_error) == pytest.approx(0.2, abs=1e-3)


def test_signed_cross_track_error_tells_you_which_way(robot, make_state):
    robot.data.root_pos_w[:, 0] = RADIUS + 0.2
    outside = float(make_state(robot).signed_cross_track_error)
    robot.data.root_pos_w[:, 0] = RADIUS - 0.2
    inside = float(make_state(robot).signed_cross_track_error)
    assert outside * inside < 0
    assert abs(outside) == pytest.approx(0.2, abs=1e-3)


def test_direction_is_inferred_from_velocity(robot, make_state):
    robot.data.root_pos_w[:, 0] = RADIUS
    robot.data.root_lin_vel_w[:, 1] = 2.0
    assert bool(make_state(robot).going_forward)
    robot.data.root_lin_vel_w[:, 1] = -2.0
    assert not bool(make_state(robot).going_forward)


def test_a_stationary_car_takes_its_direction_from_where_it_points(robot, make_state):
    """Every episode starts from rest, and a car at rest has no velocity to read.

    Reading the velocity alone called every stationary car *reversed* — a zero
    dot product is not greater than zero — so the whole track-relative half of
    the observation was mirrored for the first few steps of every episode, which
    is exactly when the policy is choosing which way to steer. A car spawned on
    the centerline facing along the track then drove itself into the wall.
    """
    robot.data.root_pos_w[:, 0] = RADIUS
    robot.data.root_lin_vel_w[:] = 0.0

    robot.set_yaw(math.pi / 2)  # along the stored centerline order
    assert bool(make_state(robot).going_forward)

    robot.set_yaw(-math.pi / 2)  # against it
    assert not bool(make_state(robot).going_forward)


def test_a_moving_car_still_takes_its_direction_from_its_velocity(robot, make_state):
    """Sliding backwards while pointing forwards reads as going backwards."""
    robot.data.root_pos_w[:, 0] = RADIUS
    robot.set_yaw(math.pi / 2)
    robot.data.root_lin_vel_w[:, 1] = -2.0
    assert not bool(make_state(robot).going_forward)


def test_heading_error_is_zero_when_aligned_either_way(robot, make_state):
    """Driving the track backwards must not read as a maximal heading error."""
    robot.data.root_pos_w[:, 0] = RADIUS
    robot.data.root_lin_vel_w[:, 1] = 2.0

    robot.set_yaw(math.pi / 2)
    assert abs(float(make_state(robot).heading_error)) == pytest.approx(0.0, abs=1e-4)
    robot.set_yaw(-math.pi / 2)
    assert abs(float(make_state(robot).heading_error)) == pytest.approx(0.0, abs=1e-4)
    robot.set_yaw(0.0)
    assert abs(float(make_state(robot).heading_error)) == pytest.approx(
        math.pi / 2, abs=1e-3
    )


def test_lookahead_mirrors_with_direction(robot, make_state):
    """The same place, driven both ways, must both report the track ahead."""
    robot.data.root_pos_w[:, 0] = RADIUS

    robot.set_yaw(math.pi / 2)
    robot.data.root_lin_vel_w[:, 1] = 2.0
    forward = make_state(robot).lookahead([40])[0, 0]

    robot.set_yaw(-math.pi / 2)
    robot.data.root_lin_vel_w[:, 1] = -2.0
    reverse = make_state(robot).lookahead([40])[0, 0]

    assert float(forward[0]) > 0.1 and float(reverse[0]) > 0.1
    assert float(forward[0]) == pytest.approx(float(reverse[0]), rel=1e-3)


def test_lookahead_shape_and_curvature(robot, make_state):
    robot.data.root_pos_w[:, 0] = RADIUS
    robot.data.root_lin_vel_w[:, 1] = 2.0
    ahead = make_state(robot).lookahead([10, 20, 40])
    assert ahead.shape == (1, 3, 3)
    # Curvature of this circle is 1/5 everywhere.
    assert float(ahead[0, 0, 2]) == pytest.approx(0.2, rel=0.05)


def test_walls_must_be_loaded_before_wall_distance_is_read(robot, make_state):
    with pytest.raises(RuntimeError, match="walls have not been loaded"):
        _ = make_state(robot).dist_to_wall


# ── Commands and episode ──────────────────────────────────────────────────


def test_commands_track_the_previous_step(robot, make_state):
    commands = ControlHistory(1, "cpu")
    commands.push(torch.tensor([0.5]), torch.tensor([0.1]))
    commands.push(torch.tensor([-0.5]), torch.tensor([-0.1]))
    car = make_state(robot, commands=commands)
    assert float(car.throttle_cmd) == pytest.approx(-0.5)
    assert float(car.throttle_cmd_prev) == pytest.approx(0.5)
    assert float(car.steer_cmd) == pytest.approx(-0.1)
    assert float(car.steer_cmd_prev) == pytest.approx(0.1)


def test_episode_step_and_time(robot, make_state):
    car = make_state(robot, episode_step=120)
    assert int(car.episode_step) == 120
    assert float(car.episode_time_s) == pytest.approx(120 * TIMING.step_dt)


# ── The locked drivetrain ─────────────────────────────────────────────────


def test_full_throttle_from_rest_gives_full_torque():
    torque = dynamics.wheel_torque(torch.tensor([1.0]), torch.zeros(1, 4))
    assert torque.shape == (1, 4)
    assert float(torque[0, 0]) == pytest.approx(VEHICLE.drive_torque_nm)


def test_torque_falls_to_zero_at_top_speed():
    at_top = dynamics.wheel_torque(
        torch.tensor([1.0]), torch.full((1, 4), VEHICLE.max_wheel_omega)
    )
    assert float(at_top[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_braking_opposes_rotation_and_fades_at_a_standstill():
    moving = dynamics.wheel_torque(torch.tensor([-1.0]), torch.full((1, 4), 50.0))
    assert float(moving[0, 0]) < 0.0
    stopped = dynamics.wheel_torque(torch.tensor([-1.0]), torch.zeros(1, 4))
    assert float(stopped[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_zero_throttle_is_zero_torque():
    torque = dynamics.wheel_torque(torch.tensor([0.0]), torch.zeros(1, 4))
    assert float(torque[0, 0]) == 0.0


def test_steering_maps_to_the_locked_limit():
    full = dynamics.steer_target(torch.tensor([1.0]))
    assert float(full) == pytest.approx(math.tan(VEHICLE.max_steer_rad))
    assert float(dynamics.steer_target(torch.tensor([0.0]))) == pytest.approx(0.0)


def test_actions_are_clamped_and_truncated_to_two():
    out = dynamics.clamp_actions(torch.tensor([[5.0, -5.0, 1.0, 1.0]]))
    assert out.shape == (1, 2)
    assert float(out[0, 0]) == 1.0 and float(out[0, 1]) == -1.0
