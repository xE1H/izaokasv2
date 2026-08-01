"""A fake articulation, so the SDK can be tested without starting Isaac Sim.

``CarState`` only ever reads ``robot.data.<buffer>``, which makes it easy to
stand in for: hand it the buffers directly and every observation, reward and
termination term can be exercised on the CPU in milliseconds.
"""

from __future__ import annotations

import csv
import math

import pytest
import torch

from lituanicax_sdk.state import ControlHistory
from lituanicax_sdk.track import Track, TrackCfg

RADIUS = 5.0
NUM_POINTS = 720


class FakeData:
    """The subset of Isaac Lab's ``ArticulationData`` that ``CarState`` reads."""

    def __init__(self, num_envs, device):
        self.root_pos_w = torch.zeros(num_envs, 3, device=device)
        self.root_quat_w = torch.zeros(num_envs, 4, device=device)
        self.root_quat_w[:, 0] = 1.0  # identity rotation
        self.root_lin_vel_b = torch.zeros(num_envs, 3, device=device)
        self.root_lin_vel_w = torch.zeros(num_envs, 3, device=device)
        self.root_ang_vel_b = torch.zeros(num_envs, 3, device=device)
        self.joint_vel = torch.zeros(num_envs, 10, device=device)
        self.joint_pos = torch.zeros(num_envs, 10, device=device)
        self.joint_names = [
            "back_left_wheel_throttle", "back_right_wheel_throttle",
            "front_left_wheel_throttle", "front_right_wheel_throttle",
            "front_left_wheel_steer", "front_right_wheel_steer",
            "front_left_wheel_suspension", "front_right_wheel_suspension",
            "back_left_wheel_suspension", "back_right_wheel_suspension",
        ]


class FakeRobot:
    """Stands in for an Isaac Lab ``Articulation``."""

    def __init__(self, num_envs=1, device="cpu"):
        self.data = FakeData(num_envs, device)

    def set_yaw(self, yaw: float | torch.Tensor):
        yaw = torch.as_tensor(yaw, dtype=torch.float32).reshape(-1)
        self.data.root_quat_w[:, 0] = torch.cos(yaw / 2)
        self.data.root_quat_w[:, 3] = torch.sin(yaw / 2)
        self.data.root_quat_w[:, 1] = 0.0
        self.data.root_quat_w[:, 2] = 0.0
        return self

    def set_roll(self, roll: float):
        r = torch.tensor(roll / 2)
        self.data.root_quat_w[:, 0] = torch.cos(r)
        self.data.root_quat_w[:, 1] = torch.sin(r)
        self.data.root_quat_w[:, 2] = 0.0
        self.data.root_quat_w[:, 3] = 0.0
        return self


@pytest.fixture
def circle_track(tmp_path):
    """A 5 m radius circle with the start/finish line at (5, 0)."""
    path = tmp_path / "circle.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for i in range(NUM_POINTS):
            theta = 2.0 * math.pi * i / NUM_POINTS
            writer.writerow([RADIUS * math.cos(theta), RADIUS * math.sin(theta), 0.0])

    return Track(
        TrackCfg(
            name="circle",
            walls_usd="unused.usd",
            centerline_csv=str(path),
            centerline_scale=(1.0, 1.0),
            start_finish_index=0,
        ),
        device="cpu",
    )


@pytest.fixture
def make_state(circle_track):
    """Build a ``CarState`` over the fake robot."""
    from lituanicax_sdk.state import CarState
    from lituanicax_sdk.vehicle import TIMING, VEHICLE

    def build(robot: FakeRobot, episode_step=0, commands=None, track=None):
        num_envs = robot.data.root_pos_w.shape[0]
        return CarState(
            robot=robot,
            track=track or circle_track,
            vehicle=VEHICLE,
            step_dt=TIMING.step_dt,
            throttle_ids=torch.tensor([0, 1, 2, 3]),
            steer_ids=torch.tensor([4, 5]),
            suspension_ids=torch.tensor([6, 7, 8, 9]),
            episode_step=torch.full((num_envs,), episode_step, dtype=torch.long),
            commands=commands or ControlHistory(num_envs, "cpu"),
            applied_wheel_torque=torch.zeros(num_envs, 4),
            wall_touched=torch.zeros(num_envs, dtype=torch.bool),
        )

    return build


@pytest.fixture
def robot():
    return FakeRobot(num_envs=1)
