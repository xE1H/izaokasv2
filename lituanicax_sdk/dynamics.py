"""How the policy's two numbers become torque and steering angle.

Locked.  This is the drivetrain: change it and you are driving a different car,
which is exactly what the competition is designed to prevent.

The policy outputs, both in ``[-1, 1]``:

* **throttle** — positive drives, negative brakes
* **steering** — scaled to ``±max_steer_rad``

Pure tensor maths with no Isaac Lab dependency, so it can be tested on its own.
"""

from __future__ import annotations

import torch

from .vehicle import VEHICLE, VehicleCfg


def wheel_torque(
    throttle: torch.Tensor,
    wheel_omega: torch.Tensor,
    vehicle: VehicleCfg = VEHICLE,
) -> torch.Tensor:
    """Torque for each wheel, from throttle and current wheel speed.

    Args:
        throttle: ``[N]`` in ``[-1, 1]``. Positive drives, negative brakes.
        wheel_omega: ``[N, 4]`` wheel speeds in rad/s.

    Returns:
        ``[N, 4]`` torque in Nm.

    Driving follows a DC motor's torque-speed curve: full torque from
    standstill, tapering to zero at the speed the motor can no longer sustain.
    Braking opposes rotation and fades out below ``brake_release_omega_rad_s``
    so the car does not shudder at a standstill.
    """
    throttle = throttle.unsqueeze(-1)  # [N, 1], broadcasts over the 4 wheels
    omega = wheel_omega.clamp(min=0.0)

    omega_no_load = throttle.clamp(min=1e-3) * vehicle.max_wheel_omega
    drive_falloff = (1.0 - omega / omega_no_load).clamp(0.0, 1.0)
    drive = vehicle.drive_torque_nm * drive_falloff

    brake_level = (-throttle).clamp(min=0.0) * vehicle.brake_throttle_scale
    brake_fade = (omega / vehicle.brake_release_omega_rad_s).clamp(0.0, 1.0)
    brake = -(
        vehicle.brake_torque_base_nm + vehicle.brake_torque_gain_nm * brake_level
    ) * brake_fade

    return torch.where(
        throttle > 0.0,
        drive,
        torch.where(throttle < 0.0, brake, torch.zeros_like(brake)),
    )


def steer_target(
    steering: torch.Tensor, vehicle: VehicleCfg = VEHICLE
) -> torch.Tensor:
    """Steering joint target from the policy's steering command.

    Args:
        steering: ``[N]`` in ``[-1, 1]``.

    Returns:
        ``[N]`` joint target. The steering joints are driven in ``tan(angle)``
        space, which is what makes the linkage behave like an Ackermann rack.
    """
    return torch.tan(steering * vehicle.max_steer_rad)


def clamp_actions(actions: torch.Tensor, vehicle: VehicleCfg = VEHICLE) -> torch.Tensor:
    """Take the two locked action channels and clamp them to ``[-1, 1]``.

    Extra columns are ignored rather than rejected, so a policy trained with a
    wider head still runs — it just cannot steer the car with them.
    """
    return actions[:, : vehicle.action_dim].clamp(-1.0, 1.0)
