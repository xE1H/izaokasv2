"""The maths behind :mod:`tools.probe`, with no Isaac Sim dependency.

Split out from the probe itself so it can be tested. ``tools/probe.py`` has to
launch the simulator before it can import anything from Isaac Lab, which makes it
unimportable on a machine without Isaac Sim — and therefore makes everything
defined inside it untestable. The measurement *logic* is the part most likely to
be wrong (a sign, a window, an off-by-one in a step count), so it lives here and
``tests/test_probe.py`` drives it against synthetic traces with known answers.

The probe's entry point keeps only what genuinely needs a running simulator:
building the environment, stepping it, and printing the verdict.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lituanicax_sdk.track import TrackCfg
from lituanicax_sdk.tracks import OFFICIAL
from lituanicax_sdk.vehicle import TIMING, VEHICLE

from .measured import Measured

#: How much larger than the official track the probe's track is. 40x turns a
#: 0.70 m corridor into 28 m and a 50 m loop into 2 km — room for a 50 m straight
#: and for cornering circles of several metres' radius.
PROBE_SCALE = 40.0

#: Steering commands swept for the cornering limit, as a fraction of full lock.
STEER_LEVELS = (0.15, 0.25, 0.4, 0.6, 0.8, 1.0)

#: Steps spent accelerating before the braking phase begins.
ACCEL_STEPS = 150

#: Step the steering command at this step, for the lag measurement.
STEER_STEP_AT = 60

#: Total steps the probe runs for.
TOTAL_STEPS = 700


def probe_track(scale: float = PROBE_SCALE) -> TrackCfg:
    """The official track's walls at ``scale`` times their size.

    Every probe manoeuvre needs room the official 0.70 m corridor does not have.
    There is no "no track" option — ``RaceEnv`` always builds walls and
    :meth:`~lituanicax_sdk.track.Track.load_walls` raises without them — so the
    same asset is reused, enlarged.

    The centerline is scaled by the same factor, which is the invariant
    :class:`~lituanicax_sdk.track.TrackCfg` warns about: whatever scales the mesh
    has to scale the line, or the line stops describing the track. The gate window
    and the corner threshold are carried along too, the latter inversely because
    curvature has units of 1/length.
    """
    return TrackCfg(
        name="probe",
        walls_usd=OFFICIAL.walls_usd,
        surface_usd=OFFICIAL.surface_usd,
        centerline_csv=OFFICIAL.centerline_csv,
        mesh_scale=OFFICIAL.mesh_scale * scale,
        centerline_scale=(
            OFFICIAL.centerline_scale[0] * scale,
            OFFICIAL.centerline_scale[1] * scale,
        ),
        start_finish_index=OFFICIAL.start_finish_index,
        lap_gate_window_m=OFFICIAL.lap_gate_window_m * scale,
        corner_curvature_threshold=OFFICIAL.corner_curvature_threshold / scale,
    )


def straightest_point(geometry, run_m: float) -> tuple[float, float, float]:
    """``(x, y, yaw)`` at the start of the flattest ``run_m`` of track.

    The acceleration and braking runs want to go straight, and the origin is
    wherever the centerline's first point happens to be — which on the official
    track is not a straight. Picking the flattest stretch means the car is not
    fighting the track's curvature while its longitudinal limits are measured.
    """
    window = max(
        1, min(int(round(run_m / geometry.spacing_m)), geometry.num_samples - 1)
    )
    curvature = geometry.kappa.abs()
    padded = torch.cat([curvature, curvature[:window]])
    cumulative = torch.cat(
        [torch.zeros(1, dtype=padded.dtype), torch.cumsum(padded, dim=0)]
    )
    rolling = (cumulative[window:] - cumulative[:-window])[: geometry.num_samples]
    start = int(rolling.argmin())

    position = geometry.pos[start]
    heading = math.atan2(
        float(geometry.tangent[start][1]), float(geometry.tangent[start][0])
    )
    return float(position[0]), float(position[1]), heading


class Recorder:
    """Collects the traces the analysis needs, one row per step."""

    FIELDS = (
        "speed_forward",
        "speed_lateral",
        "yaw_rate",
        "up_axis",
        "roll",
        "steer_angle",
        "wheel_speed",
    )

    def __init__(self):
        self.frames: dict[str, list[torch.Tensor]] = {f: [] for f in self.FIELDS}
        self.position: list[torch.Tensor] = []

    def add(self, car) -> None:
        for field in self.FIELDS:
            self.frames[field].append(getattr(car, field).detach().cpu().clone())
        self.position.append(car.pos_xy.detach().cpu().clone())

    def array(self, field: str) -> np.ndarray:
        """``[steps, num_envs]``."""
        return torch.stack(self.frames[field]).numpy()

    def positions(self) -> np.ndarray:
        """``[steps, num_envs, 2]``."""
        return torch.stack(self.position).numpy()


def layout() -> tuple[range, int, int]:
    """``(lateral_envs, lag_env, num_envs)`` — which environment measures what.

    All in one simulation, because Isaac Sim takes minutes to start and the
    alternative is paying that per measurement.
    """
    lateral = range(2, 2 + len(STEER_LEVELS))
    lag = 2 + len(STEER_LEVELS)
    return lateral, lag, lag + 1


def build_script(num_envs: int, lateral_envs: range, lag_env: int):
    """Per-environment command programs, as a function of step."""

    def script(step: int) -> torch.Tensor:
        actions = torch.zeros(num_envs, 2)
        # env 0 — full lock at a gentle speed: the kinematic turning circle.
        actions[0, 0] = 0.22
        actions[0, 1] = 1.0
        # env 1 — accelerate flat out, then stand on the brakes.
        actions[1, 0] = 1.0 if step < ACCEL_STEPS else -1.0
        actions[1, 1] = 0.0
        # envs 2.. — hold a steering angle at full throttle until something gives.
        for offset, index in enumerate(lateral_envs):
            actions[index, 0] = 1.0
            actions[index, 1] = STEER_LEVELS[offset]
        # last env — a steering step, for the actuation lag.
        actions[lag_env, 0] = 0.3
        actions[lag_env, 1] = 0.0 if step < STEER_STEP_AT else 1.0
        return actions

    return script


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════════════════


def fit_circle(points: np.ndarray) -> tuple[float, float]:
    """Least-squares circle through ``[k, 2]`` points; returns ``(radius, rms)``.

    Algebraic fit: ``x² + y² = 2ax + 2by + c`` is linear in ``(a, b, c)``, so this
    is one least-squares solve rather than an iteration that can fail to converge.
    """
    x, y = points[:, 0], points[:, 1]
    design = np.stack([2.0 * x, 2.0 * y, np.ones_like(x)], axis=1)
    solution, *_ = np.linalg.lstsq(design, x**2 + y**2, rcond=None)
    centre = solution[:2]
    radius = math.sqrt(max(solution[2] + float(centre @ centre), 0.0))
    residual = np.linalg.norm(points - centre, axis=1) - radius
    return radius, float(np.sqrt(np.mean(residual**2)))


def measure_wheelbase(robot_data) -> tuple[float, str]:
    """Front-to-rear axle distance from the articulation's own body positions.

    Takes ``robot.data`` rather than the environment, so a test can hand it a stub.
    """
    names = list(robot_data.body_names)
    positions = np.asarray(
        robot_data.body_pos_w[0].detach().cpu().numpy()
        if hasattr(robot_data.body_pos_w, "detach")
        else robot_data.body_pos_w[0]
    )

    def centre(pattern: str):
        picked = [positions[i] for i, name in enumerate(names) if pattern in name]
        if not picked:
            return None
        return np.mean(np.stack(picked), axis=0)

    front, rear = centre("front"), centre("back")
    if front is None or rear is None:
        return float("nan"), f"no front/back bodies among {names}"
    return float(np.linalg.norm((front - rear)[:2])), "from body positions"


def measure_accel_curve(speed: np.ndarray, dt: float) -> list[tuple[float, float]]:
    """``[(speed, acceleration), ...]`` sampled from the acceleration run.

    A curve rather than a number because the SDK's motor follows a torque-speed
    curve (``lituanicax_sdk/dynamics.py:44``): full torque from rest, none at top
    speed. A single figure is a summary of this, not a constant.
    """
    if len(speed) < 3:
        return []
    accel = np.gradient(speed, dt)
    samples = []
    for bucket in np.arange(0.0, VEHICLE.motor_no_load_speed_m_s, 0.5):
        mask = (speed >= bucket) & (speed < bucket + 0.5)
        if mask.sum() >= 2:
            samples.append((float(bucket + 0.25), float(np.median(accel[mask]))))
    return samples


def analyse(
    recorder: Recorder,
    robot_data,
    lateral_envs: range,
    lag_env: int,
    *,
    dt: float = TIMING.step_dt,
) -> tuple[Measured, dict]:
    """Turn the recorded traces into a :class:`~tools.measured.Measured`."""
    speed = recorder.array("speed_forward")
    yaw_rate = recorder.array("yaw_rate")
    up_axis = recorder.array("up_axis")
    steer_angle = recorder.array("steer_angle")
    positions = recorder.positions()
    notes: dict = {}
    columns = list(lateral_envs)

    # ── 1. Wheelbase ─────────────────────────────────────────────────────
    wheelbase, how = measure_wheelbase(robot_data)
    notes["wheelbase_source"] = how

    # Confirm against the full-lock circle. Second half of the trace only, by
    # which time the car has settled into a steady turn.
    circle_path = positions[len(positions) // 2 :, 0, :]
    radius, rms = fit_circle(circle_path)
    implied = radius * math.tan(VEHICLE.max_steer_rad)
    notes.update(
        {
            "full_lock_radius_m": radius,
            "full_lock_fit_rms_m": rms,
            "wheelbase_from_circle_m": implied,
        }
    )
    if math.isnan(wheelbase):
        wheelbase = implied
        notes["wheelbase_source"] = "from the full-lock circle (no body names matched)"
    elif wheelbase > 0 and abs(implied - wheelbase) > 0.25 * wheelbase:
        notes["wheelbase_disagreement"] = (
            f"geometry says {wheelbase:.3f} m, the full-lock circle implies "
            f"{implied:.3f} m. The circle includes tyre slip, so a larger implied "
            "value is expected — a big gap means the car is sliding, and R_min "
            "from the geometric figure is then optimistic."
        )

    # ── 3/4. Longitudinal ────────────────────────────────────────────────
    run = speed[:ACCEL_STEPS, 1]
    curve = measure_accel_curve(run, dt)
    target = 0.8 * VEHICLE.motor_no_load_speed_m_s
    reached = (
        int(np.argmax(run >= target)) if bool((run >= target).any()) else len(run) - 1
    )
    a_accel = (
        float(run[reached] / max(reached * dt, dt)) if reached > 0 else float("nan")
    )
    notes.update(
        {
            "accel_curve": curve,
            "steps_to_80pc_top_speed": reached,
            "top_speed_reached_m_s": float(run.max()),
        }
    )

    brake = speed[ACCEL_STEPS:, 1]
    if len(brake) > 1:
        stopped = (
            int(np.argmax(brake <= 0.2))
            if bool((brake <= 0.2).any())
            else len(brake) - 1
        )
        a_brake = (
            float((brake[0] - brake[stopped]) / max(stopped * dt, dt))
            if stopped > 0
            else float("nan")
        )
    else:
        stopped, a_brake = 0, float("nan")
    notes["steps_to_stop"] = stopped

    # ── 5/6. Lateral, and rollover ───────────────────────────────────────
    # Lateral acceleration in a steady turn is v * yaw_rate.
    lateral = np.abs(speed[:, columns] * yaw_rate[:, columns])
    tipped = up_axis[:, columns] < 0.3

    per_angle = []
    for column, level in enumerate(STEER_LEVELS):
        trace = lateral[:, column]
        tip = tipped[:, column]
        if bool(tip.any()):
            first = int(np.argmax(tip))
            # What it was pulling just before it went over.
            per_angle.append(
                {
                    "steer": level,
                    "a_lat": float(np.max(trace[: max(first, 1)])),
                    "tipped_at_step": first,
                }
            )
        else:
            # Steady state: the last third, once transients have gone.
            settled = trace[len(trace) * 2 // 3 :]
            per_angle.append({"steer": level, "a_lat": float(np.median(settled))})

    notes["lateral_by_steer"] = per_angle

    # The grip limit is what the car can *sustain*. A run that ended on its roof
    # was not sustaining anything, so its peak is the rollover threshold and not
    # evidence about grip — counting it as both would make the two equal and hide
    # the fact that the car is rollover-limited.
    sustained = [e["a_lat"] for e in per_angle if "tipped_at_step" not in e]
    at_rollover = [e["a_lat"] for e in per_angle if "tipped_at_step" in e]
    a_lat = max(sustained) if sustained else max(at_rollover)
    rollover = min(at_rollover) if at_rollover else None
    notes["tipped_at_steer"] = [e["steer"] for e in per_angle if "tipped_at_step" in e]

    everything = [e["a_lat"] for e in per_angle]
    spread = max(everything) - min(everything)
    notes["a_lat_varies_with_steer"] = bool(spread > 1.0)
    notes["a_lat_spread_m_s2"] = float(spread)

    # ── 7. Steering lag ──────────────────────────────────────────────────
    after = steer_angle[STEER_STEP_AT:, lag_env]
    final = float(np.median(after[-20:])) if len(after) > 20 else float(after[-1])
    lag = (
        float(np.argmax(np.abs(after) >= 0.9 * abs(final)))
        if abs(final) > 1e-3
        else float("nan")
    )
    notes["steer_final_angle_rad"] = final

    car = Measured(
        wheelbase_m=wheelbase,
        a_accel_m_s2=a_accel,
        a_brake_m_s2=a_brake,
        a_lat_max_m_s2=a_lat,
        rollover_a_lat_m_s2=rollover,
        steer_lag_steps=lag,
        v_max_m_s=float(max(float(run.max()), VEHICLE.motor_no_load_speed_m_s * 0.5)),
        accel_curve=curve,
        measured=True,
    )
    return car, notes
