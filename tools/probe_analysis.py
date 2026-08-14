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

#: How much larger than the official track the probe's track is. 120x turns a
#: 0.70 m corridor into 84 m and a 50 m loop into 6 km.
#:
#: 40x was not enough and the failure was silent. The steering servo is
#: effort-limited, so a car commanded to 0.15 of full lock actually holds about
#: 0.02 rad at speed — an 11 m circle, which walks straight out of a 28 m
#: corridor and into a wall. The impact flipped the car, ``up_axis`` went below
#: 0.3, and the probe reported a *rollover at 3.2 m/s^2 lateral* — a number that
#: would have put every speed target on the whole track at a third of what the
#: car can do. :func:`impact_step` now catches this whatever the scale, but the
#: cheapest fix is to not be near a wall in the first place: scaling the mesh
#: costs nothing, since it is the same asset either way.
PROBE_SCALE = 120.0

#: A one-step drop in forward speed larger than this is an impact, not braking.
#: The brakes manage about 11 m/s^2, which is 0.38 m/s in a 30 Hz step; 2 m/s is
#: 60 m/s^2 and nothing but a wall does that.
IMPACT_DROP_M_S = 2.0

#: ``up_axis`` below this is a car that is no longer on its wheels.
TIPPED_UP_AXIS = 0.3

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


def impact_step(speed: np.ndarray) -> int | None:
    """First step at which forward speed collapses — i.e. the car hit something.

    Everything the probe measures after an impact is about the wall, not the car,
    and the most damaging way to get this wrong is a wall hit that flips the car
    being read as a rollover: the reported tipping threshold is then whatever mild
    cornering the car happened to be doing when it crashed.
    """
    if len(speed) < 2:
        return None
    dropped = np.diff(speed) < -IMPACT_DROP_M_S
    return int(np.argmax(dropped)) + 1 if bool(dropped.any()) else None


def tip_step(up_axis: np.ndarray) -> int | None:
    """First step at which the car is no longer on its wheels."""
    tipped = up_axis < TIPPED_UP_AXIS
    return int(np.argmax(tipped)) if bool(tipped.any()) else None


def sustained_before(trace: np.ndarray, event: int, *, window: int = 12) -> float:
    """The typical value of ``trace`` shortly before ``event``.

    A median over a window, not the maximum. The car hops as it starts to tip and
    ``|v * yaw_rate|`` spikes with each landing — at 0.8 lock the peak reads
    25.6 m/s^2 against a sustained 14. The peak of a bouncing car is not a
    cornering limit, and using it would raise every speed target above what the
    car holds.
    """
    end = max(int(event) - 4, 1)
    return float(np.median(trace[max(end - window, 0) : end]))


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

    # Braking, up to the point the car stops being a car. Standing on the brakes
    # from top speed pitches this chassis over its nose — measured, not assumed —
    # and a deceleration averaged through a somersault is not a braking limit.
    brake = speed[ACCEL_STEPS:, 1]
    flipped_braking = tip_step(up_axis[ACCEL_STEPS:, 1])
    notes["endo_under_braking_at_step"] = flipped_braking
    if flipped_braking is not None:
        brake = brake[:flipped_braking]
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

    per_angle = []
    for column, level in enumerate(STEER_LEVELS):
        trace = lateral[:, column]
        env = columns[column]
        tip = tip_step(up_axis[:, env])
        hit = impact_step(speed[:, env])

        # A car that hit a wall tells us about the wall. If it went over *after*
        # hitting one, the tip is the crash and not a cornering limit, so the run
        # only counts up to the impact and contributes no rollover evidence.
        if hit is not None and (tip is None or hit <= tip):
            entry = {"a_lat": sustained_before(trace, hit), "hit_wall_at_step": hit}
            window = slice(max(hit - 16, 0), max(hit - 4, 1))
        elif tip is not None:
            entry = {"a_lat": sustained_before(trace, tip), "tipped_at_step": tip}
            window = slice(max(tip - 16, 0), max(tip - 4, 1))
        else:
            # Steady state: the last third, once transients have gone.
            window = slice(len(trace) * 2 // 3, len(trace))
            entry = {"a_lat": float(np.median(trace[window]))}

        # What the wheels were *actually* at, against what was asked for. The
        # steering is an effort-limited servo, so at speed it loses to the tyres
        # and holds far less angle than commanded — which means the car's real
        # minimum radius grows with speed and R_min from full lock is a
        # low-speed figure. teacher.params keeps kappa_max_eff free for this.
        entry["steer"] = level
        entry["steer_angle_rad"] = float(np.median(steer_angle[window, env]))
        entry["speed_m_s"] = float(np.median(speed[window, env]))
        per_angle.append(entry)

    notes["lateral_by_steer"] = per_angle

    # The steering angle the car held at its highest speed, which is the one that
    # decides how tight a corner it can actually take on the racing line.
    fastest = max(per_angle, key=lambda e: e["speed_m_s"])
    notes["steer_at_speed"] = {
        "speed_m_s": fastest["speed_m_s"],
        "commanded_rad": fastest["steer"] * VEHICLE.max_steer_rad,
        "achieved_rad": fastest["steer_angle_rad"],
    }

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
    # Against the plateau the step settles to within a second, not against the
    # value at the end of the run. The servo is 90% there in about nine steps and
    # then creeps for hundreds more as the car's speed and load drift; measured
    # against that final creep the same trace reports a lag of 195 steps — 6.5
    # seconds, which is not a servo.
    after = steer_angle[STEER_STEP_AT:, lag_env]
    plateau_from, plateau_to = 30, 60
    if len(after) > plateau_to:
        final = float(np.median(after[plateau_from:plateau_to]))
    elif len(after):
        final = float(after[-1])
    else:
        final = 0.0
    lag = (
        float(np.argmax(np.abs(after) >= 0.9 * abs(final)))
        if abs(final) > 1e-3
        else float("nan")
    )
    notes["steer_final_angle_rad"] = final
    notes["steer_commanded_rad"] = float(VEHICLE.max_steer_rad)

    # How much of the commanded angle the wheels actually reach once the car is
    # moving. Measured at 0.62 across every steering level at 6.8 m/s, rising to
    # 0.83 at 2.7 m/s: the servo is losing to the tyres. A controller that
    # commands the kinematic angle understeers by that factor everywhere, so the
    # warm start divides its steering gains by this rather than making the search
    # spend generations rediscovering it.
    moving = [e for e in per_angle if e["speed_m_s"] > 1.0 and e["steer"] > 0]
    ratio = (
        float(
            np.median(
                [
                    e["steer_angle_rad"] / (e["steer"] * VEHICLE.max_steer_rad)
                    for e in moving
                ]
            )
        )
        if moving
        else 1.0
    )
    notes["steer_ratio_at_speed"] = ratio

    car = Measured(
        wheelbase_m=wheelbase,
        a_accel_m_s2=a_accel,
        a_brake_m_s2=a_brake,
        a_lat_max_m_s2=a_lat,
        rollover_a_lat_m_s2=rollover,
        steer_lag_steps=lag,
        v_max_m_s=float(max(float(run.max()), VEHICLE.motor_no_load_speed_m_s * 0.5)),
        # The circle the car drove at full lock, which is the minimum radius it
        # actually has. Only trusted when the fit is a circle: a car that was not
        # turning fits an enormous one, and that would silently claim the track is
        # steerable when nothing had been measured.
        full_lock_radius_m=(
            radius if rms < 0.02 * max(radius, 1e-6) and radius < 100.0 else None
        ),
        steer_ratio_at_speed=ratio,
        accel_curve=curve,
        measured=True,
    )
    return car, notes
