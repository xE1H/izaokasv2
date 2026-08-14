"""Tests for :mod:`tools.probe_analysis`.

The probe cannot run here — it needs Isaac Sim — but its arithmetic can, and the
arithmetic is where the mistakes live: a sign, a window, an off-by-one in a step
count. So the analysis is driven against synthetic traces whose answers are known
in closed form.

This matters more than usual because the probe's output is the input to
everything else. A wheelbase that is 20% wrong makes ``R_min`` 20% wrong, which
moves the Gate 0 verdict, which decides whether the project proceeds.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lituanicax_sdk.track import Track
from lituanicax_sdk.tracks import OFFICIAL
from lituanicax_sdk.vehicle import VEHICLE
from tools.geometry import TrackGeometry
from tools.probe_analysis import (
    ACCEL_STEPS,
    STEER_LEVELS,
    STEER_STEP_AT,
    TOTAL_STEPS,
    Recorder,
    analyse,
    build_script,
    fit_circle,
    layout,
    measure_accel_curve,
    measure_wheelbase,
    probe_track,
    straightest_point,
)

DT = 1.0 / 30.0


# ══════════════════════════════════════════════════════════════════════════
#  The scaled probe track
# ══════════════════════════════════════════════════════════════════════════


def test_probe_track_scales_mesh_and_centerline_together():
    """The invariant ``TrackCfg`` warns about: whatever scales the mesh scales the
    line, or the line stops describing the track."""
    track = probe_track(40.0)
    assert track.mesh_scale == pytest.approx(OFFICIAL.mesh_scale * 40.0)
    assert track.centerline_scale[0] == pytest.approx(
        OFFICIAL.centerline_scale[0] * 40.0
    )
    assert track.centerline_scale[1] == pytest.approx(
        OFFICIAL.centerline_scale[1] * 40.0
    )
    assert track.name != OFFICIAL.name, "must not shadow the official track"


def test_probe_track_scales_the_gate_and_the_corner_threshold():
    """Curvature has units of 1/length, so its threshold scales inversely."""
    track = probe_track(40.0)
    assert track.lap_gate_window_m == pytest.approx(OFFICIAL.lap_gate_window_m * 40.0)
    assert track.corner_curvature_threshold == pytest.approx(
        OFFICIAL.corner_curvature_threshold / 40.0
    )


def test_the_scaled_track_really_has_room():
    """The whole reason for scaling: the official 0.70 m corridor cannot hold a
    50 m acceleration run."""
    geometry = TrackGeometry.from_track(
        Track(probe_track(40.0), device="cpu"), spacing_m=2.0
    )
    assert geometry.length > 1500.0
    # The tightest corner becomes gentle enough to drive at speed.
    assert 1.0 / float(geometry.kappa.abs().max()) > 10.0


def test_straightest_point_finds_a_straight():
    """It must beat the origin, which on the official track is not on a straight."""
    geometry = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
    x, y, yaw = straightest_point(geometry, run_m=5.0)

    found_s, _ = geometry.project(torch.tensor([[x, y]]))
    window = int(5.0 / geometry.spacing_m)
    start = int(found_s.item() / geometry.spacing_m)
    stretch = torch.roll(geometry.kappa.abs(), -start)[:window]
    everywhere = geometry.kappa.abs()
    assert float(stretch.mean()) < float(everywhere.mean())
    assert math.isfinite(yaw)


def test_straightest_point_on_a_circle_is_arbitrary_but_valid(circle_track):
    """A circle has no straight, so any point will do — but it must still return
    a pose on the track."""
    geometry = TrackGeometry.from_track(circle_track, smooth_rms_mm=0.0)
    x, y, _ = straightest_point(geometry, run_m=2.0)
    assert math.hypot(x, y) == pytest.approx(5.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════
#  Circle fitting
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("radius", [0.5, 2.0, 13.7])
def test_fit_circle_recovers_a_known_radius(radius):
    theta = np.linspace(0.3, 4.0, 200)
    points = np.stack(
        [3.0 + radius * np.cos(theta), -1.5 + radius * np.sin(theta)], axis=1
    )
    fitted, rms = fit_circle(points)
    assert fitted == pytest.approx(radius, rel=1e-6)
    assert rms < 1e-6


def test_fit_circle_reports_a_bad_fit_as_high_rms():
    """A car that is sliding does not follow a circle, and the rms is how the
    report says so rather than quietly returning a meaningless radius."""
    theta = np.linspace(0.0, 3.0, 200)
    points = np.stack([np.cos(theta) * 2.0, np.sin(theta) * 5.0], axis=1)  # ellipse
    _, rms = fit_circle(points)
    assert rms > 0.1


def test_fit_circle_survives_a_partial_arc():
    theta = np.linspace(0.0, 0.7, 60)
    points = np.stack([4.0 * np.cos(theta), 4.0 * np.sin(theta)], axis=1)
    fitted, _ = fit_circle(points)
    assert fitted == pytest.approx(4.0, rel=1e-4)


# ══════════════════════════════════════════════════════════════════════════
#  Wheelbase
# ══════════════════════════════════════════════════════════════════════════


def fake_bodies(wheelbase: float):
    """A ``robot.data`` stub with the SDK's own wheel link names."""
    names = list(VEHICLE.wheel_links)
    positions = []
    for name in names:
        x = wheelbase / 2.0 if "front" in name else -wheelbase / 2.0
        y = 0.08 if "left" in name else -0.08
        positions.append([x, y, 0.037])
    return SimpleNamespace(
        body_names=names, body_pos_w=np.asarray(positions)[None, ...]
    )


@pytest.mark.parametrize("wheelbase", [0.18, 0.26, 0.32])
def test_measure_wheelbase_from_body_positions(wheelbase):
    measured, how = measure_wheelbase(fake_bodies(wheelbase))
    assert measured == pytest.approx(wheelbase, abs=1e-6)
    assert "body positions" in how


def test_measure_wheelbase_reports_when_it_cannot():
    """Better a NaN and an explanation than a number nobody can trace."""
    stub = SimpleNamespace(body_names=["chassis"], body_pos_w=np.zeros((1, 1, 3)))
    measured, how = measure_wheelbase(stub)
    assert math.isnan(measured)
    assert "no front/back" in how


# ══════════════════════════════════════════════════════════════════════════
#  Acceleration curve
# ══════════════════════════════════════════════════════════════════════════


def test_accel_curve_recovers_a_constant_acceleration():
    accel = 5.0
    time = np.arange(0, 120) * DT
    curve = measure_accel_curve(accel * time, DT)
    assert curve
    assert all(value == pytest.approx(accel, rel=0.05) for _, value in curve)


def test_accel_curve_shows_the_motor_falling_off():
    """The SDK's motor is a DC torque-speed curve, so acceleration must decay —
    reporting one constant would overstate the car at speed."""
    v_max = VEHICLE.motor_no_load_speed_m_s
    time = np.arange(0, 300) * DT
    speed = v_max * (1.0 - np.exp(-time / 0.8))
    curve = measure_accel_curve(speed, DT)
    values = [value for _, value in curve]
    assert len(values) > 3
    assert values[0] > values[-1]


def test_accel_curve_on_a_stub_trace_is_empty_not_a_crash():
    assert measure_accel_curve(np.array([0.0, 1.0]), DT) == []


# ══════════════════════════════════════════════════════════════════════════
#  The command script
# ══════════════════════════════════════════════════════════════════════════


def test_the_script_covers_every_measurement_in_one_run():
    lateral, lag, count = layout()
    script = build_script(count, lateral, lag)
    early, late = script(0), script(ACCEL_STEPS + 10)

    assert early.shape == (count, 2)
    # env 0: full lock, gentle throttle.
    assert float(early[0, 1]) == pytest.approx(1.0)
    assert 0.0 < float(early[0, 0]) < 0.5
    # env 1: accelerate, then brake.
    assert float(early[1, 0]) == pytest.approx(1.0)
    assert float(late[1, 0]) == pytest.approx(-1.0)
    # the lateral sweep holds each level at full throttle.
    for offset, index in enumerate(lateral):
        assert float(early[index, 1]) == pytest.approx(STEER_LEVELS[offset])
        assert float(early[index, 0]) == pytest.approx(1.0)


def test_the_steering_step_happens_once_and_late():
    lateral, lag, count = layout()
    script = build_script(count, lateral, lag)
    assert float(script(STEER_STEP_AT - 1)[lag, 1]) == 0.0
    assert float(script(STEER_STEP_AT)[lag, 1]) == pytest.approx(1.0)


def test_every_command_is_within_the_action_bounds():
    lateral, lag, count = layout()
    script = build_script(count, lateral, lag)
    for step in (0, ACCEL_STEPS - 1, ACCEL_STEPS, TOTAL_STEPS - 1):
        actions = script(step)
        assert float(actions.min()) >= -1.0
        assert float(actions.max()) <= 1.0


# ══════════════════════════════════════════════════════════════════════════
#  End-to-end analysis on a synthetic run
# ══════════════════════════════════════════════════════════════════════════


def synthetic_recorder(*, tip_at=None, turn_radius=0.49, a_accel=6.0, a_brake=8.0):
    """A full recorder trace with known answers baked in."""
    lateral, lag, count = layout()
    recorder = Recorder()
    v_max = VEHICLE.motor_no_load_speed_m_s

    for step in range(TOTAL_STEPS + 1):
        time = step * DT
        speed = np.zeros(count)
        yaw_rate = np.zeros(count)
        up_axis = np.ones(count)
        steer = np.zeros(count)
        position = np.zeros((count, 2))

        # env 0 — a steady full-lock circle at 1.5 m/s.
        circle_speed = 1.5
        angle = circle_speed * time / turn_radius
        position[0] = [turn_radius * math.cos(angle), turn_radius * math.sin(angle)]
        speed[0] = circle_speed
        yaw_rate[0] = circle_speed / turn_radius

        # env 1 — accelerate at a_accel to v_max, then brake at a_brake.
        if step < ACCEL_STEPS:
            speed[1] = min(v_max, a_accel * time)
        else:
            peak = min(v_max, a_accel * ACCEL_STEPS * DT)
            speed[1] = max(0.0, peak - a_brake * (step - ACCEL_STEPS) * DT)

        # lateral sweep — a_lat grows with steering, saturating at 9 m/s^2.
        for offset, index in enumerate(lateral):
            level = STEER_LEVELS[offset]
            target = min(9.0, 12.0 * level)
            speed[index] = 4.0
            yaw_rate[index] = target / 4.0
            steer[index] = level * VEHICLE.max_steer_rad
            if tip_at is not None and level >= tip_at and step > 300:
                up_axis[index] = 0.1

        # lag env — first-order lag with a 3-step time constant.
        if step >= STEER_STEP_AT:
            elapsed = step - STEER_STEP_AT
            steer[lag] = VEHICLE.max_steer_rad * (1.0 - math.exp(-elapsed / 3.0))

        recorder.frames["speed_forward"].append(
            torch.tensor(speed, dtype=torch.float32)
        )
        recorder.frames["speed_lateral"].append(torch.zeros(count))
        recorder.frames["yaw_rate"].append(torch.tensor(yaw_rate, dtype=torch.float32))
        recorder.frames["up_axis"].append(torch.tensor(up_axis, dtype=torch.float32))
        recorder.frames["roll"].append(torch.zeros(count))
        recorder.frames["steer_angle"].append(torch.tensor(steer, dtype=torch.float32))
        recorder.frames["wheel_speed"].append(torch.tensor(speed, dtype=torch.float32))
        recorder.position.append(torch.tensor(position, dtype=torch.float32))

    return recorder, lateral, lag


def test_analyse_recovers_the_synthetic_answers():
    recorder, lateral, lag = synthetic_recorder()
    car, notes = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)

    assert car.measured
    assert car.wheelbase_m == pytest.approx(0.26, abs=1e-6)
    assert notes["full_lock_radius_m"] == pytest.approx(0.49, rel=0.01)
    assert car.a_accel_m_s2 == pytest.approx(6.0, rel=0.1)
    assert car.a_brake_m_s2 == pytest.approx(8.0, rel=0.1)
    assert car.a_lat_max_m_s2 == pytest.approx(9.0, rel=0.05)
    assert car.rollover_a_lat_m_s2 is None
    # First-order lag with tau = 3 steps reaches 90% at about 7 steps.
    assert car.steer_lag_steps == pytest.approx(7.0, abs=1.5)


def test_analyse_detects_a_rollover_and_lets_it_bind():
    """The measurement that changes every speed target if it fires.

    The synthetic car tips at the two largest steering angles, having reached
    9.0 m/s^2 there, and sustains 7.2 m/s^2 at the largest angle it survived. So:

    * ``a_lat_max`` is 7.2 — the highest it *sustained*. The 9.0 the tipping runs
      touched is not evidence about grip, it is where the car went over.
    * ``rollover`` is 9.0.
    * the effective limit is 7.2, the conservative one. Asking the car for 9.0
      because it once reached it before rolling would be asking it to roll.
    """
    recorder, lateral, lag = synthetic_recorder(tip_at=0.8)
    car, notes = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)

    assert car.rollover_limited, "a tip was observed, so the ceiling is tipping"
    assert car.rollover_a_lat_m_s2 == pytest.approx(9.0, rel=0.05)
    assert car.a_lat_max_m_s2 == pytest.approx(7.2, rel=0.05)
    assert car.a_lat_effective_m_s2 == pytest.approx(car.a_lat_max_m_s2)
    assert notes["tipped_at_steer"] == [0.8, 1.0]


def test_a_wall_impact_is_not_read_as_a_rollover():
    """The bug that made the first real probe run useless.

    A car commanded to a gentle steering angle holds far less than it asked for,
    drives a wide circle, walks out of the corridor and hits a wall at 6.7 m/s.
    The impact flips it, so ``up_axis`` drops below 0.3 and the old code called
    that a rollover — at whatever mild lateral acceleration it happened to be
    pulling. The reported tipping threshold came back as 3.2 m/s^2, which would
    have put every speed target on the track at a third of what the car can do.
    """
    recorder, lateral, lag = synthetic_recorder()
    hit = 400
    for step in range(hit, TOTAL_STEPS + 1):
        # Gentlest steering angle only: speed collapses, then the car goes over.
        recorder.frames["speed_forward"][step][lateral[0]] = -0.9
        if step > hit:
            recorder.frames["up_axis"][step][lateral[0]] = -0.5

    car, notes = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)

    assert notes["lateral_by_steer"][0]["hit_wall_at_step"] == hit
    assert "tipped_at_step" not in notes["lateral_by_steer"][0]
    assert not car.rollover_limited, "hitting a wall is not tipping over"
    assert car.rollover_a_lat_m_s2 is None


def test_braking_is_measured_up_to_the_point_the_car_flips():
    """Standing on the brakes from top speed pitches this chassis over its nose.

    Averaging the deceleration through the somersault that follows is not a
    braking limit — the car is no longer braking, it is tumbling.
    """
    recorder, lateral, lag = synthetic_recorder(a_brake=8.0)
    endo = 20
    for step in range(ACCEL_STEPS + endo, TOTAL_STEPS + 1):
        recorder.frames["up_axis"][step][1] = -0.4
        recorder.frames["speed_forward"][step][1] = 0.0

    car, notes = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)

    assert notes["endo_under_braking_at_step"] == endo
    assert car.a_brake_m_s2 == pytest.approx(8.0, rel=0.15)


def test_steering_lag_ignores_a_slow_later_creep():
    """The servo settles in a few steps; the car's load then drifts for minutes.

    Measured against the angle at the *end* of the run rather than the plateau it
    settles to, the same trace reported 195 steps of lag — 6.5 seconds, which is
    not a servo.
    """
    recorder, lateral, lag = synthetic_recorder()
    for step in range(400, TOTAL_STEPS + 1):
        recorder.frames["steer_angle"][step][lag] *= 1.15

    car, _ = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)
    assert car.steer_lag_steps == pytest.approx(7.0, abs=1.5)


def test_a_car_that_never_tips_is_not_rollover_limited():
    recorder, lateral, lag = synthetic_recorder(tip_at=None)
    car, notes = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)
    assert not car.rollover_limited
    assert car.rollover_a_lat_m_s2 is None
    assert notes["tipped_at_steer"] == []
    assert car.a_lat_effective_m_s2 == pytest.approx(car.a_lat_max_m_s2)


def test_analyse_reports_that_the_lateral_limit_varies_with_steering():
    recorder, lateral, lag = synthetic_recorder()
    _, notes = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)
    assert notes["a_lat_varies_with_steer"]
    assert notes["a_lat_spread_m_s2"] > 1.0


def test_analyse_falls_back_to_the_circle_when_body_names_do_not_match():
    """A wheelbase from the turning circle is worse than one from geometry, but
    far better than none — and the report says which it used."""
    recorder, lateral, lag = synthetic_recorder(turn_radius=0.49)
    stub = SimpleNamespace(body_names=["chassis"], body_pos_w=np.zeros((1, 1, 3)))
    car, notes = analyse(recorder, stub, lateral, lag, dt=DT)

    assert not math.isnan(car.wheelbase_m)
    assert "full-lock circle" in notes["wheelbase_source"]
    assert car.wheelbase_m == pytest.approx(
        0.49 * math.tan(VEHICLE.max_steer_rad), rel=0.02
    )


def test_analyse_flags_a_wheelbase_disagreement():
    """A big gap between geometry and the turning circle means the car is sliding,
    and ``R_min`` from the geometric figure is then optimistic."""
    recorder, lateral, lag = synthetic_recorder(turn_radius=2.0)
    _, notes = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)
    assert "wheelbase_disagreement" in notes


def test_the_measured_result_round_trips_to_disk(tmp_path):
    recorder, lateral, lag = synthetic_recorder()
    car, _ = analyse(recorder, fake_bodies(0.26), lateral, lag, dt=DT)
    from tools.measured import Measured

    reloaded = Measured.load(car.save(tmp_path / "dynamics.json"))
    assert reloaded.measured
    assert reloaded.wheelbase_m == pytest.approx(car.wheelbase_m)
    assert reloaded.r_min_m == pytest.approx(car.r_min_m)
