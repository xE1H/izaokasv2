"""Tests for :mod:`teacher.warmstart` and :mod:`tools.measured`.

Gate 1 is "the unoptimized controller completes a lap", and it costs a GPU run to
check. Everything that can be verified before spending that is verified here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lituanicax_sdk.track import Track
from lituanicax_sdk.tracks import OFFICIAL
from lituanicax_sdk.vehicle import VEHICLE
from teacher.params import LINE_POINTS, ControllerParams
from teacher.warmstart import build, critically_damped_gains, format_report
from tools.geometry import TrackGeometry
from tools.measured import Measured
from tools.profile import offset_path


@pytest.fixture(scope="module")
def official():
    return TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)


# ══════════════════════════════════════════════════════════════════════════
#  The measured car
# ══════════════════════════════════════════════════════════════════════════


def test_defaults_are_flagged_as_guesses():
    """Nothing downstream should mistake the built-in numbers for measurements."""
    car = Measured()
    assert car.measured is False
    assert "GUESSED" in car.describe()


def test_minimum_turn_radius_matches_the_kinematic_formula():
    car = Measured(wheelbase_m=0.26)
    assert car.r_min_m == pytest.approx(0.26 / math.tan(VEHICLE.max_steer_rad))


@pytest.mark.parametrize(
    "wheelbase,expected", [(0.18, 0.339), (0.22, 0.414), (0.26, 0.490), (0.30, 0.565)]
)
def test_the_wheelbase_radius_table(wheelbase, expected):
    """The table the plan reasons from, pinned so it cannot drift."""
    assert Measured(wheelbase_m=wheelbase).r_min_m == pytest.approx(expected, abs=0.002)


def test_rollover_replaces_the_grip_limit_when_it_is_lower():
    """A 1.27 kg chassis may tip before it slides, and then every speed target
    built on the grip limit is above what the car survives."""
    sliding = Measured(a_lat_max_m_s2=8.0, rollover_a_lat_m_s2=None)
    assert sliding.a_lat_effective_m_s2 == 8.0
    assert not sliding.rollover_limited

    tipping = Measured(a_lat_max_m_s2=12.0, rollover_a_lat_m_s2=7.0)
    assert tipping.a_lat_effective_m_s2 == 7.0
    assert tipping.rollover_limited
    assert "rollover-limited" in tipping.describe()


def test_measured_round_trips_through_json(tmp_path):
    car = Measured(
        wheelbase_m=0.243, a_accel_m_s2=7.1, rollover_a_lat_m_s2=9.4, measured=True
    )
    path = car.save(tmp_path / "dynamics.json")
    loaded = Measured.load(path)
    assert loaded.wheelbase_m == pytest.approx(0.243)
    assert loaded.rollover_a_lat_m_s2 == pytest.approx(9.4)
    assert loaded.measured is True


def test_a_missing_probe_file_falls_back_to_guesses(tmp_path):
    car = Measured.load(tmp_path / "nothing-here.json")
    assert car.measured is False


# ══════════════════════════════════════════════════════════════════════════
#  Gain design
# ══════════════════════════════════════════════════════════════════════════


def test_gains_are_actually_critically_damped():
    """``ζ = 1`` exactly, from the closed form in the docstring.

    For ``ë + (v²k_d/L) ė + (v²k_e/L) e = 0``, ``ζ = (v²k_d/L) / (2ω)``.
    """
    wheelbase, speed, frequency = 0.26, 3.0, 4.0
    k_e, k_d = critically_damped_gains(wheelbase, speed=speed, frequency=frequency)

    omega_squared = speed**2 * k_e / wheelbase
    damping = speed**2 * k_d / wheelbase
    assert math.sqrt(omega_squared) == pytest.approx(frequency)
    assert damping / (2.0 * math.sqrt(omega_squared)) == pytest.approx(1.0)


def test_the_designed_loop_settles_without_overshoot():
    """Simulate the linear model the gains were designed for.

    A pair of gains that looks plausible but is under-damped makes the car weave,
    and the search would spend generations undoing it.
    """
    wheelbase, speed = 0.26, 3.0
    k_e, k_d = critically_damped_gains(wheelbase, speed=speed)

    error, rate, dt = 0.2, 0.0, 1.0 / 30.0
    overshoot = 0.0
    for _ in range(300):
        steer = -(k_e * error + k_d * rate)
        rate += (speed**2 / wheelbase) * steer * dt
        error += rate * dt
        overshoot = min(overshoot, error)
    assert abs(error) < 0.01, "did not settle"
    assert overshoot > -0.02, f"overshot to {overshoot:.4f} m"


def test_gains_scale_with_the_wheelbase():
    short = critically_damped_gains(0.20)
    long = critically_damped_gains(0.32)
    assert long[0] > short[0] and long[1] > short[1]


# ══════════════════════════════════════════════════════════════════════════
#  The warm start
# ══════════════════════════════════════════════════════════════════════════


def test_warm_start_parameters_are_in_bounds(official):
    params, _ = build(official, Measured())
    low, high = ControllerParams.bounds_vector()
    vector = params.to_vector()
    assert np.all(vector >= low - 1e-9)
    assert np.all(vector <= high + 1e-9)
    assert params.line.shape == (LINE_POINTS,)


def test_the_warm_start_line_is_representable_without_loss(official):
    """Regression: solving in 80 control points and refitting onto 40 lost 137 mm
    of a 180 mm corridor and could break the curvature bound with it."""
    _, report = build(official, Measured())
    assert report["line_fit_error_mm"] < 1.0
    assert report["fitted_peak_radius_m"] == pytest.approx(
        report["line"]["peak_radius_m"], rel=0.02
    )


def test_the_warm_start_line_is_steerable(official):
    """The whole reason the bound exists: the controller must start from a line it
    can physically follow."""
    car = Measured(wheelbase_m=0.26)
    params, report = build(official, car)
    _, _, kappa = offset_path(official, report_line(official, params))
    assert float(np.abs(kappa).max()) <= 1.0 / car.r_min_m * 1.02


def report_line(geometry, params):
    from tools.profile import periodic_basis

    return periodic_basis(geometry.num_samples, LINE_POINTS) @ params.line


def test_the_warm_start_beats_the_centerline(official):
    lap = build(official, Measured())[1]["quasi_static_lap_s"]
    assert lap["fitted_line"] < lap["centerline"]


def test_a_long_wheelbase_is_reported_as_unsteerable(official):
    """The Gate 0 answer, and the one that decides whether pure pursuit can work.

    The corridor opens the tightest corner to about 0.545 m. A 0.32 m wheelbase
    needs 0.604 m, so no kinematic line exists and the teacher would have to
    rotate the car with the throttle.
    """
    params, report = build(official, Measured(wheelbase_m=0.32))
    assert not report["track_is_steerable"]
    assert report["r_min_m"] > report["widest_achievable_radius_m"]
    assert "NOT STEERABLE" in format_report(report)
    # It must still return usable parameters — the best the corridor allows.
    low, high = ControllerParams.bounds_vector()
    assert np.all(params.to_vector() >= low - 1e-9)


def test_a_short_wheelbase_is_reported_as_steerable(official):
    _, report = build(official, Measured(wheelbase_m=0.20))
    assert report["track_is_steerable"]
    assert "STEERABLE" in format_report(report)


def test_the_steerability_verdict_uses_the_corridor_not_the_centerline(official):
    """The corridor buys radius, and ignoring that would call the track
    undrivable for a car that can manage it comfortably."""
    _, report = build(official, Measured())
    assert report["widest_achievable_radius_m"] > report["centerline_min_radius_m"]


def test_rollover_limited_cars_get_slower_targets(official):
    """If the rollover threshold binds, it must reach the speed profile."""
    fast, _ = build(official, Measured(a_lat_max_m_s2=12.0))
    slow, _ = build(official, Measured(a_lat_max_m_s2=12.0, rollover_a_lat_m_s2=5.0))
    assert slow.a_lat_eff < fast.a_lat_eff
    assert slow.a_lat_eff == pytest.approx(5.0)


def test_works_on_a_track_it_has_never_seen(circle_track):
    geometry = TrackGeometry.from_track(circle_track)
    params, report = build(geometry, Measured())
    assert report["track_is_steerable"], "a 5 m circle is trivially steerable"
    low, high = ControllerParams.bounds_vector()
    assert np.all(params.to_vector() >= low - 1e-9)
