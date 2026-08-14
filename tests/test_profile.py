"""Tests for :mod:`tools.profile`.

The circle carries most of the weight again: on a constant-radius track the
minimum-time profile is a single known speed, so anything the sweeps get wrong
shows up immediately. The official track then checks the properties that only
matter on a real layout — that the line stays in the corridor, that the
curvature bound actually binds, and that the ranking the module claims in its
docstring is the ranking it produces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lituanicax_sdk.track import Track
from lituanicax_sdk.tracks import OFFICIAL
from tools.geometry import TrackGeometry
from tools.profile import (
    DEFAULT_HALF_WIDTH_M,
    lap_time,
    offset_path,
    profile_lap_time,
    racing_line_offsets,
    three_pass_profile,
)

RADIUS = 5.0
V_MAX = 6.7

#: The constants the line tests are compared under. Guesses until the Phase 0
#: probe measures them; the assertions below are all relative, so they hold
#: whatever the real numbers turn out to be.
EVAL = dict(a_lat=8.0, a_accel=6.0, a_brake=8.0, v_max=V_MAX)


def solve_line(geometry, **kwargs):
    return racing_line_offsets(geometry, a_lat=EVAL["a_lat"], v_max=V_MAX, **kwargs)


@pytest.fixture
def circle(circle_track):
    return TrackGeometry.from_track(circle_track, spacing_m=0.02, smooth_rms_mm=0.0)


@pytest.fixture(scope="module")
def official():
    return TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)


# ══════════════════════════════════════════════════════════════════════════
#  The speed profile
# ══════════════════════════════════════════════════════════════════════════


def test_cornering_ceiling_on_a_circle(circle):
    """On a constant radius the answer is one speed: sqrt(a_lat * R)."""
    _, seg, kappa = offset_path(circle, np.zeros(circle.num_samples))
    speed = three_pass_profile(
        seg, kappa, a_lat=8.0, a_accel=4.0, a_brake=6.0, v_max=V_MAX
    )
    assert np.allclose(speed, math.sqrt(8.0 * RADIUS), rtol=0.01)


def test_top_speed_caps_the_cornering_ceiling(circle):
    """A gentle enough circle is limited by the motor, not by grip."""
    _, seg, kappa = offset_path(circle, np.zeros(circle.num_samples))
    speed = three_pass_profile(
        seg, kappa, a_lat=1000.0, a_accel=100.0, a_brake=100.0, v_max=V_MAX
    )
    assert np.allclose(speed, V_MAX, rtol=1e-6)


def test_lap_time_on_a_circle_is_length_over_speed(circle):
    """The one case where lap time has a closed form."""
    time = profile_lap_time(
        circle,
        np.zeros(circle.num_samples),
        a_lat=1000.0,
        a_accel=100.0,
        a_brake=100.0,
        v_max=V_MAX,
    )
    assert float(time) == pytest.approx(circle.length / V_MAX, rel=1e-4)


def test_profile_never_exceeds_the_cornering_ceiling(official):
    """The sweeps may only ever take speed away."""
    _, seg, kappa = offset_path(official, np.zeros(official.num_samples))
    ceiling = np.minimum(V_MAX, np.sqrt(8.0 / np.maximum(np.abs(kappa), 1e-9)))
    speed = three_pass_profile(
        seg, kappa, a_lat=8.0, a_accel=4.0, a_brake=6.0, v_max=V_MAX
    )
    assert np.all(speed <= ceiling + 1e-9)
    assert np.all(speed <= V_MAX + 1e-9)


def test_profile_respects_the_acceleration_limit(official):
    """Consecutive speeds must be reachable under a_accel and a_brake."""
    _, seg, kappa = offset_path(official, np.zeros(official.num_samples))
    a_accel, a_brake = 4.0, 6.0
    speed = three_pass_profile(
        seg, kappa, a_lat=8.0, a_accel=a_accel, a_brake=a_brake, v_max=V_MAX
    )
    ahead = np.roll(speed, -1)
    # v_next^2 - v^2 = 2 a ds, within a hair for float error.
    gain = (ahead**2 - speed**2) / (2.0 * seg)
    assert np.all(gain <= a_accel + 1e-6)
    assert np.all(gain >= -a_brake - 1e-6)


def test_profile_wraps_around_the_start_finish_line(official):
    """A braking zone just after s = 0 must slow the car down before it.

    This is the failure the ``iterations`` argument exists for: a single
    forward/backward sweep leaves the segment before the seam untouched, and the
    car arrives at the first corner of the lap far too fast.
    """
    _, seg, kappa = offset_path(official, np.zeros(official.num_samples))
    speed = three_pass_profile(
        seg, kappa, a_lat=8.0, a_accel=4.0, a_brake=6.0, v_max=V_MAX
    )
    gain = (np.roll(speed, -1) ** 2 - speed**2) / (2.0 * seg)
    # The constraint holds at the seam specifically, not just on average.
    assert gain[-1] <= 4.0 + 1e-6
    assert gain[-1] >= -6.0 - 1e-6


def test_profile_is_batched(official):
    """A whole CMA-ES population is scored in one call."""
    lines = np.stack(
        [
            np.zeros(official.num_samples),
            np.full(official.num_samples, 0.1),
            np.full(official.num_samples, -0.1),
        ]
    )
    _, seg, kappa = offset_path(official, lines)
    speed = three_pass_profile(
        seg, kappa, a_lat=8.0, a_accel=4.0, a_brake=6.0, v_max=V_MAX
    )
    assert speed.shape == (3, official.num_samples)

    times = lap_time(seg, speed)
    assert times.shape == (3,)
    # Each row must equal what it gets computed on its own.
    for i, line in enumerate(lines):
        alone = profile_lap_time(
            official, line, a_lat=8.0, a_accel=4.0, a_brake=6.0, v_max=V_MAX
        )
        assert float(times[i]) == pytest.approx(float(alone), rel=1e-9)


def test_more_grip_is_never_slower(official):
    _, seg, kappa = offset_path(official, np.zeros(official.num_samples))
    times = [
        float(
            lap_time(
                seg,
                three_pass_profile(
                    seg, kappa, a_lat=a, a_accel=6.0, a_brake=8.0, v_max=V_MAX
                ),
            )
        )
        for a in (6.0, 8.0, 10.0, 12.0)
    ]
    assert times == sorted(times, reverse=True)


def test_profile_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        three_pass_profile(
            np.ones(10), np.ones(11), a_lat=8.0, a_accel=4.0, a_brake=6.0, v_max=6.7
        )


# ══════════════════════════════════════════════════════════════════════════
#  The racing line
# ══════════════════════════════════════════════════════════════════════════


def test_a_circle_tightens_to_the_inside(circle):
    """On a closed circle the fastest line is the *tightest* one, not the widest.

    Grip-limited time round a circle of radius R is ``2πR / sqrt(a_lat R)``, i.e.
    ``2π sqrt(R / a_lat)`` — proportional to sqrt(R), so a smaller radius is
    faster. Path length falls linearly while speed only falls as the square root.
    (Checked in closed form: 4.877 s at R = 4.82 against 5.056 s at R = 5.18.)

    'Wider is faster' is a property of corners embedded in a larger track, where
    entry and exit speed matter. It is not true of a circle, and a solver that
    got this backwards would be optimizing something other than lap time.
    """
    n, report = solve_line(circle, half_width=DEFAULT_HALF_WIDTH_M)
    assert n.min() > 0.5 * DEFAULT_HALF_WIDTH_M, "should tighten inward"
    radius = np.linalg.norm(offset_path(circle, n)[0], axis=-1)
    assert float(radius.mean()) == pytest.approx(
        RADIUS - DEFAULT_HALF_WIDTH_M, abs=0.02
    )
    assert report["peak_radius_m"] < RADIUS


def test_offsets_stay_inside_the_corridor(official):
    """The corridor is what keeps the car off the wall; it is not advisory."""
    for half_width in (0.05, 0.10, DEFAULT_HALF_WIDTH_M):
        n, _ = solve_line(official, half_width=half_width)
        assert np.abs(n).max() <= half_width + 1e-9, half_width


def test_the_line_is_faster_than_the_centerline(official):
    """The whole reason to move off the centerline."""
    n, _ = solve_line(official, half_width=DEFAULT_HALF_WIDTH_M)
    centre = float(profile_lap_time(official, np.zeros(official.num_samples), **EVAL))
    line = float(profile_lap_time(official, n, **EVAL))
    assert line < centre, f"line {line:.2f} s vs centerline {centre:.2f} s"


def test_the_line_shortens_the_path(official):
    """Cutting corners is most of where the time comes from on this layout."""
    n, report = solve_line(official, half_width=DEFAULT_HALF_WIDTH_M)
    assert report["path_length_m"] < official.length - 1.0


@pytest.mark.parametrize("radius", [0.41, 0.45, 0.49])
def test_the_curvature_bound_is_respected(official, radius):
    """A line the car cannot steer is not a line.

    These radii are all inside what the corridor can achieve (see
    :func:`widest_achievable_radius`), and the unbounded line's apex is 0.249 m,
    so the bound genuinely binds at every value here.
    """
    kappa_max = 1.0 / radius
    n, report = solve_line(
        official, half_width=DEFAULT_HALF_WIDTH_M, kappa_max=kappa_max
    )
    assert np.abs(n).max() <= DEFAULT_HALF_WIDTH_M + 1e-9
    assert report["within_kappa_max"], report
    assert not report.get("fell_back_to_centerline"), report
    _, _, kappa = offset_path(official, n)
    assert float(np.abs(kappa).max()) <= kappa_max * 1.001


def test_an_impossible_bound_is_reported_not_faked(official):
    """The corridor cannot open the tightest corner past about 0.55 m.

    Asking for more must come back saying so, rather than quietly returning a
    line that violates the bound — a silently unsteerable warm start would show
    up much later as a teacher that cannot complete a lap.
    """
    n, report = solve_line(
        official, half_width=DEFAULT_HALF_WIDTH_M, kappa_max=1.0 / 0.80
    )
    assert not report["within_kappa_max"]
    assert report.get("fell_back_to_centerline")
    assert np.abs(n).max() == 0.0


def test_widest_achievable_radius_bounds_the_car(official):
    """Gate 0's decisive number: can the car physically steer this track?

    The corridor buys radius over the centerline, but not unlimited radius. If
    the probe's ``R_min = L_wb / tan(0.488)`` exceeds this, the tightest corner
    cannot be taken by steering alone at any speed.
    """
    from tools.profile import widest_achievable_radius

    centerline_radius = 1.0 / float(official.kappa.abs().max())
    widest, n = widest_achievable_radius(official, half_width=DEFAULT_HALF_WIDTH_M)

    assert widest > centerline_radius, "the corridor must buy some radius"
    assert np.abs(n).max() <= DEFAULT_HALF_WIDTH_M + 1e-9
    # Measured at 0.50-0.55 m against a 0.366 m centerline. Bracketed loosely so
    # the test states the finding without pinning the optimizer's last digit.
    assert 0.45 < widest < 0.70, f"widest achievable radius {widest:.3f} m"


def test_a_bound_inside_the_achievable_radius_always_succeeds(official):
    """Consistency between the two functions: what one says is possible, the
    other must be able to deliver."""
    from tools.profile import widest_achievable_radius

    widest, _ = widest_achievable_radius(official, half_width=DEFAULT_HALF_WIDTH_M)
    # Ask for comfortably less than the maximum, so optimizer noise cannot
    # straddle the boundary.
    _, report = solve_line(
        official, half_width=DEFAULT_HALF_WIDTH_M, kappa_max=1.0 / (0.9 * widest)
    )
    assert report["within_kappa_max"], report


def test_the_bounded_line_still_beats_the_centerline(official):
    """The bound must not cost so much that the line stops being worth having.

    This is the assertion that would have caught the two failed objectives: the
    true-curvature min-Σκ² line came out *slower* than doing nothing.
    """
    n, report = solve_line(
        official, half_width=DEFAULT_HALF_WIDTH_M, kappa_max=1.0 / 0.49
    )
    centre = float(profile_lap_time(official, np.zeros(official.num_samples), **EVAL))
    line = float(profile_lap_time(official, n, **EVAL))
    assert report["within_kappa_max"]
    assert line < centre, f"bounded line {line:.2f} s vs centerline {centre:.2f} s"


def test_the_curvature_bound_costs_time(official):
    """Priced by measurement rather than asserted from memory.

    The plan flags an earlier estimate of this cost as untrustworthy; this
    replaces it. A steerable line cannot be faster than an unsteerable one.
    """
    free, _ = solve_line(official, half_width=DEFAULT_HALF_WIDTH_M)
    bound, report = solve_line(
        official, half_width=DEFAULT_HALF_WIDTH_M, kappa_max=1.0 / 0.49
    )
    fast = float(profile_lap_time(official, free, **EVAL))
    steerable = float(profile_lap_time(official, bound, **EVAL))
    assert report["within_kappa_max"]
    assert fast <= steerable + 1e-6
    assert steerable - fast < 1.0, f"bound costs {steerable - fast:.3f} s"


def test_the_line_beats_a_uniform_offset(official):
    """Guards against the optimizer degenerating into something trivial."""
    n, _ = solve_line(official, half_width=DEFAULT_HALF_WIDTH_M)
    best = float(profile_lap_time(official, n, **EVAL))
    for uniform in (-0.18, -0.09, 0.09, 0.18):
        flat = float(
            profile_lap_time(official, np.full(official.num_samples, uniform), **EVAL)
        )
        assert best < flat, uniform


def test_works_on_a_track_it_has_never_seen(circle_track):
    """Nothing may be specific to the official layout."""
    geometry = TrackGeometry.from_track(circle_track)
    n, report = solve_line(geometry, kappa_max=1.0 / 0.49)
    assert report["within_kappa_max"]
    assert np.abs(n).max() <= DEFAULT_HALF_WIDTH_M + 1e-9
