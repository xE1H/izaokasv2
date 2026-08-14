"""Tests for :mod:`tools.geometry`.

Two kinds. A 5 m circle, where every answer is known in closed form and an error
is unambiguous; and the real official track, where the answers are not known but
the invariants still hold — a closed loop turns through exactly one revolution
whatever shape it is.
"""

from __future__ import annotations

import math

import pytest
import torch

from lituanicax_sdk.track import Track
from lituanicax_sdk.tracks import OFFICIAL
from tools.geometry import TrackGeometry

RADIUS = 5.0


def wrapped_error(got: torch.Tensor, want: torch.Tensor, length: float) -> float:
    """Largest ``|got - want|`` treating arc length as circular.

    Without this, a point whose nearest sample lands just before the seam reads
    as a whole lap of error when it is in fact correct — ``s = 49.99`` and
    ``s = 0`` are the same place.
    """
    error = torch.remainder(got - want + 0.5 * length, length) - 0.5 * length
    return float(error.abs().max())


@pytest.fixture
def circle(circle_track):
    """An exact circle, fitted without smoothing.

    A synthetic circle has no export noise to smooth, so smoothing here would
    only move the answer away from the one closed form says it should be. The
    smoothing path is exercised against the real track below, where there is
    something to smooth.
    """
    return TrackGeometry.from_track(circle_track, spacing_m=0.02, smooth_rms_mm=0.0)


@pytest.fixture(scope="module")
def official():
    return TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)


# ══════════════════════════════════════════════════════════════════════════
#  The circle: closed-form answers
# ══════════════════════════════════════════════════════════════════════════


def test_circle_length(circle):
    assert circle.length == pytest.approx(2.0 * math.pi * RADIUS, abs=1e-3)


def test_circle_curvature_is_constant_and_correct(circle):
    # Counterclockwise, so the track turns left: curvature is +1/R.
    #
    # 1% rather than exact: a cubic spline is C2, but its second derivative is
    # only piecewise linear, so curvature ripples slightly between knots even on
    # an exactly circular input. Measured ripple is +/-0.15%, four hundred times
    # smaller than the difference between a hairpin and a straight.
    assert torch.allclose(
        circle.kappa, torch.full_like(circle.kappa, 1.0 / RADIUS), rtol=0.01
    )


def test_circle_samples_are_on_the_circle(circle):
    assert torch.allclose(
        circle.pos.norm(dim=-1), torch.full((circle.num_samples,), RADIUS), atol=1e-4
    )


def test_circle_tangent_is_perpendicular_to_the_radius(circle):
    radial = circle.pos / circle.pos.norm(dim=-1, keepdim=True)
    assert torch.allclose(
        (radial * circle.tangent).sum(dim=-1),
        torch.zeros(circle.num_samples),
        atol=1e-4,
    )


def test_circle_normal_points_inward_for_a_left_turn(circle):
    # Left normal on a counterclockwise circle points at the centre.
    radial = circle.pos / circle.pos.norm(dim=-1, keepdim=True)
    assert torch.all((radial * circle.normal).sum(dim=-1) < -0.99)


def test_circle_projection_recovers_offset(circle):
    # A point 0.2 m inside the circle is 0.2 m to the *left* of the direction of
    # travel, so n is +0.2.
    theta = torch.tensor([0.0, 1.0, 2.5, 4.0])
    inside = (RADIUS - 0.2) * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    s, n = circle.project(inside)
    assert torch.allclose(n, torch.full_like(n, 0.2), atol=2e-3)
    assert wrapped_error(s, theta * RADIUS, circle.length) < 2e-3


# ══════════════════════════════════════════════════════════════════════════
#  The plan's stated acceptance criteria, on the real track
# ══════════════════════════════════════════════════════════════════════════


def test_official_arc_length_is_50_m(official):
    """The competition describes a 50 m track; measure it rather than trust it."""
    assert official.length == pytest.approx(50.0, abs=0.2)


def test_official_total_turning_is_one_revolution(official):
    """∮κ ds = ±2π for any simple closed loop."""
    assert abs(official.total_turning()) == pytest.approx(2.0 * math.pi, abs=0.05)


def test_official_reintegrating_curvature_closes_the_loop(official):
    """Integrating the tangents back up from κ must return to the start."""
    assert official.closure_error() < 0.1


def test_official_projection_round_trips(official):
    """project(point_at(s)) == s, to within a millimetre."""
    s = torch.linspace(0.0, official.length, 501)[:-1]
    recovered, n = official.project(official.point_at(s))
    # Wrapped difference, so a point landing just below 0 does not read as a
    # whole lap of error.
    error = torch.remainder(recovered - s + 0.5 * official.length, official.length)
    error -= 0.5 * official.length
    assert float(error.abs().max()) < 1e-3
    assert float(n.abs().max()) < 1e-3


def test_official_projection_round_trips_with_an_offset(official):
    """Same, for points held off the centerline inside the corridor."""
    s = torch.linspace(0.0, official.length, 301)[:-1]
    for offset in (-0.18, -0.05, 0.05, 0.18):
        recovered, n = official.project(official.point_at(s, offset))
        error = torch.remainder(recovered - s + 0.5 * official.length, official.length)
        error -= 0.5 * official.length
        # An offset point's true foot moves slightly with curvature, so the
        # tolerance is looser than on the centerline itself, but it is still
        # sub-millimetre against the 0.15 m crash radius.
        assert float(error.abs().max()) < 5e-3, offset
        assert float((n - offset).abs().max()) < 5e-3, offset


def test_official_smoothing_stays_close_to_the_raw_centerline(official):
    """The spline must still be *this* track, not a nearby one.

    Guards the smoothing parameter: too much and the line leaves the corridor,
    which no other test would notice because every downstream quantity is
    computed from the spline alone.
    """
    raw = Track(OFFICIAL, device="cpu").points
    _, n = official.project(raw)
    assert float(n.abs().max()) < 0.01  # 10 mm; measures ~2 mm


def test_official_curvature_is_not_spacing_noise(official):
    """Smoothing must actually suppress the ringing an interpolating fit shows.

    Measured: Menger curvature on the raw points peaks at 2.95 1/m and an
    interpolating spline at 4.13 1/m, both artefacts of 17-115 mm spacing. The
    smoothed track peaks near 2.5.
    """
    assert float(official.kappa.abs().max()) < 2.9


def test_official_geometry_validates_clean(official):
    assert official.validate() == []


# ══════════════════════════════════════════════════════════════════════════
#  The seam, and resampling
# ══════════════════════════════════════════════════════════════════════════


def test_the_start_finish_seam_is_continuous(official):
    """s = 0 and s = length are the same place, approached from either side.

    Everything downstream wraps — the speed profile iterates round the loop, the
    controller looks ahead past the line — so a discontinuity here would show up
    as one corner the car mysteriously cannot drive.
    """
    before = official.length - 1e-4
    across = torch.tensor([before, 0.0, 1e-4])
    pos, tangent, kappa = official.at(across)

    assert float((pos[0] - pos[1]).norm()) < 1e-3
    assert float((pos[1] - pos[2]).norm()) < 1e-3
    assert float((tangent[0] - tangent[1]).norm()) < 1e-3
    assert float((kappa[0] - kappa[1]).abs()) < 1e-2


def test_lookahead_past_the_line_wraps(official):
    """Looking 3 m ahead from 1 m before the line lands 2 m after it."""
    s = torch.tensor([official.length - 1.0])
    ahead = official.point_at(s + 3.0)
    expected = official.point_at(torch.tensor([2.0]))
    assert float((ahead - expected).norm()) < 1e-3


def test_at_accepts_unwrapped_and_negative_arc_length(official):
    """Negative and multi-lap s must behave, so callers never have to clamp."""
    same = torch.tensor([1.0, 1.0 + official.length, 1.0 - official.length])
    pos, _, _ = official.at(same)
    assert float((pos[0] - pos[1]).norm()) < 1e-3
    assert float((pos[0] - pos[2]).norm()) < 1e-3


def test_resampled_spacing_is_uniform(official):
    """The whole point of the module: the raw track is 17-115 mm, this is not."""
    spacing = (official.pos - torch.roll(official.pos, 1, dims=0)).norm(dim=-1)
    assert float(spacing.max() - spacing.min()) < 1e-4
    assert float(spacing.mean()) == pytest.approx(official.spacing_m, abs=1e-5)


def test_reported_spacing_is_the_spacing_actually_used(official):
    """Regression: the sample count is a whole number, so the true spacing is
    ``length / num_samples`` and not what was requested. Reporting the requested
    value instead drifts index lookups by up to half a sample over a lap and
    shows up as a 8 mm discontinuity at the start/finish seam."""
    assert official.spacing_m == pytest.approx(
        official.length / official.num_samples, rel=1e-12
    )
    # 1e-4 rather than tighter: s is float32, whose resolution at 50 m is
    # already 4e-6.
    assert float(official.s[-1]) == pytest.approx(
        official.length - official.spacing_m, abs=1e-4
    )


def test_spacing_choice_does_not_move_the_geometry(circle_track):
    """A finer resample must describe the same track, not a different one."""
    coarse = TrackGeometry.from_track(circle_track, spacing_m=0.05)
    fine = TrackGeometry.from_track(circle_track, spacing_m=0.01)
    assert coarse.length == pytest.approx(fine.length, abs=1e-3)
    assert float(coarse.kappa.mean()) == pytest.approx(
        float(fine.kappa.mean()), abs=1e-4
    )


# ══════════════════════════════════════════════════════════════════════════
#  Vectorization and input handling
# ══════════════════════════════════════════════════════════════════════════


def test_project_is_vectorized_over_many_cars(official):
    """It runs for thousands of cars every step, so shape handling matters."""
    s = torch.rand(2048) * official.length
    n = (torch.rand(2048) - 0.5) * 0.3
    recovered_s, recovered_n = official.project(official.point_at(s, n))
    assert recovered_s.shape == (2048,)
    error = torch.remainder(recovered_s - s + 0.5 * official.length, official.length)
    error -= 0.5 * official.length
    assert float(error.abs().max()) < 1e-2
    assert float((recovered_n - n).abs().max()) < 1e-2


def test_rejects_malformed_input():
    with pytest.raises(ValueError, match=r"\[P, 2\]"):
        TrackGeometry(torch.zeros(10, 3))
    with pytest.raises(ValueError, match="at least 8"):
        TrackGeometry(torch.zeros(4, 2))


def test_nothing_is_hardcoded_to_the_official_track(circle_track):
    """The deliverable is a recipe that retrains on a new layout."""
    geometry = TrackGeometry.from_track(circle_track)
    assert geometry.validate() == []
    assert geometry.length == pytest.approx(2.0 * math.pi * RADIUS, abs=1e-3)
