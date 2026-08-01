"""Tests for building and validating your own track.

Adding a track is the extension teams are most likely to get subtly wrong — a
centerline that is not quite a closed loop, or unevenly spaced, produces
curvature and lookahead observations that are wrong rather than broken. These
cover what ``Track.validate()`` is supposed to catch.

    .venv/bin/python -m pytest tests/test_track.py -q
"""

from __future__ import annotations

import csv
import math

import pytest
import torch

from lituanicax_sdk.track import Track, TrackCfg


def write_csv(path, points):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for x, y in points:
            writer.writerow([x, y, 0.0])
    return str(path)


def circle_points(n=720, radius=5.0, arc=2 * math.pi):
    return [
        (radius * math.cos(arc * i / n), radius * math.sin(arc * i / n))
        for i in range(n)
    ]


def make_track(path, **kwargs):
    cfg = TrackCfg(
        name=kwargs.pop("name", "test"),
        walls_usd="unused.usd",
        centerline_csv=path,
        centerline_scale=(1.0, 1.0),
        **kwargs,
    )
    return Track(cfg, device="cpu")


# ── Loading ───────────────────────────────────────────────────────────────


def test_a_good_track_has_no_problems(tmp_path):
    track = make_track(write_csv(tmp_path / "c.csv", circle_points()))
    assert track.validate() == []
    assert track.num_points == 720
    assert track.track_length == pytest.approx(2 * math.pi * 5.0, rel=1e-3)


def test_centerline_scale_is_applied(tmp_path):
    path = write_csv(tmp_path / "c.csv", circle_points(radius=1.0))
    cfg = TrackCfg(
        name="scaled",
        walls_usd="w.usd",
        centerline_csv=path,
        centerline_scale=(2.0, 2.0),
    )
    track = Track(cfg, device="cpu")
    # A unit circle scaled by 2 has circumference 4π.
    assert track.track_length == pytest.approx(4 * math.pi, rel=1e-3)


def wall_ring(track, radius, n=360):
    """A closed polyline of ``n`` segments at ``radius``, as wall segments."""
    points = torch.tensor(
        [
            [
                radius * math.cos(2 * math.pi * i / n),
                radius * math.sin(2 * math.pi * i / n),
            ]
            for i in range(n + 1)
        ]
    )
    return points[:-1], points[1:]


def add_walls(track, inner=4.65, outer=5.35):
    """Give a circle track two rails, 0.7 m apart, centred on the centerline."""
    inner_a, inner_b = wall_ring(track, inner)
    outer_a, outer_b = wall_ring(track, outer)
    track.wall_seg_a = torch.cat([inner_a, outer_a])
    track.wall_seg_b = torch.cat([inner_b, outer_b])
    return track


def test_a_centerline_between_the_walls_is_fine(tmp_path):
    track = add_walls(make_track(write_csv(tmp_path / "c.csv", circle_points())))
    assert track.validate() == []


def test_a_centerline_that_misses_the_walls_is_reported(tmp_path):
    """The failure that shipped: one transform for the walls, another for the line.

    The official centerline was scaled 15% larger than the mesh it describes, so
    it ran through the walls for a quarter of the lap and every track-relative
    observation was measured against a line that was not on the track. Nothing
    caught it, because nothing compared the two.
    """
    path = write_csv(tmp_path / "c.csv", circle_points())
    cfg = TrackCfg(
        name="mis_scaled",
        walls_usd="unused.usd",
        centerline_csv=path,
        centerline_scale=(1.15, 1.15),  # walls stay where they are
    )
    track = add_walls(Track(cfg, device="cpu"))

    problems = track.validate()
    assert any("wall on one side only" in p for p in problems), problems
    assert any("centerline_scale" in p for p in problems), problems


def test_a_missing_centerline_says_which_track(tmp_path):
    with pytest.raises(FileNotFoundError, match="my_track"):
        make_track(str(tmp_path / "nope.csv"), name="my_track")


def test_a_nearly_empty_centerline_is_rejected(tmp_path):
    path = write_csv(tmp_path / "c.csv", [(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(ValueError, match="at least 4"):
        make_track(path)


def test_a_header_row_is_skipped(tmp_path):
    path = tmp_path / "c.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z"])
        for x, y in circle_points(n=100):
            writer.writerow([x, y, 0.0])
    assert make_track(str(path)).num_points == 100


# ── validate() ────────────────────────────────────────────────────────────


def test_an_open_loop_is_reported(tmp_path):
    """Three quarters of a circle: the ends do not meet."""
    path = write_csv(tmp_path / "c.csv", circle_points(arc=1.5 * math.pi))
    problems = make_track(path).validate()
    assert any("closed loop" in p for p in problems), problems


def test_uneven_spacing_is_reported(tmp_path):
    points = circle_points(n=200)
    points.insert(100, (40.0, 40.0))  # one wild point
    problems = make_track(write_csv(tmp_path / "c.csv", points)).validate()
    assert any("spacing" in p for p in problems), problems


def test_too_few_points_is_reported(tmp_path):
    problems = make_track(write_csv(tmp_path / "c.csv", circle_points(n=20))).validate()
    assert any("centerline points" in p for p in problems), problems


def test_a_track_says_nothing_about_where_cars_start(tmp_path):
    """Spawning is the team's business, so it is not a property of the track."""
    track = make_track(write_csv(tmp_path / "c.csv", circle_points()))
    assert not hasattr(track.cfg, "spawn_points")
    assert track.validate() == []


# ── Derived quantities ────────────────────────────────────────────────────


def test_curvature_of_a_circle_is_one_over_r(tmp_path):
    for radius in (2.0, 5.0, 10.0):
        track = make_track(
            write_csv(tmp_path / f"c{radius}.csv", circle_points(radius=radius))
        )
        assert float(track.curvature.median()) == pytest.approx(1 / radius, rel=0.02)


def test_a_track_with_no_sharp_corner_still_reports_one(tmp_path):
    """The 'next corner' observations need something to point at."""
    track = make_track(write_csv(tmp_path / "c.csv", circle_points(radius=20.0)))
    assert track.corner_idx.numel() >= 1


def test_tangents_are_unit_length(tmp_path):
    track = make_track(write_csv(tmp_path / "c.csv", circle_points()))
    assert torch.allclose(track.tangents.norm(dim=1), torch.ones(720), atol=1e-4)


def test_the_start_finish_line_follows_the_index(tmp_path):
    path = write_csv(tmp_path / "c.csv", circle_points())
    at_zero = make_track(path, start_finish_index=0)
    at_quarter = make_track(path, start_finish_index=180)

    assert float(at_zero.gate_point[0]) == pytest.approx(5.0, abs=1e-3)
    assert float(at_quarter.gate_point[1]) == pytest.approx(5.0, abs=1e-3)
    # Both still measure the same lap distance.
    assert at_zero.track_length == pytest.approx(at_quarter.track_length)


def test_walls_must_be_loaded_before_they_can_be_queried(tmp_path):
    track = make_track(write_csv(tmp_path / "c.csv", circle_points()))
    with pytest.raises(RuntimeError, match="walls have not been loaded"):
        track.distance_to_walls(torch.zeros(1, 2))
