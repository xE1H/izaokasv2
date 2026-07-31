"""Tests for the locked lap timer.

Runs on a synthetic circular track with scripted car positions, so it needs
neither Isaac Sim nor a GPU.

    .venv/bin/python -m pytest tests/test_timing.py -q
"""

from __future__ import annotations

import csv
import math

import pytest
import torch

from lituanicax_sdk.timing import LapTimer
from lituanicax_sdk.track import Track, TrackCfg

RADIUS = 5.0
NUM_POINTS = 720
STEP_DT = 1.0 / 60.0


@pytest.fixture
def circle_track(tmp_path):
    """A 5 m radius circle, start/finish at angle 0, i.e. at (5, 0)."""
    path = tmp_path / "circle.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for i in range(NUM_POINTS):
            theta = 2.0 * math.pi * i / NUM_POINTS
            writer.writerow([RADIUS * math.cos(theta), RADIUS * math.sin(theta), 0.0])

    cfg = TrackCfg(
        name="circle",
        walls_usd="unused.usd",
        centerline_csv=str(path),
        centerline_scale=(1.0, 1.0),
        spawn_points=[(RADIUS, 0.0, 90.0)],
        start_finish_index=0,
        lap_gate_window_m=2.0,
    )
    return Track(cfg, device="cpu")


def _timer(track, num_envs=1):
    return LapTimer(track, num_envs=num_envs, step_dt=STEP_DT, device="cpu")


def _drive(timer, angles, radius=RADIUS, start_step=0):
    """Drive one car through a list of angles; return each step's lap flag."""
    flags = []
    for i, theta in enumerate(angles):
        pos = torch.tensor([[radius * math.cos(theta), radius * math.sin(theta)]])
        idx, _ = timer.track.nearest(pos)
        step = torch.tensor([start_step + i])
        flags.append(bool(timer.update(pos, idx, step)[0]))
    return flags


def _laps(num_laps, steps_per_lap, offset=0.0):
    """Angles for driving ``num_laps`` at ``steps_per_lap``.

    A few extra steps are appended: a crossing is detected on the step *after*
    the car passes the plane, so stopping exactly on a whole lap would cut off
    the very crossing being tested.
    """
    total = int(num_laps * steps_per_lap) + 4
    return [offset + 2.0 * math.pi * i / steps_per_lap for i in range(total + 1)]


# ── The track itself ──────────────────────────────────────────────────────


def test_track_geometry(circle_track):
    assert circle_track.num_points == NUM_POINTS
    assert circle_track.track_length == pytest.approx(2 * math.pi * RADIUS, rel=1e-3)
    # Curvature of a circle is 1/r everywhere.
    assert float(circle_track.curvature.median()) == pytest.approx(1 / RADIUS, rel=1e-2)
    assert circle_track.validate() == []


def test_loop_distance_from_gate_peaks_halfway(circle_track):
    half = circle_track.track_length / 2
    at_gate = circle_track.loop_distance_from_gate(torch.tensor([0]))
    opposite = circle_track.loop_distance_from_gate(torch.tensor([NUM_POINTS // 2]))
    assert float(at_gate) == pytest.approx(0.0, abs=1e-3)
    assert float(opposite) == pytest.approx(half, rel=1e-3)


# ── Lap counting ──────────────────────────────────────────────────────────


def test_first_crossing_is_an_out_lap_and_is_not_timed(circle_track):
    timer = _timer(circle_track)
    # Start a quarter of the way round, drive one full circle back past the
    # line. That crossing completes only a partial loop from the spawn.
    flags = _drive(timer, _laps(1.0, 600, offset=math.pi / 2))
    assert not any(flags), "the out-lap must not be timed"
    assert int(timer.lap_count[0]) == 0
    assert float(timer.best_lap_time_s) == 0.0


def test_second_crossing_is_a_timed_lap(circle_track):
    timer = _timer(circle_track)
    steps_per_lap = 600
    flags = _drive(timer, _laps(2.2, steps_per_lap, offset=math.pi / 2))
    assert sum(flags) == 1, "exactly one timed lap after the out-lap"
    assert int(timer.lap_count[0]) == 1

    expected = steps_per_lap * STEP_DT
    assert float(timer.best_lap_time_s) == pytest.approx(expected, rel=0.02)
    assert float(timer.last_lap_time_s[0]) == pytest.approx(expected, rel=0.02)


def test_starting_on_the_line_still_needs_an_out_lap(circle_track):
    timer = _timer(circle_track)
    flags = _drive(timer, _laps(3.0, 400, offset=0.0))
    # Crossings at 1, 2 and 3 laps; the first is the out-lap.
    assert sum(flags) == 2
    assert float(timer.best_lap_time_s) == pytest.approx(400 * STEP_DT, rel=0.02)


def test_lap_times_are_measured_between_crossings_not_from_spawn(circle_track):
    """A slow out-lap must not contaminate the first timed lap."""
    timer = _timer(circle_track)
    # Half a circle at 1200 steps/lap (slow), then two laps at 300 (fast).
    slow = [math.pi + 2.0 * math.pi * i / 1200 for i in range(600)]
    fast_start = slow[-1]
    fast = [fast_start + 2.0 * math.pi * i / 300 for i in range(1, 601)]
    _drive(timer, slow + fast)
    assert int(timer.lap_count[0]) >= 1
    assert float(timer.best_lap_time_s) == pytest.approx(300 * STEP_DT, rel=0.05)


def test_both_directions_are_timed(circle_track):
    forward = _timer(circle_track)
    _drive(forward, _laps(2.2, 600, offset=math.pi / 2))

    backward = _timer(circle_track)
    _drive(backward, [-a for a in _laps(2.2, 600, offset=math.pi / 2)])

    assert int(forward.lap_count[0]) == 1
    assert int(backward.lap_count[0]) == 1
    assert float(forward.best_lap_time_s) == pytest.approx(
        float(backward.best_lap_time_s), rel=0.02
    )


# ── The anti-cheat rules ──────────────────────────────────────────────────


def test_reversing_over_the_line_is_not_a_lap(circle_track):
    """The original implementation counted this; the travel rule blocks it."""
    timer = _timer(circle_track)
    # Establish a real lap first, so the clock is running and armed.
    _drive(timer, _laps(2.0, 400, offset=0.0))
    baseline = int(timer.lap_count[0])

    # Now shuffle back and forth over the line: out ~0.4 m and back, repeatedly.
    arc = 0.45 / RADIUS
    angles = []
    for _ in range(20):
        angles += [arc * i / 10 for i in range(11)]
        angles += [arc * (10 - i) / 10 for i in range(11)]
        angles += [-arc * i / 10 for i in range(11)]
        angles += [-arc * (10 - i) / 10 for i in range(11)]
    flags = _drive(timer, angles, start_step=5000)

    assert not any(flags), "crossing the line without going round is not a lap"
    assert int(timer.lap_count[0]) == baseline


def test_a_wide_racing_line_still_trips_the_gate(circle_track):
    """The reason the gate is measured along the track, not as a radius.

    A racing car does not drive the centerline. An earlier version required the
    car to pass within 0.3 m of the start/finish *point*, and a policy taking a
    wide line through the start/finish straight completed laps that were never
    timed — 90 seconds of driving and no lap times at all.
    """
    timer = _timer(circle_track)
    # Drive the whole lap 0.6 m outside the centerline: twice the old radius.
    flags = _drive(timer, _laps(2.2, 600, offset=math.pi / 2), radius=RADIUS + 0.6)
    assert sum(flags) == 1, "a wide line still crosses the start/finish line"
    assert int(timer.lap_count[0]) == 1


def test_a_line_that_cuts_inside_also_trips_the_gate(circle_track):
    timer = _timer(circle_track)
    flags = _drive(timer, _laps(2.2, 600, offset=math.pi / 2), radius=RADIUS - 0.6)
    assert sum(flags) == 1
    assert int(timer.lap_count[0]) == 1


def test_sitting_on_the_line_is_not_a_lap(circle_track):
    timer = _timer(circle_track)
    flags = _drive(timer, [0.0] * 500)
    assert not any(flags)
    assert int(timer.lap_count[0]) == 0


def test_a_wall_touch_invalidates_the_lap_in_progress(circle_track):
    timer = _timer(circle_track)
    angles = _laps(2.2, 600, offset=math.pi / 2)

    flags = []
    for i, theta in enumerate(angles):
        pos = torch.tensor([[RADIUS * math.cos(theta), RADIUS * math.sin(theta)]])
        idx, _ = timer.track.nearest(pos)
        if i == len(angles) // 2:  # scrape a wall midway through the timed lap
            timer.invalidate(torch.tensor([True]))
        flags.append(bool(timer.update(pos, idx, torch.tensor([i]))[0]))

    assert not any(flags), "a lap with a wall touch in it must not count"
    assert int(timer.lap_count[0]) == 0


def test_invalidation_clears_at_the_next_crossing(circle_track):
    timer = _timer(circle_track)
    angles = _laps(3.2, 400, offset=math.pi / 2)

    laps = 0
    for i, theta in enumerate(angles):
        pos = torch.tensor([[RADIUS * math.cos(theta), RADIUS * math.sin(theta)]])
        idx, _ = timer.track.nearest(pos)
        if i == 200:  # spoil the out-lap only
            timer.invalidate(torch.tensor([True]))
        laps += int(timer.update(pos, idx, torch.tensor([i]))[0])

    assert laps >= 1, "later laps must count once the spoiled one is over"


# ── Resets and multiple cars ──────────────────────────────────────────────


def test_reset_clears_the_lap_in_progress(circle_track):
    timer = _timer(circle_track)
    _drive(timer, _laps(1.5, 400, offset=0.0))
    timer.reset(torch.tensor([0]))
    assert int(timer.lap_count[0]) == 0
    assert float(timer.last_lap_time_s[0]) == 0.0
    # A respawned car needs a fresh out-lap before anything is timed.
    flags = _drive(timer, _laps(1.0, 400, offset=math.pi / 2), start_step=0)
    assert not any(flags)


def test_the_summary_counts_laps_across_respawns(circle_track):
    """A report must not lose the laps of cars that have since crashed.

    lap_count is per-stint and is cleared on respawn, so summing it at the end
    of a run counts only whichever cars happened to be mid-episode — an earlier
    version reported 14 laps for a run that had actually driven 76.
    """
    timer = _timer(circle_track)
    _drive(timer, _laps(3.0, 400, offset=0.0))
    driven = timer.summary()["laps_completed"]
    assert driven >= 2

    timer.reset(torch.tensor([0]))
    assert int(timer.lap_count[0]) == 0, "the stint counter resets"
    assert timer.summary()["laps_completed"] == driven, "the run total does not"
    assert timer.summary()["cars_with_a_lap"] == 1
    assert timer.summary()["best_lap_time_s"] > 0.0


def test_run_best_survives_a_reset(circle_track):
    timer = _timer(circle_track)
    _drive(timer, _laps(2.0, 400, offset=0.0))
    best = float(timer.best_lap_time_s)
    assert best > 0.0
    timer.reset(torch.tensor([0]))
    assert float(timer.best_lap_time_s) == best


def test_cars_are_timed_independently(circle_track):
    timer = _timer(circle_track, num_envs=2)
    # Car 0 laps in 300 steps, car 1 in 600.
    for i in range(1801):
        a0 = 2.0 * math.pi * i / 300
        a1 = 2.0 * math.pi * i / 600
        pos = torch.tensor(
            [
                [RADIUS * math.cos(a0), RADIUS * math.sin(a0)],
                [RADIUS * math.cos(a1), RADIUS * math.sin(a1)],
            ]
        )
        idx, _ = timer.track.nearest(pos)
        timer.update(pos, idx, torch.tensor([i, i]))

    assert int(timer.lap_count[0]) > int(timer.lap_count[1])
    assert float(timer.best_lap_time_s_per_env[0]) == pytest.approx(
        300 * STEP_DT, rel=0.05
    )
    assert float(timer.best_lap_time_s_per_env[1]) == pytest.approx(
        600 * STEP_DT, rel=0.05
    )
    # The run-wide best is the quicker of the two.
    assert float(timer.best_lap_time_s) == pytest.approx(300 * STEP_DT, rel=0.05)


def test_log_dict_omits_best_until_a_lap_exists(circle_track):
    timer = _timer(circle_track)
    assert "Lap/best_lap_time_s" not in timer.log_dict()
    _drive(timer, _laps(2.0, 400, offset=0.0))
    assert "Lap/best_lap_time_s" in timer.log_dict()
