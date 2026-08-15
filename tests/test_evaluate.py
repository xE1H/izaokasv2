"""Tests for :mod:`tools.evaluate`.

:func:`tools.evaluate.evaluate` needs a ``RaceEnv``, which needs Isaac Sim. But
what it is most likely to get *wrong* is bookkeeping — when a lap counts, when an
attempt is over, which cars are still being driven — and none of that needs a
simulator. So it is driven here against a scripted fake environment whose cars
move round a circle at a known speed, where the right lap time is arithmetic.

That is worth more than it sounds: the real benchmark cannot be used as a
reference until a policy checkpoint exists in Phase 3, so until then these tests
plus a line-by-line match against ``benchmark.py`` are the whole of the evidence
that the scoring is right.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.evaluate import (
    AGENTS,
    CONSISTENCY_BONUS_S,
    CRASH_CEILING,
    CRASH_FLOOR,
    NO_LAP,
    Attempt,
    RepeatedStarts,
    evaluate,
    objective,
    official_start_offsets,
    summarize,
)

RADIUS = 5.0


# ══════════════════════════════════════════════════════════════════════════
#  A fake environment that drives cars round a circle
# ══════════════════════════════════════════════════════════════════════════


class FakeEnv:
    """Enough of a ``HarnessEnv`` to score against, with scripted motion.

    Cars run counterclockwise round the circle at their own constant speed. A car
    listed in ``crash_at`` has its episode terminated on that step, and a retired
    car stops moving — which is what lets the tests check that ``retire()`` is
    really being honoured rather than merely called.
    """

    def __init__(
        self, track, speeds, *, crash_at=None, max_steps=400, step_dt=1.0 / 30.0
    ):
        self.track = track
        self.device = torch.device("cpu")
        self.num_envs = len(speeds)
        self.step_dt = step_dt
        self.max_episode_length = max_steps
        self.cfg = SimpleNamespace(enforce_official_rules=True)

        self.speeds = torch.tensor(speeds, dtype=torch.float32)
        self.crash_at = dict(crash_at or {})
        self.theta = torch.zeros(self.num_envs)
        self.retired = torch.zeros(self.num_envs, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool)
        self.robot = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=torch.zeros(self.num_envs, 3))
        )
        self.latest_car = None
        self.retire_calls = 0
        self._write()

    def _write(self):
        self.robot.data.root_pos_w[:, 0] = RADIUS * torch.cos(self.theta)
        self.robot.data.root_pos_w[:, 1] = RADIUS * torch.sin(self.theta)

    def reset(self, seed=None, options=None):
        self.theta[:] = 0.0
        self.retired[:] = False
        self.episode_length_buf[:] = 0
        self._write()
        return None, {}

    def step(self, actions):
        moving = ~self.retired
        self.theta = torch.where(
            moving, self.theta + self.speeds / RADIUS * self.step_dt, self.theta
        )
        self.episode_length_buf = torch.where(
            moving, self.episode_length_buf + 1, self.episode_length_buf
        )
        self._write()

        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        for env_id, step in self.crash_at.items():
            if (
                int(self.episode_length_buf[env_id]) == step
                and not self.retired[env_id]
            ):
                terminated[env_id] = True
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.reset_terminated = terminated
        self.reset_time_outs = truncated & ~terminated
        return None, None, terminated, truncated, {}

    def retire(self, mask):
        self.retire_calls += 1
        self.retired |= mask


def nothing(car):
    return torch.zeros(1, 2)


@pytest.fixture
def circle(circle_track):
    return circle_track


def expected_lap(speed: float, length: float) -> float:
    return length / speed


# ══════════════════════════════════════════════════════════════════════════
#  Lap detection
# ══════════════════════════════════════════════════════════════════════════


def test_a_full_lap_is_timed_correctly(circle):
    """One car, one circle, one arithmetic answer."""
    env = FakeEnv(circle, [5.0])
    attempts = evaluate(env, nothing)
    assert len(attempts) == 1
    assert attempts[0].valid
    assert attempts[0].outcome == "lap"
    assert attempts[0].lap_time_s == pytest.approx(
        expected_lap(5.0, circle.track_length), rel=0.02
    )


def test_faster_cars_get_shorter_times(circle):
    env = FakeEnv(circle, [3.0, 5.0, 7.0])
    attempts = evaluate(env, nothing)
    assert all(a.valid for a in attempts)
    times = [a.lap_time_s for a in attempts]
    assert times == sorted(times, reverse=True)
    for speed, attempt in zip([3.0, 5.0, 7.0], attempts):
        assert attempt.lap_time_s == pytest.approx(
            expected_lap(speed, circle.track_length), rel=0.02
        )


def test_the_clock_starts_at_the_spawn_not_the_track_gate(circle):
    """An ``AttemptTimer``, not a ``LapTimer``: no untimed out-lap.

    On this circle the spawn and the track's start/finish line coincide, so the
    check that bites is that the time equals a *whole* lap rather than a lap plus
    the drive to the line.
    """
    env = FakeEnv(circle, [5.0])
    attempt = evaluate(env, nothing)[0]
    assert attempt.lap_time_s == pytest.approx(circle.track_length / 5.0, rel=0.02)


def test_a_crashed_car_records_no_lap(circle):
    env = FakeEnv(circle, [5.0], crash_at={0: 40})
    attempt = evaluate(env, nothing)[0]
    assert not attempt.valid
    assert attempt.lap_time_s is None
    assert attempt.outcome == "crashed"
    assert attempt.score == NO_LAP


def test_a_car_that_never_finishes_times_out(circle):
    """Slow enough that the attempt window closes first."""
    env = FakeEnv(circle, [0.5], max_steps=100)
    attempt = evaluate(env, nothing)[0]
    assert not attempt.valid
    assert attempt.outcome == "out of time"


def test_mixed_outcomes_are_attributed_to_the_right_cars(circle):
    env = FakeEnv(circle, [5.0, 5.0, 0.4, 5.0], crash_at={1: 30}, max_steps=300)
    attempts = evaluate(env, nothing)
    assert [a.outcome for a in attempts] == ["lap", "crashed", "out of time", "lap"]


# ══════════════════════════════════════════════════════════════════════════
#  One attempt per car
# ══════════════════════════════════════════════════════════════════════════


def test_a_finished_car_is_retired_and_stops_driving(circle):
    """Without this the quick cars keep lapping while the slow ones are still out,
    and the second lap can overwrite the first."""
    env = FakeEnv(circle, [7.0, 1.2], max_steps=400)
    attempts = evaluate(env, nothing)
    assert env.retire_calls > 0
    assert bool(env.retired[0]), "the car that finished should have been retired"
    # It banked one lap at its real speed, not a later, different one.
    assert attempts[0].lap_time_s == pytest.approx(
        expected_lap(7.0, circle.track_length), rel=0.02
    )
    # And it stopped where it finished: a little past one full turn, not two.
    assert float(env.theta[0]) < 2.5 * math.pi


def test_scoring_stops_once_every_car_has_settled(circle):
    """The loop must break early, or every generation costs a full episode."""
    env = FakeEnv(circle, [7.0, 7.0], max_steps=2000)
    evaluate(env, nothing)
    assert int(env.episode_length_buf.max()) < 400


def test_a_lap_and_a_termination_on_the_same_step_is_not_a_lap(circle):
    """The dones are read before the clock, because a terminated car may have been
    teleported back to its spawn point inside ``step()`` — and that jump crosses
    its own gate, scoring a lap nobody drove."""
    env = FakeEnv(circle, [5.0])
    # Crash on the step the lap would close.
    steps = int(round(circle.track_length / 5.0 / env.step_dt)) + 1
    env.crash_at = {0: steps}
    attempt = evaluate(env, nothing)[0]
    assert attempt.outcome == "crashed"
    assert attempt.lap_time_s is None


# ══════════════════════════════════════════════════════════════════════════
#  Progress, for the crash branch of the objective
# ══════════════════════════════════════════════════════════════════════════


def test_progress_measures_how_far_round_a_failed_car_got(circle):
    """The crash branch needs a gradient, so this has to be monotone past halfway
    — which neither the lap timer's short-way travel nor a raw modular distance
    manages."""
    quarter = FakeEnv(circle, [5.0], crash_at={0: 47})
    three_quarters = FakeEnv(circle, [5.0], crash_at={0: 141})
    near = FakeEnv(circle, [5.0], crash_at={0: 180})

    a = evaluate(quarter, nothing)[0].progress
    b = evaluate(three_quarters, nothing)[0].progress
    c = evaluate(near, nothing)[0].progress
    assert a == pytest.approx(0.25, abs=0.05)
    assert b == pytest.approx(0.75, abs=0.05)
    assert a < b < c <= 1.05


def test_progress_does_not_credit_a_car_that_reverses_over_the_line(circle):
    """A raw modular distance reads ~1.0 for a car that merely backs up."""
    env = FakeEnv(circle, [-1.0], max_steps=60)
    attempt = evaluate(env, nothing)[0]
    assert attempt.progress < 0.1


# ══════════════════════════════════════════════════════════════════════════
#  The search objective
# ══════════════════════════════════════════════════════════════════════════


def lap(time_s, progress=1.0):
    return Attempt(valid=True, lap_time_s=time_s, progress=progress, outcome="lap")


def fail(progress, outcome="crashed"):
    return Attempt(valid=False, lap_time_s=None, progress=progress, outcome=outcome)


def test_a_valid_lap_scores_its_best_time():
    attempts = [lap(14.0), lap(13.2), lap(15.9)] + [fail(0.5)] * 7
    score, detail = objective(attempts, min_completions=3, num_starts=10)
    assert score == pytest.approx(13.2, abs=CONSISTENCY_BONUS_S)
    assert detail["branch"] == "lap"
    assert detail["completions"] == 3


def test_one_golden_run_is_scored_on_that_run():
    """The leaderboard takes the fastest of ten attempts and ignores the rest.

    A candidate that crashes nine times and puts in one blinding lap is, for
    scoring purposes, a fast car. An objective that refuses to score it is
    optimizing a different competition than the one being entered.
    """
    golden = [lap(13.0)] + [fail(0.4)] * 9
    score, detail = objective(golden, min_completions=1, num_starts=10)
    assert detail["branch"] == "lap"
    assert score == pytest.approx(13.0, abs=CONSISTENCY_BONUS_S)


def test_lapping_candidates_are_separated_by_pace():
    """The bug this replaced: every candidate that lapped at all scored 20.000.

    With the completion floor at 8, a single completed lap put the candidate on
    the crash branch at exactly full progress — so a 20.8 s lap and a 24.1 s lap
    were worth precisely the same, and the search spent five generations unable
    to tell them apart.
    """
    quick = objective([lap(20.8)] + [fail(0.4)] * 9, min_completions=1, num_starts=10)
    slow = objective([lap(24.1)] + [fail(0.4)] * 9, min_completions=1, num_starts=10)
    assert quick[0] < slow[0] - 3.0


def test_one_golden_run_is_worth_as_much_as_a_hundred():
    """The scoring rule this competition actually uses.

    ``--agents`` has no cap and verification replays whatever count was
    submitted, so the score is the fastest of however many attempts the team
    chooses to run. Finishing often buys nothing, and a preference for it would
    make the search discard precisely the fast, marginal candidates that win.
    """
    every = objective([lap(15.0)] * 10, min_completions=1, num_starts=10)[0]
    once = objective([lap(15.0)] + [fail(0.5)] * 9, min_completions=1, num_starts=10)[0]
    assert every == once, "finishing more often must not change the score"

    faster_but_fragile = objective(
        [lap(14.0)] + [fail(0.5)] * 9, min_completions=1, num_starts=10
    )[0]
    assert faster_but_fragile < every, "pace is the only thing that counts"


def test_a_valid_lap_always_beats_a_crash():
    """The property the crash floor exists for: the search must never be tempted
    to trade a finished lap for a spectacular failure."""
    slowest = objective([lap(19.9)] * 10, min_completions=1, num_starts=10)[0]
    best_crash = objective([fail(1.0)] * 10, min_completions=1, num_starts=10)[0]
    assert slowest < CRASH_FLOOR <= best_crash


def test_the_crash_branch_rewards_getting_further():
    near = objective([fail(0.9)], min_completions=1, num_starts=1)[0]
    far = objective([fail(0.3)], min_completions=1, num_starts=1)[0]
    nowhere = objective([fail(0.0)], min_completions=1, num_starts=1)[0]
    assert near < far < nowhere
    assert nowhere == pytest.approx(CRASH_CEILING)


def test_the_completion_floor_still_works_when_asked_for():
    """Not the default any more, but Phase 3 may want a repeatable teacher."""
    fragile = [lap(12.0), lap(12.4), lap(12.9)] + [fail(0.6)] * 7
    score, detail = objective(fragile, min_completions=8, num_starts=10)
    assert detail["branch"] == "crash"
    assert score >= CRASH_FLOOR

    reliable = [lap(13.5)] * 8 + [fail(0.6)] * 2
    score, detail = objective(reliable, min_completions=8, num_starts=10)
    assert detail["branch"] == "lap"
    assert score == pytest.approx(13.5, abs=CONSISTENCY_BONUS_S)


def test_a_slow_valid_lap_beats_a_voided_one():
    """The inversion the old constants had, and it was the wrong way round.

    A car can get all the way round and still have the lap voided by touching a
    wall. At a crash floor of 20 that scored 20.0, which beat a perfectly valid
    22 s lap — the search was being offered a better score for crashing on the
    last corner than for finishing. Real laps here are 20-24 s, so the floor has
    to sit above anything the attempt window can produce.
    """
    valid_but_slow = objective([lap(24.0)], min_completions=1, num_starts=1)[0]
    voided_at_the_line = objective([fail(1.0)], min_completions=1, num_starts=1)[0]
    assert valid_but_slow < voided_at_the_line


def test_progress_beyond_one_cannot_undercut_a_lap():
    """A car can get round and still have its lap voided by a wall touch."""
    score, _ = objective([fail(1.4)], min_completions=1, num_starts=1)[0], None
    assert score >= CRASH_FLOOR


def test_summarize_reads_sensibly():
    assert "best 13.200 s" in summarize([lap(14.0), lap(13.2)])
    assert "no lap" in summarize([fail(0.42)])
    assert "42%" in summarize([fail(0.42)])


# ══════════════════════════════════════════════════════════════════════════
#  Repeated starts
# ══════════════════════════════════════════════════════════════════════════


def test_official_start_offsets_match_the_benchmark_spread():
    offsets = official_start_offsets()
    assert offsets.shape == (AGENTS,)
    assert np.abs(offsets).max() <= math.radians(5.0)


def test_measured_headings_win_over_the_even_spacing(tmp_path, monkeypatch):
    """When the benchmark's own ten are known, they are what gets optimized.

    The fallback spreads headings evenly, which is the right answer when the
    real ones are unknown and strictly worse when they are not: the benchmark's
    actual draw is skewed, six of ten above +2.9 degrees, so evenly spaced
    offsets put half their effort where it never goes.
    """
    from tools import evaluate

    real = [0.01, -0.02, 0.03, 0.04]
    path = tmp_path / "headings.json"
    path.write_text(json.dumps({"offsets_rad": real}))
    monkeypatch.setattr(evaluate, "MEASURED_HEADINGS", path)
    assert np.allclose(evaluate.official_start_offsets(4), real)


def test_even_spacing_is_the_fallback(tmp_path, monkeypatch):
    """For a track or a seed nobody has measured yet."""
    from tools import evaluate

    monkeypatch.setattr(evaluate, "MEASURED_HEADINGS", tmp_path / "absent.json")
    offsets = evaluate.official_start_offsets()
    limit = math.radians(5.0)
    assert offsets.min() == pytest.approx(-limit)
    assert offsets.max() == pytest.approx(limit)
    gaps = np.diff(offsets)
    assert np.allclose(gaps, gaps[0]), "evenly spaced, so coverage has no holes"


def test_official_start_offsets_do_not_depend_on_luck():
    """Every candidate in every generation must face an identical set of starts,
    or CMA-ES is comparing luck rather than controllers."""
    assert np.allclose(official_start_offsets(seed=1), official_start_offsets(seed=2))


def test_zero_jitter_gives_identical_starts(tmp_path, monkeypatch):
    """Only meaningful for the fallback: measured headings are what they are."""
    from tools import evaluate

    monkeypatch.setattr(evaluate, "MEASURED_HEADINGS", tmp_path / "absent.json")
    assert np.allclose(evaluate.official_start_offsets(jitter_deg=0.0), 0.0)


def test_repeated_starts_gives_every_candidate_the_same_starts(circle):
    """The whole point: candidate A and candidate B must be compared on identical
    starts, or on a chaotic system the comparison is mostly noise."""
    offsets = official_start_offsets(count=4, seed=0)
    spawn = RepeatedStarts(offsets)
    candidates, starts = 3, 4
    spawn.setup(circle, candidates * starts, "cpu")

    env_ids = torch.arange(candidates * starts)
    _, yaw = spawn.sample(env_ids)

    base = float(spawn.pose()[2])
    for candidate in range(candidates):
        block = yaw[candidate * starts : (candidate + 1) * starts]
        assert torch.allclose(
            block - base,
            torch.tensor(offsets, dtype=torch.float32),
            atol=1e-6,
        ), candidate


def test_repeated_starts_puts_every_car_on_one_point(circle):
    spawn = RepeatedStarts(official_start_offsets(count=2))
    spawn.setup(circle, 6, "cpu")
    xy, _ = spawn.sample(torch.arange(6))
    assert torch.allclose(xy, xy[:1].expand_as(xy))


def test_repeated_starts_rejects_an_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        RepeatedStarts([])


def test_evaluate_refuses_to_score_without_official_rules(circle):
    env = FakeEnv(circle, [5.0])
    env.cfg.enforce_official_rules = False
    with pytest.raises(ValueError, match="official rules"):
        evaluate(env, nothing)
