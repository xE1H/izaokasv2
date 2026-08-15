"""Tests for :mod:`teacher.controller` and :mod:`teacher.params`.

The circle is the test track again: a constant-radius loop has an obvious right
answer for every term, so a sign error cannot hide.

The most important test here is :func:`test_controller_reads_only_car_state` — it
pins the promise the whole approach rests on. If the controller reads something a
policy cannot, the demonstrations it produces are unlearnable and Phase 3 fails
for a reason that will look like a network problem.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lituanicax_sdk.vehicle import VEHICLE
from teacher.controller import Controller, Reference, build_reference
from teacher.params import (
    DIMENSION,
    LINE_POINTS,
    SCALAR_BOUNDS,
    SPEED_POINTS,
    ControllerParams,
)
from tests.conftest import FakeRobot
from tools.geometry import TrackGeometry

RADIUS = 5.0
V_MAX = 6.7
WHEELBASE = 0.26


@pytest.fixture
def circle(circle_track):
    return TrackGeometry.from_track(circle_track, spacing_m=0.02, smooth_rms_mm=0.0)


@pytest.fixture
def controller(circle):
    params = ControllerParams()
    return Controller(
        circle,
        build_reference(circle, params, v_max=V_MAX),
        params,
        wheelbase_m=WHEELBASE,
        max_steer_rad=VEHICLE.max_steer_rad,
    )


def place(robot: FakeRobot, *, radius: float, theta: float, speed: float, yaw=None):
    """Put a car on the circle at angle ``theta``, moving along the track."""
    robot.data.root_pos_w[:, 0] = radius * math.cos(theta)
    robot.data.root_pos_w[:, 1] = radius * math.sin(theta)
    # Counterclockwise, so the tangent at theta is (-sin, cos).
    heading = theta + math.pi / 2 if yaw is None else yaw
    robot.set_yaw(heading)
    robot.data.root_lin_vel_b[:, 0] = speed
    robot.data.root_lin_vel_w[:, 0] = speed * math.cos(heading)
    robot.data.root_lin_vel_w[:, 1] = speed * math.sin(heading)
    return robot


# ══════════════════════════════════════════════════════════════════════════
#  The parameter vector
# ══════════════════════════════════════════════════════════════════════════


def test_dimension_matches_the_documented_groups():
    assert DIMENSION == LINE_POINTS + SPEED_POINTS + len(
        ControllerParams.scalar_names()
    )
    assert DIMENSION == 171


def test_a_checkpoint_from_an_older_resolution_still_loads():
    """Checkpoints outlive the resolution they were searched at.

    ``LINE_POINTS`` has gone 40 -> 120 once already, and a 40-knot file kept
    driving correctly the whole time, because the reference is built on a basis
    sized from the vector it is handed. So nothing complained until the search
    normalized one and got 91 numbers where it wanted 171 — a failure a long
    way from its cause, and one that cost three searches their startup.

    The refit has to preserve the *curve*, not the knots: the two bases have
    different bump widths, so copying values across would quietly reload a
    different driver than the one that was saved.
    """
    from tools.profile import periodic_basis

    from teacher.params import DIMENSION, LINE_POINTS

    knots = np.linspace(-0.15, 0.15, 40)
    params = ControllerParams.from_dict(
        {"line": list(knots), "speed_scale": [1.0] * SPEED_POINTS, "k_e": 2.5}
    )
    assert params.to_vector().size == DIMENSION
    assert params.to_normalized().size == DIMENSION
    assert params.k_e == 2.5, "scalars must survive the resolution change"

    before = periodic_basis(2000, 40) @ knots
    after = periodic_basis(2000, LINE_POINTS) @ params.line
    assert np.abs(before - after).max() < 1e-3, "the refit changed the line"


def test_the_line_cannot_be_searched_into_a_wall():
    """The corridor bound is the only thing standing between the search and a
    voided lap, and it has been widened once already.

    A car is retired the moment its centre comes within 0.15 m of a wall, and
    the walls sit 0.351 m either side, so 0.201 m is the whole legal budget.
    Widening past it would let CMA-ES chase lap times that no attempt can
    actually bank — the fastest candidate would be one that never finishes.
    """
    from lituanicax_sdk.rules import WALL_COLLISION_RADIUS_M

    from teacher.params import LINE_BOUND

    legal = 0.351 - WALL_COLLISION_RADIUS_M
    assert LINE_BOUND.high <= legal
    assert LINE_BOUND.low >= -legal
    assert LINE_BOUND.high == -LINE_BOUND.low, "the corridor must be symmetric"


def test_vector_round_trip():
    params = ControllerParams()
    assert np.allclose(
        ControllerParams.from_vector(params.to_vector()).to_vector(), params.to_vector()
    )


def test_normalized_round_trip_and_range():
    params = ControllerParams()
    normalized = params.to_normalized()
    assert normalized.shape == (DIMENSION,)
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))
    assert np.allclose(
        ControllerParams.from_normalized(normalized).to_vector(),
        params.to_vector(),
        atol=1e-9,
    )


def test_normalization_makes_every_parameter_the_same_scale():
    """Why it exists: the raw ranges span 0.36 m and 17 m/s^2, and CMA-ES would
    otherwise spend its early generations learning the scaling."""
    low, high = ControllerParams.bounds_vector()
    span = high - low
    assert span.min() > 0.0
    # In normalized space every span is exactly 1.
    assert np.allclose(
        (
            ControllerParams.from_vector(high).to_normalized()
            - ControllerParams.from_vector(low).to_normalized()
        ),
        1.0,
    )


def test_out_of_bounds_values_are_clipped_not_rejected():
    """CMA-ES samples a Gaussian and will step outside the box. Evaluating the
    nearest legal driver is honest; rejecting the sample biases the search."""
    params = ControllerParams.from_vector(np.full(DIMENSION, 1e6))
    low, high = ControllerParams.bounds_vector()
    assert np.all(params.to_vector() <= high + 1e-9)
    params = ControllerParams.from_vector(np.full(DIMENSION, -1e6))
    assert np.all(params.to_vector() >= low - 1e-9)


def test_wrong_length_vector_is_rejected():
    with pytest.raises(ValueError, match="expected 171"):
        ControllerParams.from_vector(np.zeros(12))


def test_params_survive_a_round_trip_through_json(tmp_path):
    params = ControllerParams(
        line=np.linspace(-0.1, 0.1, LINE_POINTS), a_lat_eff=9.5, k_e=3.25
    )
    path = params.save(tmp_path / "teacher.json")
    assert np.allclose(ControllerParams.load(path).to_vector(), params.to_vector())


# ══════════════════════════════════════════════════════════════════════════
#  The reference
# ══════════════════════════════════════════════════════════════════════════


def test_reference_has_one_entry_per_sample(circle):
    reference = build_reference(circle, ControllerParams(), v_max=V_MAX)
    for table in (
        reference.offset,
        reference.kappa,
        reference.speed,
        reference.speed_gradient,
    ):
        assert table.shape == (circle.num_samples,)
        assert torch.isfinite(table).all()


def test_reference_speed_respects_the_top_speed(circle):
    reference = build_reference(
        circle, ControllerParams(a_lat_eff=1000.0, a_accel_eff=100.0), v_max=V_MAX
    )
    assert float(reference.speed.max()) <= V_MAX + 1e-6


def test_reference_speed_on_a_circle_is_the_cornering_limit(circle):
    """A constant-radius loop has one target speed: sqrt(a_lat * R)."""
    reference = build_reference(circle, ControllerParams(a_lat_eff=8.0), v_max=V_MAX)
    assert float(reference.speed.mean()) == pytest.approx(
        math.sqrt(8.0 * RADIUS), rel=0.02
    )
    # Nothing is accelerating or braking, so the gradient is ~0 everywhere.
    #
    # Not *exactly* zero, and the tolerance had to grow when the profile started
    # modelling the motor's falloff. The discretized circle's curvature ripples
    # by a fraction of a percent, so the cornering ceiling does too; at 6.3 m/s
    # against a 6.7 m/s top speed the car has 6% of its standing acceleration
    # left and can no longer pull those dips back out. That is correct — a car
    # near its top speed really cannot re-accelerate — and 0.06 1/s is nothing
    # against the +-0.47 of alternating noise this assertion was written for.
    assert float(reference.speed_gradient.abs().max()) < 0.1


def test_speed_gradient_is_smooth_but_not_flattened():
    """The gradient must survive smoothing on a track that really does accelerate.

    Guards the other half of the fix: killing the ripple is easy, and killing the
    signal with it would silently disable the throttle feedforward.
    """
    from lituanicax_sdk.track import Track
    from lituanicax_sdk.tracks import OFFICIAL

    geometry = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
    reference = build_reference(
        geometry, ControllerParams(a_lat_eff=8.0, a_accel_eff=6.0), v_max=V_MAX
    )
    gradient = reference.speed_gradient
    # Real accelerating and braking on this layout: the profile has to gain and
    # lose several m/s over a couple of metres.
    assert float(gradient.max()) > 0.5, "no acceleration survived the smoothing"
    assert float(gradient.min()) < -0.5, "no braking survived the smoothing"
    # But no sample-to-sample alternation: neighbouring gradients must agree.
    jitter = (gradient - torch.roll(gradient, 1)).abs()
    assert float(jitter.max()) < 0.5, "gradient still alternating between samples"


def test_speed_scale_moves_the_target(circle):
    slow = build_reference(
        circle, ControllerParams(speed_scale=np.full(SPEED_POINTS, 0.7)), v_max=V_MAX
    )
    fast = build_reference(
        circle, ControllerParams(speed_scale=np.full(SPEED_POINTS, 1.15)), v_max=V_MAX
    )
    assert float(slow.speed.mean()) < float(fast.speed.mean())


def test_reference_line_follows_the_control_points(circle):
    """A uniform offset must come through as that offset, not a smoothed fraction."""
    params = ControllerParams(line=np.full(LINE_POINTS, 0.12))
    reference = build_reference(circle, params, v_max=V_MAX)
    assert float(reference.offset.mean()) == pytest.approx(0.12, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════
#  The control law
# ══════════════════════════════════════════════════════════════════════════


def test_actions_are_in_bounds_for_random_states(controller, circle):
    """The SDK clamps anyway, but a controller that saturates constantly is
    telling the search nothing."""
    count = 512
    robot = FakeRobot(num_envs=count)
    generator = torch.Generator().manual_seed(0)
    robot.data.root_pos_w[:, :2] = (
        torch.rand(count, 2, generator=generator) - 0.5
    ) * 12.0
    robot.set_yaw(torch.rand(count, generator=generator) * 2 * math.pi)
    robot.data.root_lin_vel_b[:, 0] = torch.rand(count, generator=generator) * V_MAX

    from lituanicax_sdk.state import CarState, ControlHistory
    from lituanicax_sdk.vehicle import TIMING

    car = CarState(
        robot=robot,
        track=circle_track_of(controller),
        vehicle=VEHICLE,
        step_dt=TIMING.step_dt,
        throttle_ids=torch.tensor([0, 1, 2, 3]),
        steer_ids=torch.tensor([4, 5]),
        suspension_ids=torch.tensor([6, 7, 8, 9]),
        episode_step=torch.zeros(count, dtype=torch.long),
        commands=ControlHistory(count, "cpu"),
        applied_wheel_torque=torch.zeros(count, 4),
        wall_touched=torch.zeros(count, dtype=torch.bool),
    )
    actions = controller(car)
    assert actions.shape == (count, 2)
    assert torch.isfinite(actions).all()
    assert float(actions.min()) >= -1.0 and float(actions.max()) <= 1.0


def circle_track_of(controller):
    """The controller keeps geometry, not the SDK Track; tests need the latter."""
    import csv
    import tempfile
    from pathlib import Path

    from lituanicax_sdk.track import Track, TrackCfg

    directory = Path(tempfile.mkdtemp())
    path = directory / "circle.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for i in range(720):
            theta = 2.0 * math.pi * i / 720
            writer.writerow([RADIUS * math.cos(theta), RADIUS * math.sin(theta), 0.0])
    return Track(
        TrackCfg(
            name="circle",
            walls_usd="unused.usd",
            centerline_csv=str(path),
            centerline_scale=(1.0, 1.0),
        ),
        device="cpu",
    )


def test_steers_back_towards_the_line(controller, robot, make_state):
    """Displaced outward, the car must steer inward, and vice versa."""
    inward = controller(
        make_state(place(robot, radius=RADIUS + 0.15, theta=0.0, speed=2.0))
    )
    assert float(inward[0, 1]) > 0.0, "outside the line -> steer left (inward)"

    outward = controller(
        make_state(place(FakeRobot(1), radius=RADIUS - 0.15, theta=0.0, speed=2.0))
    )
    assert float(outward[0, 1]) < float(inward[0, 1])


def test_steers_into_the_corner_when_on_the_line(controller, robot, make_state):
    """On a counterclockwise circle with no error, the answer is left."""
    actions = controller(make_state(place(robot, radius=RADIUS, theta=1.0, speed=2.0)))
    assert float(actions[0, 1]) > 0.0


def test_faster_cars_look_further_ahead(circle, robot, make_state):
    """The lookahead is speed-scaled; a fixed one is twitchy fast or lazy slow.

    Checked through behaviour: at higher speed the same displacement produces
    less steering, because the aim point is further away.
    """
    params = ControllerParams(k_v=0.3, L_0=0.2, L_min=0.05, L_max=3.0, k_e=0.0, k_d=0.0)
    controller = Controller(
        circle,
        build_reference(circle, params, v_max=V_MAX),
        params,
        wheelbase_m=WHEELBASE,
        max_steer_rad=VEHICLE.max_steer_rad,
    )
    slow = controller(
        make_state(place(robot, radius=RADIUS + 0.15, theta=0.0, speed=0.5))
    )
    fast = controller(
        make_state(place(FakeRobot(1), radius=RADIUS + 0.15, theta=0.0, speed=6.0))
    )
    assert abs(float(fast[0, 1])) < abs(float(slow[0, 1]))


def test_throttle_opens_when_below_target_and_closes_above(
    controller, robot, make_state
):
    target = math.sqrt(8.0 * RADIUS)
    slow = controller(
        make_state(place(robot, radius=RADIUS, theta=0.0, speed=target - 2.0))
    )
    fast = controller(
        make_state(place(FakeRobot(1), radius=RADIUS, theta=0.0, speed=target + 2.0))
    )
    assert float(slow[0, 0]) > 0.0
    assert float(fast[0, 0]) < 0.0


def test_a_braking_feedforward_cannot_hold_a_slow_car_still(
    controller, robot, make_state
):
    """The failure that stopped Gate 1 dead, and did not look like a failure.

    The feedforward describes the acceleration a car *already on* the speed
    profile needs. A car well below it, in a braking zone because there is a
    corner coming, got a feedforward of -2.9 against a proportional term of
    +2.9. They cancelled, the throttle sat at 0.03, and ten out of ten cars
    crawled at 0.15 m/s until the stall rule retired them 5.5 m into the lap.

    Nothing about that reads as broken from the outside: the profile was
    sensible, the gains were sensible, and the car simply would not go.
    """
    reference = controller.reference
    controller.reference = Reference(
        offset=reference.offset,
        kappa=reference.kappa,
        speed=reference.speed,
        # Everywhere a hard braking zone, which is the case that broke it.
        speed_gradient=torch.full_like(reference.speed_gradient, -3.0),
    )
    target = math.sqrt(8.0 * RADIUS)
    crawling = controller(
        make_state(place(robot, radius=RADIUS, theta=0.0, speed=target - 2.9))
    )
    assert float(crawling[0, 0]) > 0.5, "a car 2.9 m/s too slow must open the throttle"


def test_a_braking_feedforward_still_brakes_a_fast_car(controller, robot, make_state):
    """The gate must not cost the feedforward its job, only its veto."""
    reference = controller.reference
    controller.reference = Reference(
        offset=reference.offset,
        kappa=reference.kappa,
        speed=reference.speed,
        speed_gradient=torch.full_like(reference.speed_gradient, -3.0),
    )
    target = math.sqrt(8.0 * RADIUS)
    hot = controller(
        make_state(place(robot, radius=RADIUS, theta=0.0, speed=target + 0.2))
    )
    assert float(hot[0, 0]) < 0.0, "over the target in a braking zone means brake"


def dynamic(circle, **gains):
    """A controller with the dynamic terms turned on."""
    params = ControllerParams(**gains)
    return Controller(
        circle,
        build_reference(circle, params, v_max=V_MAX),
        params,
        wheelbase_m=WHEELBASE,
        max_steer_rad=VEHICLE.max_steer_rad,
    )


def test_the_dynamic_terms_are_off_by_default(circle, robot, make_state):
    """The baseline must survive adding them.

    The kinematic law produced a 17.1 s lap; the search starts from that vector,
    so with the three new gains at zero the actions have to be bit-identical to
    what they were. Anything else means the search restarts from somewhere worse
    than where it left off.
    """
    plain = dynamic(circle)
    car = make_state(place(robot, radius=RADIUS, theta=0.3, speed=2.0))
    before = plain(car).clone()

    with_zero_gains = dynamic(circle, k_r=0.0, k_beta=0.0, k_rotate=0.0)
    assert torch.allclose(with_zero_gains(car), before)


def test_sideslip_feedback_counter_steers(circle, robot, make_state):
    """The whole point of the term, and it needs no mode switch to do it.

    A car whose rear has stepped out to the left is sliding right; the correction
    is to steer *right*, against the direction the geometry is asking for. That
    falls straight out of subtracting the sideslip.
    """
    controller = dynamic(circle, k_beta=2.0)
    straight = make_state(place(robot, radius=RADIUS, theta=0.0, speed=2.0))
    neutral = float(controller(straight)[0, 1])

    # Same pose, but the car is moving sideways: positive lateral velocity.
    sliding = FakeRobot(1)
    place(sliding, radius=RADIUS, theta=0.0, speed=2.0)
    sliding.data.root_lin_vel_b[:, 1] = 1.5
    slid = float(controller(make_state(sliding))[0, 1])

    assert slid < neutral, "a rear stepping out one way must be caught the other"


def test_yaw_feedback_asks_for_more_lock_when_under_rotating(
    circle, robot, make_state
):
    """A car going straight on in a corner is not turning as fast as v*kappa."""
    controller = dynamic(circle, k_r=1.0)
    plain = dynamic(circle)
    car = make_state(place(robot, radius=RADIUS, theta=0.0, speed=2.0, yaw=None))
    # place() puts the car on the circle with the right yaw but no yaw rate, so
    # it is under-rotating by exactly v*kappa.
    assert abs(float(controller(car)[0, 1])) > abs(float(plain(car)[0, 1]))


def test_rotation_braking_only_fires_at_the_steering_stop(circle, robot, make_state):
    """Inert while the steering still has authority in hand.

    Otherwise it would be braking the car through every corner it was handling
    perfectly well, which is a slower lap and a worse baseline.
    """
    gentle = make_state(place(robot, radius=RADIUS, theta=0.0, speed=1.0))
    plain = dynamic(circle)
    braking = dynamic(circle, k_rotate=2.0)
    # A 5 m circle at 1 m/s needs almost no steering, so the stop is far away.
    assert float(braking(gentle)[0, 0]) == pytest.approx(float(plain(gentle)[0, 0]))


def test_lead_is_inert_at_zero_and_on_a_constant_corner(circle, robot, make_state):
    """Two properties at once, and the second is why the first is not enough.

    At ``k_lead = 0`` the law must be exactly the one that verified at 15.367 s,
    so the search starts where it left off. And on a circle — whose curvature is
    the same everywhere — leading the reference can have no effect at any gain,
    which pins the term to *changing* curvature rather than to arc length.
    """
    car = make_state(place(robot, radius=RADIUS, theta=0.4, speed=2.5))
    plain = dynamic(circle)
    assert torch.allclose(dynamic(circle, k_lead=0.0)(car), plain(car))
    # Not bit-identical, unlike the gain at zero: a circle sampled every 20 mm
    # has curvature that ripples by a fraction of a percent, so reading it 0.75 m
    # further round moves the command by 2e-4 rad. That is the discretization,
    # not the term — a real change of corner moves it by two orders more.
    assert torch.allclose(dynamic(circle, k_lead=0.3)(car), plain(car), atol=1e-3)


def test_lead_steers_for_the_corner_ahead_not_the_one_underneath(robot, make_state):
    """The point of the term: on the approach to a corner it turns in early.

    A servo that takes ten steps to place the wheels puts the angle on the road
    about two metres further round than where it was asked for, so a controller
    reading the reference at its own arc length is permanently a corner behind.
    Here the car sits on a straight with a corner ahead: without lead the
    curvature underneath it is nil and the feedforward asks for nothing.
    """
    from lituanicax_sdk.track import Track
    from lituanicax_sdk.tracks import OFFICIAL

    geometry = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
    params = ControllerParams()
    reference = build_reference(geometry, params, v_max=V_MAX)
    kappa = reference.kappa

    speed, lead = 5.0, 0.4
    step = int(round(speed * lead / geometry.spacing_m))
    # Somewhere flat now with a real corner one lead-distance ahead. Restricted
    # to the flat samples first: the largest *gain* in curvature on this layout
    # is corner-to-tighter-corner, which is not the case being tested.
    flat = kappa.abs() < 0.2
    index = int(torch.argmax(torch.where(flat, kappa.roll(-step).abs(), -1.0)))
    assert float(kappa[index].abs()) < 0.2, "the chosen spot is not a straight"

    position = geometry.pos[index]
    tangent = geometry.tangent[index]
    yaw = float(torch.atan2(tangent[1], tangent[0]))
    robot.data.root_pos_w[:, 0] = float(position[0])
    robot.data.root_pos_w[:, 1] = float(position[1])
    robot.set_yaw(yaw)
    robot.data.root_lin_vel_b[:, 0] = speed
    robot.data.root_lin_vel_w[:, 0] = speed * math.cos(yaw)
    robot.data.root_lin_vel_w[:, 1] = speed * math.sin(yaw)
    car = make_state(robot, track=Track(OFFICIAL, device="cpu"))

    def steer(k_lead):
        tuned = ControllerParams(k_lead=k_lead)
        controller = Controller(
            geometry,
            build_reference(geometry, tuned, v_max=V_MAX),
            tuned,
            wheelbase_m=WHEELBASE,
            max_steer_rad=VEHICLE.max_steer_rad,
        )
        return float(controller(car)[0, 1])

    ahead = float(kappa[(index + step) % geometry.num_samples])
    late, early = steer(0.0), steer(lead)
    # Turning *into* the corner that is coming, so the extra steering carries
    # the sign of the curvature ahead rather than merely being different.
    assert abs(early) > abs(late)
    assert (early - late > 0) == (ahead > 0)


def test_slip_target_is_inert_at_zero(circle, robot, make_state):
    """The property that lets it be searched from the existing best.

    At slip_gain = 0 the sideslip term drives beta to zero exactly as before, so
    the law is the one that produced 15.067 s and the search starts from a
    verified point rather than a hopeful one.
    """
    car = make_state(place(robot, radius=RADIUS, theta=0.4, speed=2.5))
    plain = dynamic(circle, k_beta=1.0)
    with_target = dynamic(circle, k_beta=1.0, slip_gain=0.0)
    assert torch.allclose(plain(car), with_target(car))


def test_slip_target_asks_for_more_steering_into_the_corner(
    circle, robot, make_state
):
    """A car pointing where its wheels point is *under* the target, so the term
    adds steering rather than removing it.

    That is what makes it a plan instead of an error: the same feedback builds
    the slide going in and catches it coming out, with no mode switch and no
    slip estimate to threshold on.
    """
    car = make_state(place(robot, radius=RADIUS, theta=0.0, speed=2.5))
    plain = dynamic(circle, k_beta=1.0, slip_gain=0.0)
    sliding = dynamic(circle, k_beta=1.0, slip_gain=0.15)
    # The circle turns one way, so more steering means a larger command with the
    # same sign, not merely a different one.
    before, after = float(plain(car)[0, 1]), float(sliding(car)[0, 1])
    assert abs(after) > abs(before)
    assert (after > 0) == (before > 0), "the extra steering must go into the corner"


def test_slip_target_is_capped(circle, robot, make_state):
    """Past about 23 degrees the car is spinning, not cornering."""
    from teacher.controller import MAX_SLIP_RAD

    params = ControllerParams(slip_gain=SCALAR_BOUNDS["slip_gain"].high)
    reference = build_reference(circle, params, v_max=V_MAX)
    # The circle's curvature times the largest gain, capped.
    worst = float((params.slip_gain * reference.kappa.abs()).max())
    assert min(worst, MAX_SLIP_RAD) <= MAX_SLIP_RAD


def test_separate_accelerate_and_brake_gains_are_both_used(circle, robot, make_state):
    """The car's limits are asymmetric, so one gain cannot serve both."""
    params = ControllerParams(k_p_accel=4.0, k_p_brake=0.5, k_ff=0.0)
    controller = Controller(
        circle,
        build_reference(circle, params, v_max=V_MAX),
        params,
        wheelbase_m=WHEELBASE,
        max_steer_rad=VEHICLE.max_steer_rad,
    )
    target = math.sqrt(8.0 * RADIUS)
    under = controller(
        make_state(place(robot, radius=RADIUS, theta=0.0, speed=target - 0.5))
    )
    over = controller(
        make_state(place(FakeRobot(1), radius=RADIUS, theta=0.0, speed=target + 0.5))
    )
    # 4.0 * 0.5 = 2.0 saturates; 0.5 * -0.5 = -0.25 does not.
    assert float(under[0, 0]) == pytest.approx(1.0)
    assert float(over[0, 0]) == pytest.approx(-0.25, abs=0.05)


def test_the_controller_is_memoryless(controller, robot, make_state):
    """Two identical states must give identical actions, in any order.

    Load-bearing for the search: a whole CMA-ES population shares one
    environment, so any hidden state would leak between candidates.
    """
    first = controller(
        make_state(place(robot, radius=RADIUS + 0.1, theta=2.0, speed=3.0))
    )
    _ = controller(make_state(place(FakeRobot(1), radius=RADIUS, theta=0.5, speed=1.0)))
    again = controller(
        make_state(place(FakeRobot(1), radius=RADIUS + 0.1, theta=2.0, speed=3.0))
    )
    assert torch.allclose(first, again)


def test_batched_cars_do_not_interfere(controller, make_state):
    """Each row must be what that car would get on its own."""
    thetas = [0.0, 1.0, 2.0, 3.0, 4.0]
    together = FakeRobot(len(thetas))
    for i, theta in enumerate(thetas):
        together.data.root_pos_w[i, 0] = (RADIUS + 0.1) * math.cos(theta)
        together.data.root_pos_w[i, 1] = (RADIUS + 0.1) * math.sin(theta)
        together.data.root_lin_vel_b[i, 0] = 2.0
    yaws = torch.tensor([t + math.pi / 2 for t in thetas])
    together.set_yaw(yaws)
    for i, theta in enumerate(thetas):
        together.data.root_lin_vel_w[i, 0] = 2.0 * math.cos(theta + math.pi / 2)
        together.data.root_lin_vel_w[i, 1] = 2.0 * math.sin(theta + math.pi / 2)

    batched = controller(make_state(together))
    for i, theta in enumerate(thetas):
        alone = controller(
            make_state(place(FakeRobot(1), radius=RADIUS + 0.1, theta=theta, speed=2.0))
        )
        assert torch.allclose(batched[i], alone[0], atol=1e-5), theta


# ══════════════════════════════════════════════════════════════════════════
#  Driving a whole population at once
# ══════════════════════════════════════════════════════════════════════════


def test_a_population_matches_its_candidates_one_by_one(circle, make_state):
    """The correctness test CMA-ES depends on.

    A generation is scored by putting every (candidate, start) pair in its own
    environment and driving them together. If the batched path diverges from the
    single-candidate path at all, the search optimizes something other than what
    a lap will be driven with — and the discrepancy would show up as CMA-ES
    converging on a candidate that then benchmarks differently.
    """
    from teacher.controller import stack_references

    candidates = [
        ControllerParams(line=np.full(LINE_POINTS, 0.05), k_e=1.0, k_p_accel=1.5),
        ControllerParams(line=np.full(LINE_POINTS, -0.10), k_e=4.0, L_0=0.5),
        ControllerParams(
            speed_scale=np.full(SPEED_POINTS, 0.8), a_lat_eff=11.0, w_ff=0.3
        ),
    ]
    references = [build_reference(circle, p, v_max=V_MAX) for p in candidates]

    # Two starts each, so the layout is the same shape the search uses.
    thetas = [0.4, 2.2]
    rows = torch.tensor([c for c in range(len(candidates)) for _ in thetas])
    count = len(rows)

    robot = FakeRobot(count)
    yaws = []
    for index in range(count):
        theta = thetas[index % len(thetas)]
        robot.data.root_pos_w[index, 0] = (RADIUS + 0.08) * math.cos(theta)
        robot.data.root_pos_w[index, 1] = (RADIUS + 0.08) * math.sin(theta)
        heading = theta + math.pi / 2
        yaws.append(heading)
        robot.data.root_lin_vel_b[index, 0] = 2.5
        robot.data.root_lin_vel_w[index, 0] = 2.5 * math.cos(heading)
        robot.data.root_lin_vel_w[index, 1] = 2.5 * math.sin(heading)
    robot.set_yaw(torch.tensor(yaws))

    batched = Controller(
        circle,
        stack_references(references),
        candidates,
        wheelbase_m=WHEELBASE,
        max_steer_rad=VEHICLE.max_steer_rad,
        rows=rows,
    )
    together = batched(make_state(robot))

    for index in range(count):
        candidate = candidates[int(rows[index])]
        alone = Controller(
            circle,
            references[int(rows[index])],
            candidate,
            wheelbase_m=WHEELBASE,
            max_steer_rad=VEHICLE.max_steer_rad,
        )
        theta = thetas[index % len(thetas)]
        single = alone(
            make_state(
                place(FakeRobot(1), radius=RADIUS + 0.08, theta=theta, speed=2.5)
            )
        )
        assert torch.allclose(together[index], single[0], atol=1e-5), index


def test_population_rejects_inconsistent_arguments(circle):
    from teacher.controller import stack_references

    params = ControllerParams()
    reference = build_reference(circle, params, v_max=V_MAX)
    common = dict(wheelbase_m=WHEELBASE, max_steer_rad=VEHICLE.max_steer_rad)

    with pytest.raises(ValueError, match="rows only makes sense"):
        Controller(
            circle, reference, params, rows=torch.zeros(3, dtype=torch.long), **common
        )

    with pytest.raises(ValueError, match="needs rows"):
        Controller(circle, stack_references([reference]), [params], **common)

    with pytest.raises(ValueError, match=r"stacked \[C, M\]"):
        Controller(
            circle, reference, [params], rows=torch.zeros(2, dtype=torch.long), **common
        )


# ══════════════════════════════════════════════════════════════════════════
#  The constraint the whole approach rests on
# ══════════════════════════════════════════════════════════════════════════


class Recorder:
    """Forwards to a ``CarState`` and remembers what was asked for."""

    def __init__(self, target):
        self.__dict__["_target"] = target
        self.__dict__["seen"] = set()

    def __getattr__(self, name):
        self.seen.add(name)
        return getattr(self._target, name)


def test_controller_reads_only_car_state(controller, robot, make_state):
    """Checklist item 1: the controller may only read what a policy can.

    ``CarState`` *is* the policy-visible surface — the SDK hands exactly this to
    ``compute_observations`` — so the test is that nothing private is touched and
    that every name read is really on ``CarState``. A controller that reached for
    ``car._robot`` would work fine and produce demonstrations no network could
    ever reproduce.
    """
    car = make_state(place(robot, radius=RADIUS + 0.1, theta=1.0, speed=3.0))
    recorder = Recorder(car)
    controller(recorder)  # type: ignore[arg-type]

    assert recorder.seen, "the recorder saw nothing; the proxy is not working"
    private = {name for name in recorder.seen if name.startswith("_")}
    assert not private, f"controller reached past the policy surface: {private}"

    for name in recorder.seen:
        assert hasattr(type(car), name), f"{name} is not a CarState attribute"


def test_controller_does_not_need_the_simulator(controller, robot, make_state):
    """It runs against the fake robot in these tests, which is the point: the
    control law is Isaac-free and so is its whole test suite."""
    actions = controller(make_state(place(robot, radius=RADIUS, theta=0.0, speed=2.0)))
    assert actions.shape == (1, 2)


def test_works_on_a_track_it_has_never_seen(circle_track):
    """Nothing may be hardcoded to one layout."""
    geometry = TrackGeometry.from_track(circle_track)
    params = ControllerParams()
    controller = Controller(
        geometry,
        build_reference(geometry, params, v_max=V_MAX),
        params,
        wheelbase_m=WHEELBASE,
        max_steer_rad=VEHICLE.max_steer_rad,
    )
    assert controller.geometry.length > 0.0
