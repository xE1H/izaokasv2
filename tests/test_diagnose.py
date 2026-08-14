"""Tests for :mod:`tools.diagnose`.

The verdict this module prints decides how Phases 3 and 4 get planned, so it is
driven against synthetic traces whose answer is known by construction. A
diagnostic that quietly says the wrong thing is worse than none: it would send
the next phase after the wrong mechanism.
"""

from __future__ import annotations

import numpy as np
import pytest

from lituanicax_sdk.vehicle import VEHICLE
from tools.diagnose import report

STEPS, CARS = 200, 4
WHEELBASE = 0.224


def trace(*, steer, wheels, kappa=0.5, speed=2.0, throttle=0.5, up=1.0):
    """A trace where every quantity is whatever the test says it is."""
    shape = (STEPS, CARS)
    return {
        "up_axis": np.full(shape, up),
        "steer_cmd": np.full(shape, steer),
        "steer_angle": np.full(shape, wheels),
        "ref_kappa": np.full(shape, kappa),
        "speed": np.full(shape, speed),
        "ref_speed": np.full(shape, speed),
        "throttle": np.full(shape, throttle),
    }


def test_a_saturated_car_is_called_steering_limited(capsys):
    summary = report(trace(steer=1.0, wheels=0.25), wheelbase_m=WHEELBASE)
    assert summary["saturated"] == pytest.approx(1.0)
    assert "STEERING-LIMITED" in capsys.readouterr().out


def test_a_car_with_steering_in_hand_is_not(capsys):
    summary = report(trace(steer=0.3, wheels=0.14), wheelbase_m=WHEELBASE)
    assert summary["saturated"] == pytest.approx(0.0)
    assert "ACCELERATION-LIMITED" in capsys.readouterr().out


def test_the_radius_reported_is_the_one_the_wheels_can_hold():
    """The decisive number, and the reason the verdict is what it is.

    Commanding full lock is not reaching it: the servo is effort-limited, so what
    matters is the angle the wheels got to. At 0.251 rad on a 0.224 m wheelbase
    that is 0.87 m — against a corridor whose flattest line is 0.545 m.
    """
    summary = report(trace(steer=1.0, wheels=0.251), wheelbase_m=WHEELBASE)
    assert summary["wheels_at_full_lock_rad"] == pytest.approx(0.251)
    assert summary["radius_at_full_lock_m"] == pytest.approx(0.87, abs=0.01)
    # Half of what was asked for, which is the finding in one number.
    assert summary["wheels_at_full_lock_rad"] / VEHICLE.max_steer_rad < 0.6


def test_tipped_cars_do_not_count():
    """Everything a car does after going over is about the crash, not the law."""
    upright = report(trace(steer=1.0, wheels=0.25), wheelbase_m=WHEELBASE)
    rolled = report(trace(steer=1.0, wheels=0.25, up=-0.5), wheelbase_m=WHEELBASE)
    assert upright["saturated"] == pytest.approx(1.0)
    assert rolled["saturated"] == 0.0, "a tipped car contributes nothing"


def test_nothing_left_needs_both_axes_pinned():
    both = report(trace(steer=1.0, wheels=0.25, throttle=1.0), wheelbase_m=WHEELBASE)
    steering_only = report(
        trace(steer=1.0, wheels=0.25, throttle=0.2), wheelbase_m=WHEELBASE
    )
    assert both["nothing_left"] == pytest.approx(1.0)
    assert steering_only["nothing_left"] == pytest.approx(0.0)


def test_bands_split_by_how_tight_the_reference_is():
    straight = report(trace(steer=1.0, wheels=0.25, kappa=0.1), wheelbase_m=WHEELBASE)
    hairpin = report(trace(steer=1.0, wheels=0.25, kappa=2.5), wheelbase_m=WHEELBASE)
    assert straight["bands"][0]["kappa_low"] == 0.0
    assert hairpin["bands"][0]["kappa_low"] == 2.0
