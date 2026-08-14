"""Phase 2.4 — read a controller trace and say what the lap was actually limited by.

    python -m teacher.optimize --headless --measure <params> --trace run.npz
    python -m tools.diagnose run.npz

**Needs no simulator** — it reads the ``.npz`` that ``--trace`` writes, which is
why this is a separate module from the thing that produces it.

The question it exists to answer is the one that decides how Phases 3 and 4 get
planned, and the plan is explicit that it must not be decided in advance: is the
lap lost to *steering saturation* in a few tight corners, or gradually to
*acceleration* out of the many medium ones?

* Saturation — throttle rotation is the remaining mechanism, a pure-pursuit law
  cannot discover it at any parameter setting, and the RL stage is where the
  remaining time lives.
* Acceleration — the controller structure is close to sufficient and the RL stage
  is polish.

On the official track with the measured car it came back saturation, and not
narrowly: rising to nearly every step in the tightest corners, against a
planning-time prediction that saturation would cover about 0.75 m of the 50 m lap.

Two ways to get this badly wrong, both found the hard way and both guarded
against here. A crashed car is *frozen*, not removed, so it keeps appearing in
the trace as upright and holding its last command — count those and the steering
looks far weaker than it is. And the servo needs ten steps to arrive, so
averaging over every saturated step measures the wheels on their way rather than
where they got to.
"""

from __future__ import annotations

import argparse

import numpy as np

from lituanicax_sdk.rules import FLIPPED_UP_AXIS_THRESHOLD
from lituanicax_sdk.vehicle import VEHICLE
from tools.measured import Measured

#: Curvature bands to report, 1/m. Chosen so each holds a meaningful share of the
#: official track rather than to make a point.
BANDS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 10.0)

#: ``up_axis`` above this is a car the rules still consider to be racing.
#:
#: Deliberately the rulebook's own figure (``rules.FLIPPED_UP_AXIS_THRESHOLD``,
#: about 73° of roll) rather than something tidier. Nothing in the rules
#: penalises sliding, and leaning is free right up to that line — so a filter at
#: 0.9, which is 26° of roll, would throw away exactly the hard cornering that
#: the fastest laps are made of and report the car as tamer than it is.
UPRIGHT = FLIPPED_UP_AXIS_THRESHOLD

#: A car slower than this is not driving, and must not be counted as if it were.
#:
#: Under official rules a crashed car is *frozen in place* rather than removed,
#: and the trace keeps recording it: upright, stationary, holding whatever
#: command it died with. On a 10-car trace that was 13% of all "upright" steps,
#: every one of them counted as saturated steering at full lock.
#:
#: It mattered. Including them, the wheels appeared to reach 52% of the commanded
#: angle, implying a 0.87 m minimum radius against a corridor that can only
#: deliver 0.545 m — which reads as "no drivable line exists". Excluding them the
#: figure is 75% and 0.58 m, which is a 7% shortfall rather than a 60% one, and a
#: completely different conclusion about whether this approach can work.
MOVING_M_S = 0.2

#: Commands this close to the limit are saturated. Not 1.0 exactly: the action is
#: a float that has been through a clamp and a division.
SATURATED = 0.99


def _held(flag: np.ndarray) -> np.ndarray:
    """For each step, how many consecutive steps ``flag`` has been true.

    Used to tell a servo that has arrived from one still on its way. Written as a
    running count rather than a Python loop over steps because a trace is
    ``[steps, cars]`` and the loop would be over the long axis.
    """
    held = np.zeros_like(flag, dtype=np.int32)
    for step in range(1, flag.shape[0]):
        held[step] = np.where(flag[step], held[step - 1] + 1, 0)
    return held


def report(trace: dict, *, wheelbase_m: float, settle_steps: int = 10) -> dict:
    """Print the diagnostic, and return the numbers it printed.

    Args:
        trace: the arrays ``teacher.optimize --trace`` wrote.
        wheelbase_m: from the probe, not the spec sheet — it turns the steering
            angle the wheels actually reached into the radius they can hold.
        settle_steps: how long a command must be held before the wheels are
            taken to have arrived. The measured servo lag is ten steps.
    """
    up, steer, kappa = trace["up_axis"], trace["steer_cmd"], trace["ref_kappa"]
    speed, ref_speed, throttle = trace["speed"], trace["ref_speed"], trace["throttle"]
    wheels = trace["steer_angle"]

    alive = (up > UPRIGHT) & (speed > MOVING_M_S)
    saturated = np.abs(steer) > SATURATED
    limit = VEHICLE.max_steer_rad

    if not alive.any():
        # Every mean below would be nan, and `nan > threshold` is False, so the
        # verdict would come back ACCELERATION-LIMITED from no data at all —
        # confidently, and in the direction that sends the next phase after the
        # wrong mechanism.
        print(
            "  NO VERDICT: no car was upright and moving at any point, so there is\n"
            "  nothing here about the control law. Fix the crash first."
        )
        return {"saturated": 0.0, "bands": [], "upright_steps": 0}

    share = float(saturated[alive].mean())

    print(
        f"{int(alive.sum())} steps upright and moving, "
        f"{share:.1%} with the steering saturated"
    )

    print("\n  how tight the reference is        steps   saturated   speed   wheels")
    bands = []
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        band = alive & (np.abs(kappa) >= low) & (np.abs(kappa) < high)
        if band.sum() < 20:
            continue
        radius = f"R {1 / high:.2f}-{1 / low:.2f} m" if low else "R > 4 m"
        entry = {
            "kappa_low": low,
            "kappa_high": high,
            "steps": int(band.sum()),
            "saturated": float(saturated[band].mean()),
            "speed_m_s": float(np.median(speed[band])),
            "wheels_rad": float(np.median(np.abs(wheels[band]))),
        }
        bands.append(entry)
        print(
            f"  {radius:>16} {entry['steps']:>12d} {entry['saturated']:>10.1%} "
            f"{entry['speed_m_s']:>7.2f} {entry['wheels_rad']:>8.3f}"
        )

    # The decisive number. The steering is an effort-limited servo, so asking for
    # full lock does not mean getting it, and how much of it the car gets decides
    # the radius it can actually hold.
    # Only where the command has been held long enough for the servo to arrive.
    # It takes about ten steps, so averaging over every saturated step measures
    # the wheels on their way rather than where they got to.
    asking = alive & saturated & (_held(saturated) >= settle_steps)
    delivered = radius_at_full_lock = None
    if asking.sum():
        delivered = float(np.median(np.abs(wheels[asking])))
        at_speed = float(np.median(speed[asking]))
        radius_at_full_lock = float(wheelbase_m / np.tan(max(delivered, 1e-6)))
        print(
            f"\n  full lock held past the servo lag {int(asking.sum())} times: the "
            f"wheels reached\n  {delivered:.3f} rad ({delivered / limit:.0%} of the "
            f"limit) at {at_speed:.2f} m/s, a real minimum radius of "
            f"{radius_at_full_lock:.2f} m."
        )

    pinned = alive & saturated & (throttle >= SATURATED)
    summary = {
        "saturated": share,
        "upright_steps": int(alive.sum()),
        "bands": bands,
        "wheels_at_full_lock_rad": delivered,
        "radius_at_full_lock_m": radius_at_full_lock,
        "below_target_speed": float((alive & (speed < ref_speed - 0.3))[alive].mean()),
        "full_throttle": float((throttle >= SATURATED)[alive].mean()),
        "full_brake": float((throttle <= -SATURATED)[alive].mean()),
        "nothing_left": float(pinned[alive].mean()),
    }

    print(
        f"\n  below target speed  {summary['below_target_speed']:>6.1%}\n"
        f"  full throttle       {summary['full_throttle']:>6.1%}\n"
        f"  full brake          {summary['full_brake']:>6.1%}\n"
        f"  steering saturated  {summary['saturated']:>6.1%}\n"
        f"  both at once        {summary['nothing_left']:>6.1%}  "
        "(no authority left in either axis)"
    )

    print()
    if share > 0.25:
        print(
            f"  STEERING-LIMITED. The car is at its limit for {share:.0%} of the\n"
            "  lap, which is not a few clips in the tightest corners — it is the\n"
            "  lap. A pure-pursuit law has no move left here: it is already asking\n"
            "  for everything. The remaining time is in rotating the car with the\n"
            "  throttle, which this control law cannot express."
        )
    else:
        print(
            "  ACCELERATION-LIMITED. Steering has authority in hand for most of the\n"
            "  lap, so the controller structure is close to sufficient and the time\n"
            "  is going into getting out of the medium corners."
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", help="An .npz written by teacher.optimize --trace.")
    parser.add_argument("--dynamics", default="artifacts/dynamics.json")
    args = parser.parse_args()
    car = Measured.load(args.dynamics)
    report(
        dict(np.load(args.trace)),
        wheelbase_m=car.wheelbase_m,
        settle_steps=int(round(car.steer_lag_steps)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
