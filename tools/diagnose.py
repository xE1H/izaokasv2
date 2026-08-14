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
narrowly: 47.5% of every upright step, rising to 98% in the tightest corners,
against a planning-time prediction that saturation would cover about 0.75 m of
the 50 m lap.
"""

from __future__ import annotations

import argparse

import numpy as np

from lituanicax_sdk.vehicle import VEHICLE
from tools.measured import Measured

#: Curvature bands to report, 1/m. Chosen so each holds a meaningful share of the
#: official track rather than to make a point.
BANDS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 10.0)

#: ``up_axis`` above this is a car still on its wheels. Everything a tipped car
#: does is about the crash, not about the control law.
UPRIGHT = 0.9

#: Commands this close to the limit are saturated. Not 1.0 exactly: the action is
#: a float that has been through a clamp and a division.
SATURATED = 0.99


def report(trace: dict, *, wheelbase_m: float) -> dict:
    """Print the diagnostic, and return the numbers it printed.

    Args:
        trace: the arrays ``teacher.optimize --trace`` wrote.
        wheelbase_m: from the probe, not the spec sheet — it turns the steering
            angle the wheels actually reached into the radius they can hold.
    """
    up, steer, kappa = trace["up_axis"], trace["steer_cmd"], trace["ref_kappa"]
    speed, ref_speed, throttle = trace["speed"], trace["ref_speed"], trace["throttle"]
    wheels = trace["steer_angle"]

    alive = up > UPRIGHT
    saturated = np.abs(steer) > SATURATED
    limit = VEHICLE.max_steer_rad

    if not alive.any():
        # Every mean below would be nan, and `nan > threshold` is False, so the
        # verdict would come back ACCELERATION-LIMITED from no data at all —
        # confidently, and in the direction that sends the next phase after the
        # wrong mechanism.
        print(
            "  NO VERDICT: every car was on its roof for the whole trace, so there\n"
            "  is nothing here about the control law. Fix the crash first."
        )
        return {"saturated": 0.0, "bands": [], "upright_steps": 0}

    share = float(saturated[alive].mean())

    print(f"{int(alive.sum())} upright steps, {share:.1%} with the steering saturated")

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
    asking = alive & saturated
    delivered = radius_at_full_lock = None
    if asking.sum():
        delivered = float(np.median(np.abs(wheels[asking])))
        at_speed = float(np.median(speed[asking]))
        radius_at_full_lock = float(wheelbase_m / np.tan(max(delivered, 1e-6)))
        print(
            f"\n  full lock commanded {int(asking.sum())} times: the wheels reached "
            f"{delivered:.3f} rad ({delivered / limit:.0%} of the\n  limit) at "
            f"{at_speed:.2f} m/s, which is a real minimum radius of "
            f"{radius_at_full_lock:.2f} m."
        )

    pinned = alive & saturated & (throttle >= SATURATED)
    summary = {
        "saturated": share,
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
            f"  STEERING-LIMITED. The car is at its steering limit for {share:.0%} of\n"
            "  the lap, which is not a few clips in the tightest corners — it is the\n"
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
    report(dict(np.load(args.trace)), wheelbase_m=car.wheelbase_m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
