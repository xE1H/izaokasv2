"""Does the line demand steering faster than the servo can move?

A line is only worth what the car can follow. The wheel angle a line needs is
atan(L*kappa); how fast that angle has to change is v * d/ds of it. The servo
reaches 90% of a commanded angle in about ten control steps, so there is a rate
it cannot beat, and a line demanding more than that is a line the car is always
behind — however good it looks to a point mass.
"""

import sys

import numpy as np

sys.path.insert(0, "/root/ltxsim")
from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.vehicle import TIMING, VEHICLE  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from teacher.controller import build_reference  # noqa: E402
from teacher.params import ControllerParams  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.measured import Measured  # noqa: E402
from tools.profile import offset_path, three_pass_profile  # noqa: E402

geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
car = Measured.load("/root/ltxsim/artifacts/dynamics.json")
n = geom.num_samples
ds = geom.spacing_m

# What the servo can do: it covers most of its travel in the measured lag.
lag_s = car.steer_lag_steps * TIMING.step_dt
servo_rate = VEHICLE.max_steer_rad / lag_s
print(f"servo: {VEHICLE.max_steer_rad:.3f} rad in {lag_s:.3f} s "
      f"-> about {servo_rate:.2f} rad/s at full travel\n")

print(f"{'line':<24}{'p50':>8}{'p95':>8}{'max':>8}   share of steps over servo rate")
for label, path in (
    ("carried (40 knots)", "/root/base-carried.json"),
    ("solved (120 knots)", "/root/base-solved.json"),
    ("v3 (verified)", "/root/v3.json"),
):
    params = ControllerParams.load(path)
    ref = build_reference(
        geom, params, v_max=car.v_max_m_s, wheelbase_m=car.wheelbase_m
    )
    off = ref.offset.numpy()
    _, seg, kap = offset_path(geom, off)
    speed = three_pass_profile(
        seg, kap, a_lat=params.a_lat_eff, a_accel=params.a_accel_eff,
        a_brake=params.a_brake_eff, v_max=car.v_max_m_s,
    )
    delta = np.arctan(car.wheelbase_m * kap)
    # Central difference round the loop, then to rad/s at the planned speed.
    ddelta_ds = (np.roll(delta, -1) - np.roll(delta, 1)) / (2 * ds)
    rate = np.abs(ddelta_ds * speed)
    over = float((rate > servo_rate).mean())
    print(
        f"{label:<24}{np.percentile(rate, 50):8.2f}{np.percentile(rate, 95):8.2f}"
        f"{rate.max():8.2f}   {over:.1%}"
    )
