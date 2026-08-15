"""Solve a line for the corridor that is now legal, and keep the driver that works.

Scaling the existing line outward failed -- it does not widen corners so much as
push the straights into walls. The corridor has to be re-solved, so the min-time
solve is run at the new half-width with the measured car and the servo's rate
limit, and dropped into the gains that verified at 14.9 s.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/root/ltxsim")
from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from teacher.params import LINE_POINTS  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.measured import Measured  # noqa: E402
from tools.profile import (  # noqa: E402
    lap_time,
    offset_path,
    periodic_basis,
    racing_line_offsets,
    three_pass_profile,
)

base = json.load(open("/root/v5.json"))
geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
car = Measured.load("/root/ltxsim/artifacts/dynamics.json")
print(car.describe())
fine = periodic_basis(geom.num_samples, LINE_POINTS)

# The search found a_lat_eff ~12.8 on its own, which is the transient limit the
# grip probe measured rather than the sustained one. Solve the line for the car
# the search says it is driving, not a more cautious one.
a_lat = float(base.get("a_lat_eff", car.a_lat_effective_m_s2))
print(f"solving at a_lat {a_lat:.2f}")

print(f"\n{'tag':<16}{'half':>7}{'rate':>7}{'path':>8}{'R_min':>8}{'model':>8}")
for tag, half, rate in (
    ("wide", 0.195, None),
    ("wide-rate", 0.195, 2.5),
    ("mid-rate", 0.170, 2.5),
):
    _, rep = racing_line_offsets(
        geom,
        a_lat=a_lat,
        v_max=car.v_max_m_s,
        half_width=half,
        kappa_max=1.0 / car.r_min_m,
        control_points=LINE_POINTS,
        steer_rate_limit=rate,
        wheelbase_m=car.wheelbase_m,
    )
    coeffs = np.asarray(rep["coefficients"])
    _, seg, kap = offset_path(geom, fine @ coeffs)
    speed = three_pass_profile(
        seg, kap, a_lat=a_lat, a_accel=car.a_accel_standing_m_s2,
        a_brake=car.a_brake_m_s2, v_max=car.v_max_m_s,
    )
    print(
        f"{tag:<16}{half:7.3f}{str(rate):>7}{seg.sum():8.2f}"
        f"{1 / max(abs(kap).max(), 1e-9):8.3f}{float(lap_time(seg, speed)):8.2f}"
    )
    out = dict(base)
    out["line"] = coeffs.tolist()
    json.dump(out, open(f"/root/{tag}.json", "w"), indent=2)
