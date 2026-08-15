"""Solve lines under a steering-rate limit and see what it costs on paper."""

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

base = json.load(open("/root/best.json"))
geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
car = Measured.load("/root/ltxsim/artifacts/dynamics.json")
n = geom.num_samples
fine = periodic_basis(n, LINE_POINTS)
ds = geom.spacing_m

print(f"{'limit rad/s':<13}{'path':>8}{'R_min':>8}{'model lap':>11}{'over rate':>11}")
for limit in (None, 4.0, 2.5, 1.5):
    line, rep = racing_line_offsets(
        geom,
        a_lat=car.a_lat_effective_m_s2,
        v_max=car.v_max_m_s,
        half_width=0.14,
        kappa_max=1.0 / car.r_min_m,
        control_points=LINE_POINTS,
        steer_rate_limit=limit,
        wheelbase_m=car.wheelbase_m,
    )
    coeffs = np.asarray(rep["coefficients"])
    _, seg, kap = offset_path(geom, fine @ coeffs)
    speed = three_pass_profile(
        seg, kap, a_lat=car.a_lat_effective_m_s2,
        a_accel=car.a_accel_standing_m_s2, a_brake=car.a_brake_m_s2,
        v_max=car.v_max_m_s,
    )
    angle = np.arctan(car.wheelbase_m * kap)
    rate = np.abs((np.roll(angle, -1) - np.roll(angle, 1)) / (2 * ds) * speed)
    tag = "none" if limit is None else f"{limit:.1f}"
    print(
        f"{tag:<13}{seg.sum():8.2f}{1 / max(abs(kap).max(), 1e-9):8.3f}"
        f"{float(lap_time(seg, speed)):11.2f}{(rate > 1.46).mean():11.1%}"
    )
    out = dict(base)
    out["line"] = coeffs.tolist()
    json.dump(out, open(f"/root/rl-{tag}.json", "w"), indent=2)
