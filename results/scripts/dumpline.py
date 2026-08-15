"""Dump the track, its walls, and a candidate's racing line as plain coordinates."""

import json
import sys

import numpy as np

sys.path.insert(0, "/root/ltxsim")
from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from teacher.controller import build_reference  # noqa: E402
from teacher.params import ControllerParams  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.measured import Measured  # noqa: E402

geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
car = Measured.load("/root/ltxsim/artifacts/dynamics.json")

centre = geom.pos.cpu().numpy()
normal = geom.normal.cpu().numpy()
HALF = 0.351  # track half width; the wall is here, the crash radius is 0.15 inside

out = {
    "centerline": centre.tolist(),
    "wall_left": (centre + HALF * normal).tolist(),
    "wall_right": (centre - HALF * normal).tolist(),
    "limit_left": (centre + (HALF - 0.15) * normal).tolist(),
    "limit_right": (centre - (HALF - 0.15) * normal).tolist(),
    "length_m": float(geom.length),
    "lines": {},
}

for label, path in [(a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]:
    params = ControllerParams.load(path)
    ref = build_reference(
        geom, params, v_max=car.v_max_m_s, wheelbase_m=car.wheelbase_m
    )
    offset = ref.offset.cpu().numpy()
    speed = ref.speed.cpu().numpy()
    out["lines"][label] = {
        "xy": (centre + offset[:, None] * normal).tolist(),
        "speed": speed.tolist(),
        "offset": offset.tolist(),
        "v_min": float(speed.min()),
        "v_max": float(speed.max()),
    }
    print(f"{label}: offset {np.abs(offset).max():.3f} m, speed {speed.min():.2f}-{speed.max():.2f}")

json.dump(out, open("/root/line-data.json", "w"))
print("written /root/line-data.json")
