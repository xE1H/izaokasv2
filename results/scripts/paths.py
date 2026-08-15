"""Is the wider line simply a longer one?"""

import sys

import numpy as np

sys.path.insert(0, "/root/ltxsim")
from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from teacher.controller import build_reference  # noqa: E402
from teacher.params import ControllerParams  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.measured import Measured  # noqa: E402
from tools.profile import lap_time, offset_path, three_pass_profile  # noqa: E402

geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
car = Measured.load("/root/ltxsim/artifacts/dynamics.json")
n = geom.num_samples

print(f"{'candidate':<24}{'path m':>9}{'R_min':>8}{'model lap':>11}")
rows = [
    ("centerline", None),
    ("carried (40 knots)", "/root/base-carried.json"),
    ("solved (120 knots)", "/root/base-solved.json"),
    ("v3 (verified 15.367)", "/root/v3.json"),
]
for label, path in rows:
    if path is None:
        off = np.zeros(n)
    else:
        ref = build_reference(
            geom,
            ControllerParams.load(path),
            v_max=car.v_max_m_s,
            wheelbase_m=car.wheelbase_m,
        )
        off = ref.offset.numpy()
    _, seg, kap = offset_path(geom, off)
    speed = three_pass_profile(
        seg,
        kap,
        a_lat=car.a_lat_effective_m_s2,
        a_accel=car.a_accel_standing_m_s2,
        a_brake=car.a_brake_m_s2,
        v_max=car.v_max_m_s,
    )
    print(
        f"{label:<24}{seg.sum():9.2f}{1 / max(abs(kap).max(), 1e-9):8.3f}"
        f"{float(lap_time(seg, speed)):11.2f}"
    )
