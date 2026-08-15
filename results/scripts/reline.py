"""Give the tuned driver a line its new basis can actually express.

The gains are good — they achieve the quasi-static model's own prediction — so
they are kept. The line is the part that was basis-limited, so it is solved
fresh at the new resolution rather than refitted from the coarse one, which
would only carry the old limitation across in finer clothing.

Writes three candidates: the fitted-across line (what the old one becomes at the
new resolution), a freshly solved min-time line, and the flattest line the
corridor allows. Which is faster is a question for the simulator.
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
    offset_path,
    periodic_basis,
    racing_line_offsets,
    widest_line_coefficients,
)

base = json.load(open("/root/best.json"))
geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
car = Measured.load("/root/ltxsim/artifacts/dynamics.json")
n = geom.num_samples
HALF = 0.14  # leave a little corridor for tracking error; the search may spend it

old = np.asarray(base["line"], dtype=np.float64)
fine = periodic_basis(n, LINE_POINTS)


def report(label, coeffs):
    line = fine @ coeffs
    _, _, k = offset_path(geom, line)
    print(
        f"{label:<12} R_min {1 / max(np.abs(k).max(), 1e-9):.3f} m   "
        f"peak offset {np.abs(line).max():.3f} m"
    )
    out = dict(base)
    out["line"] = coeffs.tolist()
    json.dump(out, open(f"/root/seed-{label}.json", "w"), indent=2)


# 1. What the old line becomes at the new resolution: least squares, so the
#    profile is preserved rather than the knot values copied across a basis
#    whose bumps are three times narrower.
old_line = periodic_basis(n, len(old)) @ old
carried, *_ = np.linalg.lstsq(fine, old_line, rcond=None)
report("carried", carried)

# 2. A min-time line solved at the resolution that can hold an apex.
solved, rep = racing_line_offsets(
    geom,
    a_lat=car.a_lat_effective_m_s2,
    v_max=car.v_max_m_s,
    half_width=HALF,
    kappa_max=1.0 / car.r_min_m,
    control_points=LINE_POINTS,
)
report("solved", np.asarray(rep["coefficients"]))

# 3. The flattest line the corridor allows — the other end of the trade.
report("flattest", widest_line_coefficients(
    geom, half_width=HALF, control_points=LINE_POINTS
).numpy())
