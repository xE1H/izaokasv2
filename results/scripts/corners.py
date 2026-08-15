"""Is the line leaving radius on the table, corner by corner?

For each corner: the radius the line achieves, the radius the *flattest possible*
line in the corridor achieves there, and how much of the corridor the line is
actually using. A corner where the line sits near the centerline while the
flattest line is far wider is a corner where radius is being given away.
"""

import sys

import numpy as np

sys.path.insert(0, "/root/ltxsim")
from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from teacher.params import ControllerParams  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.profile import (  # noqa: E402
    offset_path,
    periodic_basis,
    widest_line_coefficients,
)

geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
n = geom.num_samples
arc = np.arange(n) * geom.spacing_m

params = ControllerParams.load(sys.argv[1] if len(sys.argv) > 1 else "/root/best.json")
basis = periodic_basis(n, len(params.line))
line = basis @ np.asarray(params.line, dtype=np.float64)
_, _, k_line = offset_path(geom, line)
_, _, k_centre = offset_path(geom, np.zeros(n))

# The flattest line the full corridor allows, in the same basis the search uses
# and in a much finer one, so the basis itself can be blamed or cleared.
flat_coarse = widest_line_coefficients(geom, half_width=0.18, control_points=40)
flat_fine = widest_line_coefficients(geom, half_width=0.18, control_points=160)
f_coarse = periodic_basis(n, 40) @ flat_coarse.numpy()
f_fine = periodic_basis(n, 160) @ flat_fine.numpy()
_, _, k_flat_c = offset_path(geom, f_coarse)
_, _, k_flat_f = offset_path(geom, f_fine)

print(f"peak |offset|: line {np.abs(line).max():.3f}  "
      f"flattest@40 {np.abs(f_coarse).max():.3f}  flattest@160 {np.abs(f_fine).max():.3f}")
print(f"R_min:        line {1/np.abs(k_line).max():.3f}  "
      f"flattest@40 {1/np.abs(k_flat_c).max():.3f}  flattest@160 {1/np.abs(k_flat_f).max():.3f}"
      f"  centerline {1/np.abs(k_centre).max():.3f}")

# Corners: contiguous runs where the centerline is meaningfully curved.
tight = np.abs(k_centre) > 0.5
runs, start = [], None
for i, t in enumerate(np.r_[tight, False]):
    if t and start is None:
        start = i
    elif not t and start is not None:
        if i - start > 15:
            runs.append((start, i))
        start = None

print(f"\n{len(runs)} corners with |kappa| > 0.5 (R < 2 m)\n")
print(f"{'at':>7} {'len':>5} {'R centre':>9} {'R line':>8} {'R flat':>8} "
      f"{'offset':>8} {'used':>6}  giving away")
lost = []
for a, b in runs:
    sl = slice(a, b)
    r_c = 1 / max(np.abs(k_centre[sl]).max(), 1e-9)
    r_l = 1 / max(np.abs(k_line[sl]).max(), 1e-9)
    r_f = 1 / max(np.abs(k_flat_f[sl]).max(), 1e-9)
    off = np.abs(line[sl]).max()
    used = off / 0.18
    gap = r_f - r_l
    lost.append(gap)
    flag = "  <-- " + f"{gap:+.2f} m" if gap > 0.05 else ""
    print(f"{arc[a]:7.1f} {arc[b]-arc[a]:5.1f} {r_c:9.3f} {r_l:8.3f} {r_f:8.3f} "
          f"{off:8.3f} {used:5.0%}{flag}")

print(f"\ncorners where the flattest line beats this one: "
      f"{sum(1 for g in lost if g > 0.05)} of {len(runs)}")
print(f"mean radius given away: {np.mean([g for g in lost if g > 0.05] or [0]):.3f} m")

# Can 40 control points even place an apex? Width of one basis bump.
row = periodic_basis(n, 40)[:, 20]
support = (row > 0.05 * row.max()).sum() * geom.spacing_m
print(f"\none control point at 40 knots influences {support:.2f} m of track")
row80 = periodic_basis(n, 80)[:, 40]
print(f"one control point at 80 knots influences "
      f"{(row80 > 0.05 * row80.max()).sum() * geom.spacing_m:.2f} m")
print(f"typical corner length: {np.mean([arc[b]-arc[a] for a, b in runs]):.2f} m")
