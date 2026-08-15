"""The line the car actually takes, against the line it was told to take.

Every picture so far has drawn the *reference* -- the curve in the parameter
vector. The car does not drive that: the steering is a first-order lag and pure
pursuit aims ahead, so the driven path is smoother than its reference and sits
somewhere else entirely. This draws what the wheels did, coloured by speed,
with the reference and the centerline underneath for comparison.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, "/root/ltxsim")
from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from teacher.controller import build_reference  # noqa: E402
from teacher.params import ControllerParams  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.measured import Measured  # noqa: E402

INK, RULE = "#14161A", "#D8D5CE"
SURFACE, WALL, LIMIT, CENTRE = "#EFEDE8", "#8A867C", "#C6C2B8", "#B3AFA5"
REF = "#B0553F"

out_path, trace_path, params_path = sys.argv[1], sys.argv[2], sys.argv[3]

geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
car = Measured.load("/root/ltxsim/artifacts/dynamics.json")
centre = geom.pos.cpu().numpy()
normal = geom.normal.cpu().numpy()
HALF, CRASH = 0.351, 0.15

ref = build_reference(
    geom, ControllerParams.load(params_path), v_max=car.v_max_m_s,
    wheelbase_m=car.wheelbase_m,
)
ref_xy = centre + ref.offset.cpu().numpy()[:, None] * normal

d = np.load(trace_path)
# The scored car is the one that actually got round the loop.
travelled = np.array([
    np.nansum(np.linalg.norm(np.diff(d["pos"][:, i], axis=0), axis=1))
    for i in range(d["pos"].shape[1])
])
i = int(np.argmax(travelled))
alive = np.isfinite(d["s"][:, i]) & (d["speed"][:, i] > 0.05)
pos, speed = d["pos"][alive, i], d["speed"][alive, i]

fig, ax = plt.subplots(figsize=(13, 11), dpi=170)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

outer, inner = centre + HALF * normal, centre - HALF * normal
ax.fill(np.r_[outer[:, 0], outer[0, 0], inner[::-1, 0], inner[-1, 0]],
        np.r_[outer[:, 1], outer[0, 1], inner[::-1, 1], inner[-1, 1]],
        color=SURFACE, zorder=0, lw=0)
for off, c, w, ls in ((HALF, WALL, 1.5, "-"), (-HALF, WALL, 1.5, "-"),
                      (HALF - CRASH, LIMIT, 1.0, (0, (5, 4))),
                      (-(HALF - CRASH), LIMIT, 1.0, (0, (5, 4))),
                      (0.0, CENTRE, 0.9, (0, (2, 5)))):
    p = centre + off * normal
    ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]],
            color=c, lw=w, ls=ls, zorder=2)

ax.plot(np.r_[ref_xy[:, 0], ref_xy[0, 0]], np.r_[ref_xy[:, 1], ref_xy[0, 1]],
        color=REF, lw=1.6, zorder=3, alpha=0.85, label="reference it was given")

segments = np.stack([pos[:-1], pos[1:]], axis=1)
lc = LineCollection(segments, cmap="viridis", zorder=5,
                    norm=plt.Normalize(speed.min(), speed.max()))
lc.set_array(speed[:-1])
lc.set_linewidth(3.4)
ax.add_collection(lc)
bar = fig.colorbar(lc, ax=ax, fraction=0.030, pad=0.02)
bar.set_label("speed driven, m/s", color=INK, fontsize=11)
bar.outline.set_edgecolor(RULE)

ax.plot([], [], color="#2C6E63", lw=3.4, label="path actually driven")
ax.plot(*pos[0], "o", ms=8, color="#C77C1E", zorder=7, label="spawn")

ax.set_aspect("equal")
ax.axis("off")
ax.legend(fontsize=11.5, frameon=False, labelcolor=INK, loc="lower left")
ax.set_title(
    f"What the car drives, not what it is told to drive\n"
    f"driven {np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.1f} m   "
    f"mean {speed.mean():.2f} m/s   peak {speed.max():.2f} m/s",
    fontsize=13.5, color=INK, pad=16)
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
print("wrote", out_path)
