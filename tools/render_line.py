"""Show what changed between two racing lines, where it changed.

    python -m tools.render_line out.png "old"=a.json "new"=b.json

**Needs no simulator** — it reads parameter files and the track, and wants
``matplotlib``, which is not a dependency of anything else here.

At full-track scale two lines through a 0.70 m corridor are indistinguishable,
so the whole plan view is one panel with both drawn over each other, and the
corners where they diverge most get their own zoomed panels underneath. The
zoom targets are chosen by measurement — the largest gain in corner radius —
not by eye.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from teacher.controller import build_reference  # noqa: E402
from teacher.params import ControllerParams  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.measured import Measured  # noqa: E402
from tools.profile import offset_path  # noqa: E402

INK, DIM, RULE = "#14161A", "#6B7078", "#DCDBD6"
SURFACE, WALL, LIMIT, CENTRE = "#EDECE7", "#8C887E", "#C3BFB5", "#B0ACA2"
OLD_C, NEW_C = "#B0553F", "#1F7A8C"

out_path = sys.argv[1]
(old_label, old_path), (new_label, new_path) = [a.split("=", 1) for a in sys.argv[2:4]]

geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
car = Measured.load("artifacts/dynamics.json")
centre = geom.pos.cpu().numpy()
normal = geom.normal.cpu().numpy()
n = geom.num_samples
arc = np.arange(n) * geom.spacing_m
HALF, CRASH = 0.351, 0.15


def load(path):
    params = ControllerParams.load(path)
    ref = build_reference(geom, params, v_max=car.v_max_m_s, wheelbase_m=car.wheelbase_m)
    off = ref.offset.cpu().numpy()
    _, _, kap = offset_path(geom, off)
    return {"xy": centre + off[:, None] * normal, "offset": off, "kappa": kap,
            "speed": ref.speed.cpu().numpy()}


old, new = load(old_path), load(new_path)
_, _, k_centre = offset_path(geom, np.zeros(n))

# Corners, and where the new line gains the most radius through one.
tight = np.abs(k_centre) > 0.5
runs, start = [], None
for i, t in enumerate(np.r_[tight, False]):
    if t and start is None:
        start = i
    elif not t and start is not None:
        if i - start > 15:
            runs.append((start, i))
        start = None

gains = []
for a, b in runs:
    r_old = 1 / max(np.abs(old["kappa"][a:b]).max(), 1e-9)
    r_new = 1 / max(np.abs(new["kappa"][a:b]).max(), 1e-9)
    gains.append((r_new - r_old, a, b, r_old, r_new))
gains.sort(reverse=True)
picks = gains[:3]

fig = plt.figure(figsize=(13.6, 12.4), dpi=170)
gs = fig.add_gridspec(2, 3, height_ratios=[2.35, 1], hspace=0.06, wspace=0.09)
fig.patch.set_facecolor("white")


def draw_track(ax, lw=1.0):
    outer, inner = centre + HALF * normal, centre - HALF * normal
    ax.fill(np.r_[outer[:, 0], outer[0, 0], inner[::-1, 0], inner[-1, 0]],
            np.r_[outer[:, 1], outer[0, 1], inner[::-1, 1], inner[-1, 1]],
            color=SURFACE, zorder=0, lw=0)
    for off, c, w, ls in ((HALF, WALL, lw * 1.3, "-"), (-HALF, WALL, lw * 1.3, "-"),
                          (HALF - CRASH, LIMIT, lw * 0.9, (0, (5, 4))),
                          (-(HALF - CRASH), LIMIT, lw * 0.9, (0, (5, 4))),
                          (0.0, CENTRE, lw * 0.8, (0, (3, 4)))):
        p = centre + off * normal
        ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]],
                color=c, lw=w, ls=ls, zorder=2)


ax = fig.add_subplot(gs[0, :])
ax.set_facecolor("white")
draw_track(ax)
for L, c, lab, w in ((old, OLD_C, old_label, 1.7), (new, NEW_C, new_label, 1.9)):
    p = L["xy"]
    ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]],
            color=c, lw=w, zorder=4, label=lab, solid_capstyle="round")
ax.plot(*old["xy"][0], "o", ms=6, color="#C77C1E", zorder=6)

for rank, (gain, a, b, r_old, r_new) in enumerate(picks, 1):
    mid = (a + b) // 2
    ax.annotate(str(rank), xy=centre[mid], xytext=(centre[mid] + 0.62 * normal[mid]),
                fontsize=11, fontweight="bold", color=INK, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.26", fc="white", ec=INK, lw=1.1),
                zorder=7)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(fontsize=11.5, frameon=False, labelcolor=INK, loc="lower left",
          bbox_to_anchor=(0.02, 0.02))
ax.set_title(
    f"{old_label}   R$_{{min}}$ {1 / np.abs(old['kappa']).max():.3f} m        "
    f"{new_label}   R$_{{min}}$ {1 / np.abs(new['kappa']).max():.3f} m",
    fontsize=13, color=INK, pad=14)

for col, (gain, a, b, r_old, r_new) in enumerate(picks):
    zx = fig.add_subplot(gs[1, col])
    zx.set_facecolor("white")
    draw_track(zx, lw=1.4)
    for L, c, w in ((old, OLD_C, 2.4), (new, NEW_C, 2.7)):
        zx.plot(L["xy"][:, 0], L["xy"][:, 1], color=c, lw=w, zorder=4,
                solid_capstyle="round")
    pad = 0.75
    seg = centre[max(a - 40, 0):b + 40]
    zx.set_xlim(seg[:, 0].min() - pad, seg[:, 0].max() + pad)
    zx.set_ylim(seg[:, 1].min() - pad, seg[:, 1].max() + pad)
    zx.set_aspect("equal")
    zx.set_xticks([]); zx.set_yticks([])
    for s in zx.spines.values():
        s.set_color(RULE)
    zx.set_title(f"{col + 1}   at {arc[a]:.0f} m      R  {r_old:.2f} → "
                 f"{r_new:.2f} m", fontsize=11, color=INK, pad=8)

fig.savefig(out_path, bbox_inches="tight", facecolor="white")
print("wrote", out_path)
for gain, a, b, r_old, r_new in gains[:6]:
    print(f"  corner at {arc[a]:5.1f} m: R {r_old:.3f} -> {r_new:.3f}  ({gain:+.3f} m)")
