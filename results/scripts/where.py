"""Where does the lap actually go? Read it off the car, not off the plan."""

import sys

import numpy as np

sys.path.insert(0, "/root/ltxsim")
from lituanicax_sdk.track import Track  # noqa: E402
from lituanicax_sdk.tracks import OFFICIAL  # noqa: E402
from lituanicax_sdk.vehicle import VEHICLE  # noqa: E402
from tools.geometry import TrackGeometry  # noqa: E402
from tools.profile import offset_path  # noqa: E402

d = np.load("/root/trace-v5.npz")
geom = TrackGeometry.from_track(Track(OFFICIAL, device="cpu"), spacing_m=0.02)
length = float(d["track_length_m"])

# The scored car is the one that got round. Find it by how far it actually
# travelled: `s` wraps at the line, so max(s)-min(s) calls a car that merely
# straddled the wrap a lapper, and a retired car freezes in place and answers
# every other question with nonsense (R_min 0.023 m, servo delivering 0.01).
s, n = d["s"], d["n"]
travelled = np.array([
    np.nansum(np.linalg.norm(np.diff(d["pos"][:, i], axis=0), axis=1))
    for i in range(s.shape[1])
])
for i, t in enumerate(travelled):
    print(f"  car {i}: travelled {t:6.2f} m")
car = int(np.argmax(travelled))
print(f"\nanalysing car {car} — the only one that can have been scored\n")

alive = np.isfinite(s[:, car]) & (d["speed"][:, car] > 0.05)
sc, nc = s[alive, car], n[alive, car]
ref_n, ref_v = d["ref_n"][alive, car], d["ref_speed"][alive, car]
v, cmd, ang = d["speed"][alive, car], d["steer_cmd"][alive, car], d["steer_angle"][alive, car]
pos = d["pos"][alive, car]

step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
print(f"driven path        {step.sum():.2f} m over {alive.sum()} steps")
print(f"lateral offset     driven mean|n| {np.abs(nc).mean():.3f}  peak {np.abs(nc).max():.3f}")
print(f"                   reference mean {np.abs(ref_n).mean():.3f}  peak {np.abs(ref_n).max():.3f}")
print(f"tracking error     median {np.median(np.abs(nc - ref_n)):.3f} m  "
      f"p95 {np.percentile(np.abs(nc - ref_n), 95):.3f}  max {np.abs(nc - ref_n).max():.3f}")

# Curvature of the path the car actually drove, smoothed over ~0.2 m.
head = np.unwrap(np.arctan2(*np.diff(pos, axis=0)[:, ::-1].T))
win = 9
k_drv = np.abs(np.gradient(head) / np.maximum(step, 1e-6))
k_drv = np.convolve(k_drv, np.ones(win) / win, mode="same")
print(f"driven R_min       {1 / max(k_drv[win:-win].max(), 1e-9):.3f} m "
      f"(reference {1 / max(np.abs(d['ref_kappa'][alive, car]).max(), 1e-9):.3f} m)")

print(f"\nspeed              mean {v.mean():.2f}  vs reference {ref_v.mean():.2f} m/s")
deficit = ref_v - v
print(f"speed deficit      mean {deficit.mean():+.2f} m/s, "
      f"{100 * (deficit > 0.3).mean():.0f}% of steps more than 0.3 below plan")

sat = np.abs(cmd) >= VEHICLE.max_steer_rad * 0.99
print(f"\nsteering           commanded at the stop {sat.mean():.1%} of steps")
reach = np.abs(ang) / np.maximum(np.abs(cmd), 1e-6)
print(f"servo delivered    median {np.median(reach[np.abs(cmd) > 0.05]):.2f} of what was asked")

# Where the time goes: the car is only as fast as its slowest sections.
print(f"\n{'arc band':>12}{'v driven':>10}{'v plan':>9}{'deficit':>9}{'|n|':>7}{'sat':>7}")
edges = np.linspace(0, length, 11)
band = np.digitize(sc % length, edges) - 1
for b in range(10):
    m = band == b
    if m.sum() < 3:
        continue
    print(f"{edges[b]:6.0f}-{edges[b + 1]:<5.0f}{v[m].mean():10.2f}{ref_v[m].mean():9.2f}"
          f"{(ref_v - v)[m].mean():+9.2f}{np.abs(nc[m]).mean():7.3f}{sat[m].mean():7.0%}")
