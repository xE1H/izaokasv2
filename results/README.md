# Measured results — 2026-08-14/15, RTX 4090 (vast.ai)

Committed deliberately. `artifacts/` is gitignored because most of what lands
there is regenerable and the defaults in it are guesses until a probe has run.
These are not: they are what the simulator actually said, and regenerating them
costs a GPU session.

## The controller

| file | what it is |
|---|---|
| `teacher-best.json` | **the best verified driver: 15.067 s** |
| `final-s1.json`, `final-s2.json` | where two independent CMA-ES runs finished |
| `teacher-warmstart.json` | the unoptimized starting point |

`teacher-best.json` verifies at **15.067 s** at the official 60 s window, ten
environments, official rules — `python -m teacher.optimize --headless --measure
results/teacher-best.json`. It re-measures at 15.167 s; the ~0.1 s is chaotic
sensitivity between runs, not drift.

Against a board best of 14.3 s, and 17.367 s where this session started.

## The car

| file | what it is |
|---|---|
| `dynamics.json` | the seven Phase 0 numbers, **as corrected** |
| `grip.json` | the cornering limit, measured properly |
| `headings.json` | the ten headings the benchmark actually draws |

Three of these were wrong the first time in ways that looked like physics:

* **`a_lat_max` was 8.83 m/s²** — taken at steering angles where the car was
  *steering*-limited, not grip-limited, so it recorded where the servo stopped
  rather than where the tyres did. `tools/grip.py` measures 9.62 sustained and
  12.5 held for a full second. That number sets every corner speed in the
  profile.
* **The steering lag read 195 steps** (6.5 s, which is not a servo) because it
  was measured against a slow later creep instead of the plateau. It is 10.
* **A rollover at 3.2 m/s²** that was a wall impact — forward speed went
  +6.67 → −0.87 m/s in one 33 ms step. Believing it would have put every speed
  target at a third of what the car can do.

`headings.json` is the set of ten the benchmark will score, read out of a running
environment. They are stable across processes and across a change of episode
window, and they are **not** shaped like an evenly spaced proxy — six of the ten
sit above +2.9°. Optimizing against the proxy cost real time before this was
measured.

## What is known about the ceiling

The quasi-static model on these numbers predicts **15.05 s** for a line the car
can steer; the car does 15.067. It is executing its own plan to within 0.02 s, so
the remaining gap to 14.3 s is in the model rather than the tracking.

All three ways to beat a point-mass model were implemented and measured on
`teacher-best.json`, and all three are worse — deliberate sideslip gives no lap
at any value, rotation by braking degrades monotonically then stops lapping, and
slowing down to regain steering authority gives no lap. Two independent searches
also drove `slip_gain` to ~0.001 on their own, having been given it free.

The binding constraint is the steering actuator: full lock delivers R = 0.56 m at
2.4 m/s while the reference asks for 0.39 m, and the car is saturated for 41% of
the lap. Yaw-rate feedback (`k_r`) is the one dynamic term that is load-bearing —
remove it and there is no lap at all.

See `RUNBOOK.md` for the operational detail, including the finding that the
simulator only reproduces up to 120 environments.
