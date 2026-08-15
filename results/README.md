# Measured results — 2026-08-14/15, RTX 4090 (vast.ai)

Committed deliberately. `artifacts/` is gitignored because most of what lands
there is regenerable and the defaults in it are guesses until a probe has run.
These are not: they are what the simulator actually said, and regenerating them
costs a GPU session.

## The controller

| file | what it is |
|---|---|
| `teacher-best.json` | **the best verified driver: 14.900 s** |
| `trace-best.npz` | its scored lap, per step — inputs and outputs of the control law |
| `final-s1.json`, `final-s2.json` | where two independent CMA-ES runs finished |
| `teacher-warmstart.json` | the unoptimized starting point |
| `driven-line.png` | the path the car drives, against the one it is given |

`teacher-best.json` verifies at **14.900 s** at the official 60 s window, ten
environments, official rules — `python -m teacher.optimize --headless --measure
results/teacher-best.json`. Re-runs land within about 0.03 s.

Against a board best of 14.3 s, and 17.367 s where this session started.

### Measure one file per process

`--measure` used to build one environment and reuse it for every file named on
the command line. The benchmark does not: it builds a fresh one. The identical
file measured three times in a single process reads

    v3.json  15.500 s    <- the only benchmark-faithful number
    v3.json  15.367 s
    v3.json  15.367 s

so everything after the first is scored on a simulator carrying the previous
attempt's residue and reads about 0.13 s fast. That silently corrupted every
multi-file comparison — file #1 was held to the official condition and its
rivals were not — and it is how a six-candidate sweep crowned a vector that,
measured alone, does not complete a lap. `--measure` now fans out to one cold
process per file. **Any figure in an older note that came from a multi-file
run is suspect.**

## The RL policy

Training in `teamcode/` — 40 observations with a curvature preview, a convex
lap-time bonus, `gamma` 0.9965, 3072 environments. Progress from a standing
start, all figures the *flying* lap TensorBoard reports:

| iteration | best flying lap |
|---|---|
| 38 | 28.0 s — first completed laps |
| 47 | 17.6 s |
| 93 | 15.1 s |
| 175 | 14.2 s |
| 258 | 14.1 s |

### TensorBoard is not the score, and the gap is not constant

`Lap/best_lap_time_s` times a **flying** lap, gate to gate. The benchmark times
a **standing start** from the world origin. Measured on the same checkpoints:

    flying 17.5 s  ->  scored 17.933 s   (+0.43)
    flying 14.2 s  ->  scored 15.000 s   (+0.77)

The offset **grows as the car gets faster**, because the launch costs a roughly
fixed amount of time against an ever-shorter lap. So a target of 14.3 s scored
needs about **13.5 s flying**, not 13.9 s. Any projection from TensorBoard that
assumes a constant offset will be optimistic, and the first one made here was.

### The heading band has a fast tail, unlike the controller

Ten attempts at seed 0 on the 15.000 s checkpoint spanned **15.000 to 16.033 s**
— a 1.03 s range, with the two quickest at −2.7° and −4.8°. That matters
because `--agents` is uncapped and replayed verbatim by verification, so a
distribution with a real basin can be sampled for its best. The deterministic
controller failed exactly this test: its best lap was flat from 10 agents to
150, because when it lapped at all it lapped at 14.9–15.2 and the variance was
in *whether* it finished rather than in how fast.

Worth noting the two fastest headings sit outside the ±1.15° band the policy
trains on, so this tail is generalisation rather than the specialisation it was
designed for. The conclusion — that spending agents pays here — holds either
way, but the mechanism is not the intended one.

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

Read `driven-line.png` first: it is the traced lap coloured by speed, and it
settles where the time is. The lap is **power-limited where the track is
straight and grip-limited where it is not**:

* peak **5.21 m/s against a 6.92 m/s top speed** — never close to maximum, even
  on the long straight. The falloff explains it: at 3 m/s the car has 2.78 m/s²
  left and at 5 m/s only 1.36, so ~5.2 is the straight's honest ceiling.
* **2.85–3.05 m/s through the R = 1.0 m corners**, against a grip limit of
  `sqrt(9.62 × 1.0) = 3.10` — already at the limit.

A note on reading the line at all: the **reference is not the path**. The
steering is a first-order lag and pure pursuit aims ahead, so the car smooths
whatever it is handed — reference R_min 0.308 m against a driven R_min of
0.583 m on the same lap. Judging the line off the parameter vector is measuring
the wrong curve, and the corridor bound applies to *knot values*, not to the
lateral offset the car reaches.

### What moved the number, and what did not

The one structural gain this session was **lead compensation**. Every steering
term read the reference at the car's own arc length, but the servo needs about
0.33 s, so the commanded angle lands two metres further round — most of a corner
here. Reading the reference at `s + k_lead·v` inverts that known lag and is
worth **0.567 s** (15.500 → 14.933). At `k_lead = 0` the law is bit-identical to
its predecessor.

Everything substituted wholesale **failed to complete a lap**: a re-solved wider
line, a scaled line, a blended line, rate-limited lines, and physically honest
profile constants. The gains and the line are tightly co-adapted, so the search
is the only thing that can move them together. Relatedly, the `*_eff` scalars
are *effective* parameters absorbing model error, not physical claims —
`a_accel_eff` sits at 10.89 against a measured 4.90, and correcting it to the
truth stops the car lapping.

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
