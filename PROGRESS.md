# How the lap time got from 17.4 s to 13.967 s

Written for someone arriving with no context. It covers what the problem is,
what was built, what the numbers mean, every wrong turn worth knowing about,
and where everything lives.

**Result: 13.967 s, submitted, awaiting verification, would be P1.**
Checkpoint `results/rl/model_1125_13.967s_1k.pt`, scored with
`--agents 1000 --seed 1`.

---

## 1. The problem

Drive a MuSHR nano v2 RC car round a 50 m × 0.70 m closed track in NVIDIA Isaac
Sim, as fast as possible. The score is the **fastest single attempt** out of
however many are run. Every attempt starts at the world origin facing along the
track, with a seeded yaw jitter of up to ±5°, and the policy runs
**deterministically** — so the only thing separating one attempt from another is
that spawn heading.

Three consequences follow from that, and most of the work in this session came
from taking them seriously:

* **Consistency is worth nothing.** A policy that laps once in a thousand at
  13.9 s beats one that laps every time at 14.2 s.
* **There is no action noise to fish in at scoring time.** The distribution over
  attempts is induced entirely by the ±5° heading.
* **`--agents` and `--seed` are the competitor's to choose** and are replayed
  verbatim by verification (`verify.py:318-333`), so sampling more attempts is
  legitimate. The cap is 1000.

What you cannot change: the car (mass, motor, brakes, steering, tyres) and the
simulation rate. What you can change: `teamcode/env.py`'s three hooks
(observations, reward, terminations) and all of `teamcode/ppo_cfg.py`.

---

## 2. Two solutions were built. The second one won.

### Track A — a deterministic controller ("the teacher"), 14.900 s

A ~170-parameter racing line + speed profile + pure-pursuit controller, tuned by
CMA-ES against real lap time. Lives in `teacher/` and `tools/`. It reached
**14.900 s** and then stopped improving, for reasons that were measured rather
than guessed (`results/driven-line.png` shows it):

* **power-limited on the straights** — peak 5.21 m/s against a 6.92 m/s top
  speed, which matches the measured acceleration falling from 4.9 m/s² at rest
  to 0.29 at 6.75 m/s;
* **already at the grip limit in the corners** — 2.85–3.05 m/s through the
  R = 1.0 m corners against `sqrt(9.62 × 1.0) = 3.10`.

Both axes saturated, so it is not a tuning problem. Every route past the
quasi-static model was tried and measured worse: commanded sideslip and
counter-steer (monotonically worse), re-solved/widened/blended/rate-limited
lines (all failed to complete a lap), and physically honest profile constants
(also no lap). See `results/README.md` for the numbers.

**The teacher is not used by the RL policy at all** — not as a warm start, not
as reference observations. It sits at 14.900 s while plain PPO reaches 14.0, so
conditioning on it would install its ceiling in the policy's input space.

### Track B — PPO, 13.967 s

The submitted solution. `teamcode/env.py` + `teamcode/ppo_cfg.py`, trained with
RSL-RL on 3072 environments.

---

## 3. What actually made the RL policy fast

The baseline in `teamcode/` is deliberately a floor: 3 observations, one reward
term, a 3-second discount horizon, 50 iterations. Five changes mattered, in
rough order of value.

**(a) Observations 3 → 40.** The single biggest gap was that the baseline could
not see the track ahead — with only present-tense state a policy cannot know a
corner is coming, and the one safe strategy is to go slowly enough to correct
whatever appears. The vector is 16 scalars plus an 8-point curvature preview
(`car.lookahead`, reaching 0.25–6.2 m). Two entries are worth calling out:

* `car.steer_angle`, the **actual** wheel angle. The steering is an
  effort-limited servo that needs ~10 control steps to reach what it was asked
  for, so for a third of a second the command and the reality are different
  numbers. Feeding only the command side leaves the policy guessing at its own
  actuator state.
* `cos`/`sin` of the arc position. This is deliberately against the usual advice
  not to hand a policy its own position — but one track is raced, in one
  direction, from one point, and the preview describes *local* shape without
  saying that this particular corner feeds the 9.6 m straight and is worth
  sacrificing entry speed for.

**Observation width is locked by `--resume`.** Decide it once; changing it
throws away every hour already spent.

**(b) A convex lap-time bonus.** `distance + (18 − lap_time)² × 2` on completing
a clean lap. The curvature is the entire design: PPO maximises *expected*
return, which by default buys a careful policy that is good on average. A reward
convex in pace makes an expected-return maximiser prefer the gamble — with the
square, lapping at 14 s half the time beats lapping at 15 s every time, and a
linear bonus ranks those the wrong way round.

**(c) `gamma` 0.99 → 0.9965.** At 0.99 the horizon is ~3 s against a ~15 s lap,
so the policy cannot trade speed here for exit there — and cannot see a lap
bonus paid at the end of a lap at all.

**(d) Entropy annealed to 0** for the final phase, with `desired_kl` halved and
a lower learning rate. The policy is scored on its deterministic **mean**
action, and a high training noise floor makes that mean the *robust-under-noise*
optimum rather than the fastest one. This phase took mean reward 126 → 214 and
was worth roughly 0.3 s.

**(e) A reward for the launch.** This broke a two-checkpoint plateau and is the
subtlest thing here. `car.lap.last_time_s` times a **flying** lap, gate to gate.
A scored attempt is a **standing start** from the origin. The gap was 0.77 s —
a fifth of the scored run — and none of it was inside the objective, so the only
part of the run that begins at 0 m/s was optimised by the generic distance term
alone. Paying for `episode_time_s` on the first lap of a stint (which covers
standing start + out-lap + flying lap) collapsed the offset to 0.33 s.

Also: spawn jitter 0 → 0.02 rad. The SDK default is **zero**, which would train
every car on one identical pose and then score across ±5°.

---

## 4. Wrong turns, and what they cost

Recorded because each looked reasonable and each was caught by measuring.

| what | what happened |
|---|---|
| **Wheel-slip penalty** | `car.slip` is a *ratio* dividing by a ground speed clamped at `1e-3`, so it explodes at a standing start. Measured `penalty_slip: 921.83` against `distance: 0.0017`, mean reward −46 541. It would have taught the car that touching the throttle from rest is catastrophic, and presented as "PPO can't find a lap". |
| **Steering-rate penalty** | Measured 3–6× the reward it was shaping, and largest exactly when the policy is most exploratory. First lesson would have been "stop steering", on a car saturated 41% of a fast lap. Removed; the reward now has no shaping penalties at all. |
| **500-environment training** | Recommended by prior experience and 3.5× faster per iteration (22.1 s → 6.3 s), but it degraded *this* run monotonically: 14.300 → 14.567 → 14.667 → 14.700. Reverted to 3072. Kept at `results/rl/runs/500env-degraded/`. |
| **"`--agents` is not a lever"** | Concluded from a sweep that stopped at 300 agents and moved the best lap only 0.067 s. **Wrong.** At 1000 agents the same checkpoint gained 0.133 s, and 1000 vs 30 was the difference between 14.200 and 14.033. Do not extrapolate a tail from its flat segment. |
| **13.967 s at 2000 agents** | Achieved, then discarded — verification caps `--agents` at 1000. Had to be re-found within the limit. |
| **Rollover without a grace period** | The official rule gates *both* wall contact and rollover behind an 8-step spawn grace. Terminating on roll during the settle would have killed cars a scored attempt lets run. Fixed by calling `rules.official_terminations` instead of restating it. |
| **Instantaneous stall test** | The real rule needs 45 *consecutive* slow steps. An instantaneous test punishes hard braking into a hairpin — exactly the behaviour being trained for. |

---

## 5. Measurement traps that will bite you

**Lap times are quantised to 1/30 s.** The achievable values are 14.033, 14.067,
14.100, … so "under 14" means exactly **13.967**, and improvements arrive in
33 ms steps rather than continuously.

**TensorBoard's `Lap/best_lap_time_s` is not the score.** It times a flying lap;
the benchmark times a standing start. The offset is **not constant** — measured
0.43 s at 17.5 s pace and 0.77 s at 14.2 s pace, because a fixed launch cost
lands against an ever-shorter lap. A projection assuming a constant offset was
0.37 s optimistic.

**`--measure` (teacher tooling) used to reuse one simulator across files.** The
benchmark builds a fresh one. The identical file measured three times in one
process read 15.500, then 15.367, then 15.367 — so every file after the first
was scored on a simulator carrying the previous attempt's residue and read
~0.13 s fast. Fixed by fanning out to one cold process per file. Any figure in
an old note from a multi-file run is suspect.

**The fast tail is single-car, and that is the verification risk.** At 1000
attempts on the submitted checkpoint:

    13.967 s :  1 car     <- the submitted lap
    14.033 s :  5 cars
    14.067 s : 16 cars

A time held by one car in a thousand sits exactly where cross-machine PhysX
differences decide whether it reproduces. **A robust sub-14 needs the whole
distribution ~0.1 s faster — that is training, not sampling.** The final
checkpoint `model_1274` is better here: 2 cars at 13.967 and 6 at ≤14.000, on
two different seeds. It was not submitted; the earlier one was.

---

## 6. Publishing cannot be done from the GPU box

The vast.ai instance uploads at **546 B/s**. The lap itself posts fine — small
JSON — but the 5 MB policy bundle cannot finish inside the 120 s per-part
timeout, and every attempt reported *"the policy did not upload"*. That looks
like a successful submission and is not: without the bundle the lap can never be
verified.

`lituanicax_sdk.bundle` and `lituanicax_sdk.submit` are stdlib-only, and the SDK
fingerprint matches across machines (`e08cd1b71e18`), so build and publish from
a laptop:

```python
import json
from pathlib import Path
from lituanicax_sdk import bundle as B, submit as S

report = json.load(open("submission.json"))          # from the box
bd = B.build_bundle(report, "path/to/model.pt",
                    team="Vilnius Lyceum Carbotics",
                    project_root=Path("/path/to/ltxsim"))
out = S.submit(report, team="Vilnius Lyceum Carbotics", bundle=bd)
S.print_outcome(out, url="https://isaacleaderboard.netlify.app/")
```

The checkpoint's directory must contain a `params/` folder (`agent.yaml`,
`env.yaml`) — `build_bundle` looks for it next to the checkpoint.

---

## 7. Where everything is

```
teamcode/env.py            observations, reward, terminations — the solution
teamcode/ppo_cfg.py        PPO hyperparameters
teacher/, tools/           the deterministic controller and its measurement tools

results/
  rl/model_1125_13.967s_1k.pt   THE SUBMITTED POLICY
  rl/model_1274_13.967s_2cars.pt  better-supported, not submitted
  rl/model_*.pt                 milestone checkpoints, named by their scored time
  rl/params/                    agent.yaml + env.yaml, needed to bundle
  rl/runs/<timestamp>/          every checkpoint from each training run
  rl/runs/500env-degraded/      the run that got worse; kept as evidence
  teacher-best.json             the 14.900 s controller
  trace-best.npz                its scored lap, step by step
  driven-line.png               the path the car actually drives, by speed
  racing-line.png               reference line comparison
  teacher-artifacts/            dynamics/grip/headings probes and CMA-ES history
  logs/train.log                the full training log
  scripts/                      analysis scripts used along the way
  README.md                     measured numbers, in more detail than here
```

**Reproduce the score:**

```bash
export OMNI_KIT_ACCEPT_EULA=YES
python -u -m lituanicax_sdk.benchmark --headless --no-submit \
  --checkpoint results/rl/model_1125_13.967s_1k.pt --agents 1000 --seed 1
```

**Resume training** (needs a GPU box with Isaac Sim; `--resume` picks the newest
checkpoint in `logs/`, so stage the one you want):

```bash
python -u -m teamcode.train --resume --num_envs 3072 --headless \
  agent.max_iterations=1200 agent.algorithm.entropy_coef=0.0 \
  agent.algorithm.desired_kl=0.005 agent.algorithm.learning_rate=2e-4
```

---

## 8. If you pick this up, do this next

1. **Train more.** It was still improving when the box was terminated — mean
   reward rising, mean flying lap 14.31 → 13.69 across the final phase. The
   remaining gap to a *verifiable* sub-14 is ~0.1 s of distribution shift.
2. **Submit `model_1274` instead**, or better, whatever a longer run produces.
   Same 13.967 s but held by 2 cars rather than 1, which is what decides
   verification.
3. **Do not chase `--agents` or seeds.** At the 1000 cap the seed spread is a
   single 33 ms step, and the tail is single-car. That well is dry.
4. **Leave the observation width alone** unless you intend to restart from
   scratch — `--resume` is width-locked.

---

## 9. Things found in the SDK, reported not exploited

While auditing verification, a subagent found that `_code_digest`
(`_locked.py:188-214`) hashes class attribute *names* but never their *values*,
so class-level constants are invisible to the runtime fingerprint — setting
`AttemptTimer.MIN_TRAVEL_FRACTION = 0.0` would make a ~4.2 m out-and-back score
as a lap and would survive verification because the hash does not move. It also
found `state`, `spawn` and `tracks` missing from `_RUNTIME_CRITICAL`.

These were **not used**. Using them would mean submitting a lap the car never
drove, and it would verify clean precisely because the check is broken. They are
written down here so they can be reported to the organisers. The suggested fix
is a scalar case in `_code_digest`'s class branch, plus the three missing
modules.
