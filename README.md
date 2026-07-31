# LituanicaX — Isaac Sim Challenge

A racing SDK. Every team trains a policy to drive the **same** MuSHR nano v2 RC
car round a cone-lined track in **NVIDIA Isaac Sim 5.1**, as fast as it can, and
the lap times are directly comparable.

Reinforcement learning with **Isaac Lab** and **RSL-RL**. The baseline trains on
an **RTX 3070 (8 GB)** with **200 cars in parallel** in roughly **6 hours**
(1000 PPO iterations), and the trained policy runs at 30 Hz.

## Quick start

```bash
./install.sh                 # one-time setup, see Installation
train --num_envs 200 --headless   # train
play --num_envs 1                 # watch the newest checkpoint drive
evaluate                          # official score: 100 agents, best lap
```

## The two halves

The repository is split so that teams can change everything that makes a policy
better, and nothing that makes the *car* faster.

```
lituanicax_sdk/    LOCKED. The car, the ground, the physics, the lap clock —
                   and every measurable property of the car, exposed to you.
                   It ships no rewards and no observations, deliberately.
├── env.py           RaceEnv + its config: the simulation loop
├── scene.py         one-time USD scene building
├── vehicle.py       the car; dynamics.py the motor model
├── state.py         CarState — what your code reads
├── timing.py        the lap clock;  rules.py the crash rules
├── track.py         tracks;  tracks/ the official one
└── evaluate.py      ★ the official score

team_solution/     YOURS. This is what you edit.
├── env.py         ★ Start here: what the policy sees, what it is paid for,
│                    when an episode ends, and where the cars drive.
├── ppo_cfg.py     PPO settings: network size, learning rate, batch size.
├── train.py         Run this to train.  --num_envs  --headless  --resume
├── play.py          Watch a policy, and export it as policy.pt / policy.onnx
│                    for the real car.   --num_envs  --checkpoint  --video
└── tracks/        Drop your own tracks in here.
```

**[`docs/SDK.md`](docs/SDK.md) is the reference** — what is locked, what is not,
and everything `CarState` can tell you about the car.

```
tests/             Runs on the CPU in seconds, without Isaac Sim.

install.sh         One-shot setup script (see Installation)
pyproject.toml     Dependencies and tool settings
logs/              One folder per training run (see below)
Track.blend        Blender source for the official track
IsaacLab/          Isaac Lab, pinned to v2.3.0 — not our code
```

## What is locked, in one table

| Locked | Yours |
|---|---|
| The car: mass, motor, brakes, steering, tyres | Observations — any signal, any number of them |
| Physics at 120 Hz, policy at 30 Hz | Rewards |
| Actions: throttle and steering, both `[-1, 1]` | Terminations |
| What counts as a crash *when a lap is measured* | Spawn points and curriculum |
| Lap timing | Tracks, extra objects and sensors |
| The official track | The RL algorithm and everything about it |

Try to change a locked value and you get an error saying so, including through
a Hydra command-line override. Details and rationale in
[`docs/SDK.md`](docs/SDK.md).

Every run is self-contained in a single timestamped folder, so you can copy,
compare or delete a run by moving one directory:

```
logs/2026-06-28_01-07-24/
├── model_0.pt … model_200.pt   checkpoints, saved every 10 iterations
├── events.out.tfevents.*       TensorBoard data
├── params/env.yaml             the exact task config this run used
├── params/agent.yaml           the exact PPO config this run used
├── git/                        a snapshot of the code at the time
├── exported/                   policy.pt + policy.onnx, written by play.py
└── videos/                     only if you passed --video
```

## The task

The car has to drive around a closed track as fast as it can (top speed
~6.7 m/s) without touching the walls or flipping over. An episode lasts up to
90 seconds (5400 policy steps) and ends early on a wall hit, a roll-over, or if
the car stops making progress.

Cars start from 5 preset points on the track, each of which can be used facing
either direction, so a single policy learns to drive the track both ways.

### What the policy sees — 23 numbers

This is the *baseline*, not a requirement. Observations are entirely yours —
add signals, remove them, rescale them; the policy's input size follows.

| Index | Signal | Scaled by |
|-------|--------|-----------|
| 0 | `car.wheel_speed` | top speed (6.7 m/s) |
| 1 | `car.speed_forward` | top speed |
| 2 | `car.speed_lateral` — drift | top speed |
| 3 | `car.yaw_rate` | 10 rad/s |
| 4 | `car.cross_track_error` | 0.3 m |
| 5 | `car.heading_error` | π |
| 6 | `car.dist_to_next_corner` | 5 m |
| 7 | `car.next_corner_curvature` | 10 m⁻¹ |
| 8–22 | `car.lookahead(...)` — 5 points: (x, y) in the car's own frame + curvature | 5 m / 10 m⁻¹ |

The lookahead points sit roughly 0.5, 1, 2, 3.5 and 5 m ahead. Which way the car
is going is worked out from its velocity, so every track-relative signal mirrors
itself automatically when the car drives the other way round.

Also on `car` and not used by the baseline: `dist_to_wall`, `steer_angle`,
`wheel_speeds` and `wheel_slips` (per wheel), `suspension_travel`,
`applied_wheel_torque`, `progress_m`, the raw `joint_pos` / `joint_vel`, and
the whole locked vehicle spec. See [`docs/SDK.md`](docs/SDK.md).

### What the policy controls — 2 numbers

- **Throttle** `[-1, 1]` — positive accelerates, negative brakes
- **Steering** `[-1, 1]` — scales to ±0.488 rad (±28°) of steering angle

These two are locked. Every team's policy drives the car the same way.

### How it is rewarded

This is the *baseline*, written out in `team_solution/env.py`. The SDK ships no
rewards at all — one it wrote would be one every team shared. The principle
worth keeping even if you replace the rest: only covering ground quickly earns
real reward, and everything else is a small penalty shaping *how*.

| Term | Weight | What it does |
|------|--------|--------------|
| Forward distance | 4.0 | The main reward: `forward_speed / max_speed × dt` |
| Alive bonus | 0.02 | A constant trickle for staying on the track |
| Steering usage | −0.003 | Discourages steering beyond a ±0.05 deadzone |
| Steering rate | −0.003 | Discourages jerky steering |
| Throttle rate | −0.002 | Discourages jerky throttle |
| Wheel slip | −0.03 | Discourages spinning or locking the wheels |
| Body roll | −0.1 | Discourages leaning, which precedes a roll-over |

### When an episode ends

- **Wall hit** — the car's centre comes within 0.15 m of any wall
- **Flipped** — the car is no longer upright, roll beyond ~73°
- **Stalled** — forward speed under 0.27 m/s, after step 45
- **Time out** — the full 90 s elapses (`episode_length_s`)

All three are plain tensor code in `team_solution/env.py`, and that is the file
you edit to change how the task behaves. Nothing above is imposed by the SDK —
during training you decide entirely what ends an episode. Only when a lap is
being *measured* does the SDK add its own crash rules on top, so that a recorded
lap is always a clean one.

### Lap times

Lap timing is built into the SDK and is the same for everyone. A lap runs from
the track's start/finish line back to itself — anchored to the *track*, not to
wherever you spawn your cars, which is what makes two teams' times comparable.

The line spans the width of the track, so a wide racing line crosses it just
like a car on the centerline. Three rules keep it honest: the car has to leave
the gate window before a crossing counts, it has to reach the far side of the
loop (so nudging over the line and reversing back is not a lap), and the first
crossing after a spawn only starts the clock. A lap during which the car
touched a wall does not count.

During training you get `Lap/best_lap_time_s`, `Lap/mean_lap_time_s` and
`Lap/laps_per_min` in TensorBoard.

### The official score

```bash
evaluate                              # 100 agents, one attempt each
evaluate --agents 100 --batch-size 25 # lower the batch if 25 cars will not fit
evaluate --spawn-presets              # the track's preset points, not (0, 0)
```

One hundred agents each get a **single attempt**: an agent is placed on the
official track and drives until it completes a lap — after which it is
terminated on the spot — or fails, by crashing into a wall, rolling over, or
stalling. **The fastest of the hundred laps is the team's submission.** Not the
mean, not the median: a team is judged on the best lap it can produce, the way
a qualifying session works.

Every agent starts at the **world origin (0, 0)**, facing along the track, so
all hundred attempts begin from the same point. The origin sits on the
centerline a little before the start/finish line, so an attempt is a short
out-lap to the line plus one timed lap back to it. Pass `--spawn-presets` to
start from the track's preset points instead.

The evaluation runs under the SDK's own rules and **ignores your
`compute_terminations`**, so every team's cars fail for the same reasons: the
crash rules, plus a stall rule that cuts off a car that stops making progress
(a car pinned against a wall sits there otherwise). It lives in
`lituanicax_sdk/evaluate.py` because it is the scoring rules, not a project
script. The result is printed and written to `submission.json` next to the
checkpoint.

## Training

```bash
train --num_envs 200 --headless
```

`--headless` skips on-screen rendering, which trains considerably faster, and
`--resume` continues from the most recent checkpoint. Those three flags are all
`train` has.

How long it trains and every other learning setting lives in
`team_solution/ppo_cfg.py`. To change one for a single run without editing the
file, pass it through: `train --num_envs 200 agent.max_iterations=200`.

Each run writes to `logs/<timestamp>/`. `team_solution/train.py` starts
TensorBoard for you and prints the URL (port 6006, or the next free one). You
can also start it yourself:

```bash
tensorboard --logdir logs
```

Metrics worth watching:

| Metric | What it means |
|--------|---------------|
| `Train/mean_reward` | Average episode reward — should climb |
| `Train/mean_episode_length` | How long cars survive (5400 = a full episode) |
| `Train/mean_lr` | Adaptive learning rate (falls as KL rises) |
| `Info/kl` | How much the policy changed — should hover near 0.01 |

### Playing a checkpoint

```bash
play                                # the newest checkpoint
play --num_envs 1                   # one car
play --checkpoint logs/<timestamp>/model_1000.pt
play --video                        # record a clip into the run folder
```

Playback always runs at wall-clock speed — it is for watching — and
`--enable_cameras` is set for you, since Isaac Sim will not produce frames
without it and video recording fails with a confusing error.

`team_solution/play.py` also exports the policy into the same run folder, at
`logs/<timestamp>/exported/`, as `policy.pt` (TorchScript) and `policy.onnx`
(ONNX), ready to run on the real car.

## PPO hyperparameters

All of these live in `team_solution/ppo_cfg.py`, and all of them are yours.

| Parameter | Value |
|-----------|-------|
| Actor hidden layers | [512, 256] (Mish) |
| Critic hidden layers | [512, 256] (Mish) |
| Initial noise std | 0.8 |
| Learning rate | 1e−4 (adaptive) |
| PPO epochs | 6 |
| Mini-batches | 6 |
| GAE λ | 0.95 |
| Discount γ | 0.9965 |
| Entropy coefficient | 0.004 |
| Clip parameter | 0.2 |
| Desired KL | 0.01 |
| Max gradient norm | 0.75 |
| Steps per env per iteration | 1600 |

## Installation

You need an **NVIDIA GPU** with at least 8 GB of VRAM (tested on an RTX 3070),
an **NVIDIA driver in the 550–580 series**, and **uv** (a Python package
manager).

```bash
# 1. Install uv, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo
git clone git@github.com:MCiuzelis/LituanicaX_IsaacSimChallenge.git
cd LituanicaX_IsaacSimChallenge

# 3. Run the install script
chmod +x install.sh
./install.sh
```

`install.sh` does the following:

1. **Driver check** — warns if the NVIDIA driver is ≥ 590, which is known to
   crash Isaac Sim 5.1 during RTX start-up, and exits if no driver is running.
2. **Python 3.11** — installs it via `uv python install 3.11` if needed.
3. **Isaac Lab** — checks out the `IsaacLab/` tree at tag `v2.3.0`, the only
   version that resolves cleanly against Isaac Sim 5.1.
4. **Virtual environment** — `uv venv --python 3.11` then `uv sync`, installing
   everything from the lockfile into `.venv/`.
5. **Patches** — reinstalls the GUI build of `opencv-python` (Isaac Sim pulls in
   the headless one) and pins `numpy<2.0.0` and `setuptools<82.0.0` for ABI and
   import compatibility.
6. **Helper commands** — writes `train` and `play` into `~/.local/bin/`, which
   `cd` into the project and run `uv run python <name>.py "$@"`. On dual-GPU
   laptops they also set `__NV_PRIME_RENDER_OFFLOAD=1` so Vulkan uses the
   NVIDIA card.

Then check it worked:

```bash
train --help
```

## Spawn direction balancing

Because each start point is used in both directions, one direction can end up
easier than the other and dominate the learning signal. To prevent that, the
environment tracks a rolling average episode length per direction and spawns
more cars into whichever direction is currently doing *worse*. As the two even
out, the split converges back to roughly 50/50. See `SpawnManager` in
`lituanicax_sdk/spawn.py` — it is the default, not a rule, and you can swap it
for your own by setting `spawn_manager` in `team_solution/env.py`.

## Tests

The parts of the SDK that do not need Isaac Sim — the lap timer, the track
maths, `CarState`, every observation and reward term, and the locks — are
tested on the CPU in seconds:

```bash
.venv/bin/python -m pytest tests/ -q
```

Worth running after editing `team_solution/`: a term that returns the wrong
shape, or an observation that stops being finite at speed, is caught here
rather than an hour into training.
