# LituanicaX — Isaac Sim Challenge

A racing SDK. Every team trains a policy to drive the **same** MuSHR nano v2 RC
car round a cone-lined track in **NVIDIA Isaac Sim 5.1**, as fast as it can, and
the lap times are directly comparable.

Reinforcement learning with **Isaac Lab** and **RSL-RL**. The baseline trains on
an **RTX 3070 (8 GB)** with **200 cars in parallel**, and the trained policy runs
at 30 Hz.

This file is the whole manual: install, the split between the SDK and your code,
everything `CarState` can tell you, how a lap is timed, and how a submission is
scored.

```bash
./install.sh                      # one-time setup
train --num_envs 200 --headless   # train
play  --num_envs 1                # watch the newest checkpoint drive
benchmark                         # official score: best of ten laps
```

---

## Contents

- [Installation](#installation)
- [The two halves](#the-two-halves)
- [What is locked](#what-is-locked)
- [The task](#the-task)
- [Writing your environment](#writing-your-environment)
- [What `car` gives you](#what-car-gives-you)
- [The baseline solution](#the-baseline-solution)
- [Terminations, and the two modes](#terminations-and-the-two-modes)
- [Where cars start](#where-cars-start)
- [Lap timing](#lap-timing)
- [The official score](#the-official-score)
- [The leaderboard](#the-leaderboard)
- [Training](#training)
- [Playing a checkpoint](#playing-a-checkpoint)
- [Runs and logs](#runs-and-logs)
- [Tracks](#tracks)
- [Extra objects and sensors](#extra-objects-and-sensors)
- [Integrity](#integrity)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

---

## Installation

You need:

- an **NVIDIA GPU** with at least 8 GB of VRAM (tested on an RTX 3070)
- an **NVIDIA driver in the 550–580 series** — see below, this one matters
- **Ubuntu 24.04** (what everything here is tested on)
- **uv**, a Python package manager

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

Then check it worked:

```bash
train --help
```

### The driver version really matters

Isaac Sim 5.1.0 needs an NVIDIA driver in the **550–580** series. Driver 590 and
newer crash inside `librtx.scenedb.plugin.so` while the RTX engine starts up —
a known incompatibility, and not something this repository can work around.
`install.sh` checks and warns, but it will not stop you.

Tested working: 550.163.01 and 580.159.03. To downgrade on Ubuntu 24.04:

```bash
sudo apt-get install --reinstall nvidia-driver-550-open
# reboot afterwards
```

On kernels 6.17 and newer the 550/570 meta-packages pull in the 580 kernel
module automatically. That is fine — only the userspace libraries matter.

If `nvidia-smi` reports nothing at all, install a driver first:

```bash
sudo apt-get install nvidia-driver-570-open
# reboot, and if Secure Boot is on, enrol the MOK key at the blue EFI prompt
```

### What `install.sh` does

1. **Driver check** — warns if the NVIDIA driver is ≥ 590, exits if none is
   running.
2. **Python 3.11** — installs it through `uv python install 3.11` if it is
   missing. The project is pinned to 3.11; Isaac Sim's wheels do not exist for
   anything newer.
3. **Isaac Lab** — initialises the `IsaacLab/` submodule and checks it out at
   tag **v2.3.0**. That is the only version that resolves cleanly against Isaac
   Sim 5.1; newer refs pull a Starlette pin that fights Isaac Sim's FastAPI
   stack.
4. **Virtual environment** — `uv venv --python 3.11` then `uv sync`, installing
   everything from the lockfile into `.venv/`.
5. **Two ABI patches** — reinstalls the GUI build of `opencv-python` (Isaac Sim
   pulls in the headless one, which has no `imshow`), pins `numpy<2.0.0`
   (Isaac Sim's compiled extensions ship against 1.26) and `setuptools<82.0.0`
   (82 dropped `pkg_resources`, which TensorBoard still imports).
6. **Helper commands** — writes `train`, `play` and `benchmark` into
   `~/.local/bin/`. Each one `cd`s into the project and runs the matching module
   through `uv run`, so they work from any directory. On dual-GPU laptops they
   also set `__NV_PRIME_RENDER_OFFLOAD=1`, which forces Vulkan onto the NVIDIA
   card instead of the Intel iGPU that drives the display.

`train` and `play` live in `teamcode/` because they are yours; `benchmark`
lives in the SDK because it is the scoring rules, not a project script.

Everything the helpers do, you can do by hand:

```bash
uv run python -m teamcode.train --num_envs 200 --headless
uv run python -m teamcode.play --num_envs 1
uv run python -m lituanicax_sdk.benchmark --headless
```

---

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
├── spawn.py         where cars start; the default is (0, 0)
├── runs.py          where runs live, finding checkpoints
├── submit.py        publishing a lap, and the policy that set it
├── bundle.py        that policy, zipped: checkpoint + teamcode + config
├── verify.py        the organisers' side: re-run a published lap and report
└── benchmark.py     ★ the official score

teamcode/          YOURS. This is what you edit.
├── env.py         ★ Start here: what the policy sees, what it is paid for,
│                    when an episode ends, and where the cars start. It is
│                    deliberately the least that works — a floor to beat.
├── ppo_cfg.py     PPO settings: network size, learning rate, batch size.
├── train.py         Run this to train.  --num_envs  --headless  --resume
├── play.py          Watch a policy, and export it as policy.pt / policy.onnx
│                    for the real car.   --num_envs  --checkpoint  --video
└── tracks/        Drop your own tracks in here.

tests/             Runs on the CPU in seconds, without Isaac Sim.
install.sh         One-shot setup script
pyproject.toml     Dependencies and tool settings
logs/              One folder per training run
Track.blend        Blender source for the official track
IsaacLab/          Isaac Lab, pinned to v2.3.0 — not our code
```

The rule of thumb for the split: **if changing it makes the car faster rather
than the policy better, it is locked.**

---

## What is locked

| Locked | Why |
|---|---|
| The MuSHR nano v2: mass, wheels, USD, scale | Everyone drives the same car |
| Motor torque curve, brakes, steering limits | Same acceleration, same braking |
| Ground and tyre friction | Same grip |
| Policy rate (30 Hz) and decimation (4, so 120 Hz physics) | Lap times are counted in policy steps |
| The action space: throttle and steering, both `[-1, 1]` | One control interface |
| Crash rules *during a measured lap* | A recorded lap is always a clean lap |
| Lap timing | The number the competition is about |
| The official track and its start/finish line | A lap means the same thing for everyone |

And what is **yours**: observations · reward · terminations · where cars start
and the whole curriculum · episode length · tracks · extra objects and sensors ·
number of cars · the RL algorithm, network and hyperparameters · the trainer
itself.

Try to change a locked value and you get an error naming the parameter, not a
silently different simulation:

```python
>>> from lituanicax_sdk import VEHICLE
>>> VEHICLE.drive_torque_nm = 0.5
LockedParameterError: VehicleCfg.drive_torque_nm cannot be changed.
This parameter is fixed by the competition rules so that every team's car
behaves identically — see README.md for what you can change instead.
```

The same applies to Hydra overrides on the command line: `train env.sim.dt=0.001`
is rejected when the environment is built.

The clock is two numbers, `TimingCfg.policy_hz` and `TimingCfg.decimation` in
`lituanicax_sdk/vehicle.py`. Everything else about it — `physics_hz`,
`physics_dt`, `step_dt`, `render_interval` — is derived from those, so whoever
runs the competition changes the rate in one place and nothing falls out of step
with it. Teams read the values and never set them.

You can **read** every locked value, and you should. That is the other half of
the deal: the car is fixed, so the SDK holds nothing back about it.
`car.max_speed_m_s`, `car.wheel_radius_m`, `car.max_steer_rad`, `car.mass_kg`,
`car.drive_torque_nm`, `car.step_dt` — and `car.vehicle` for the whole spec,
down to the brake fade constants. You should never have to hardcode `6.7` or
`0.037`.

The SDK ships **no rewards and no observations**. Not as an oversight — a reward
the SDK wrote would be a reward every team shared, and picking weights off a
menu is not the same exercise as deciding what matters.

---

## The task

The car drives round a closed 50 m track as fast as it can (top speed ~6.7 m/s)
without touching the walls or flipping over. An episode lasts up to 60 seconds
(1800 policy steps) and ends early on a wall hit, a roll-over, or when the car
stops making progress.

The official track is 0.70 m wide with the centerline running exactly down the
middle, and it is lined with cones. The cones you see are scenery; the crash
boundary is an invisible wall mesh in the same place, and the car is "touching a
wall" when its centre comes within 0.15 m of it.

---

## Writing your environment

Subclass `RaceEnv` and implement three methods. That is the whole interface.

```python
import torch
from lituanicax_sdk import RaceEnv

class TeamEnv(RaceEnv):
    def compute_observations(self, car):        # -> [num_envs, observation_space]
        return torch.stack([
            car.speed_forward / car.max_speed_m_s,
            car.speed_lateral / car.max_speed_m_s,
            car.yaw_rate / 10.0,
            car.cross_track_error / 0.3,
            car.heading_error / math.pi,
        ], dim=-1).clamp(-1.0, 1.0)

    def compute_reward(self, car):              # -> [num_envs]
        forward = car.speed_forward / car.max_speed_m_s * car.step_dt
        slip = car.slip.clamp(0.0, 1.0)
        self.log("Rewards/forward", forward)    # traced in TensorBoard
        self.log("Rewards/slip", -slip)
        return 4.0 * forward - 0.03 * slip

    def compute_terminations(self, car):        # -> [num_envs] bool, optional
        return car.wall_touched | (car.up_axis < 0.3)
```

Declare the observation width in your config — it sizes the policy's input
before the simulation starts, so it cannot be discovered:

```python
@configclass
class TeamRaceEnvCfg(RaceEnvCfg):
    observation_space: int = 5
```

The first step checks that `compute_observations` really returns that width and
says so plainly if it does not.

`self.log(name, value)` records a scalar per step, averaged over the training
iteration. Trace each part of your reward: when a policy does something strange,
the fastest way to find out why is to see which term paid for it.

`teamcode/env.py` is a worked example — 3 observations, 1 reward term, 2
terminations — that trains as-is, gets round the track slowly, and is yours to
rewrite.

### One constraint worth knowing

`scene.env_spacing` must be `0`. Every car drives the *same* shared track rather
than its own copy, so they all occupy the same space and the track is loaded
once. Cross-track error, lookahead and lap timing are all measured in world
coordinates against that one track. A non-zero spacing is rejected at start-up
rather than silently measuring most of your cars against a track that is no
longer under them.

---

## What `car` gives you

`CarState`, one row per car, in raw SI units. How you scale it is your business.

**Motion:** `speed_forward` `speed_lateral` `speed_ground` `yaw_rate`
`lin_vel_b` `lin_vel_w` `ang_vel_b`

**Pose:** `pos_w` `pos_xy` `quat_w` `yaw` `roll` `pitch` `up_axis` `forward_xy`

**Drivetrain:** `wheel_omega` `wheel_speed` `slip` `steer_angle` `wheel_speeds`
`wheel_slips` `applied_wheel_torque` `suspension_travel` (the per-wheel arrays
are `[N, 4]`, ordered as `wheelbase_names`)

**Raw joints:** `joint_pos` `joint_vel` `joint_names` — everything, if the
shaped readouts are not enough

**Commands:** `throttle_cmd` `steer_cmd` and their `_prev`

**On the track:** `nearest_idx` `cross_track_error` `signed_cross_track_error`
`heading_error` `going_forward` `direction_sign` `progress_m`
`dist_to_next_corner` `next_corner_curvature` `dist_to_wall`
`lookahead(offsets)` `track`

**Episode and lap:** `episode_step` `episode_time_s` `wall_touched` `lap.count`
`lap.last_time_s` `lap.best_time_s` `lap.just_finished`

**The locked car, readable in full:** `max_speed_m_s` `wheel_radius_m`
`max_steer_rad` `mass_kg` `drive_torque_nm` `max_wheel_omega` `step_dt`, and
`car.vehicle` for the rest of the spec.

**Your sensors:** `car.sensors["<name>"]` for anything in
`cfg.extra_scene_entities`.

`CarState` is a view: it hands out numbers, never the articulation they came
from, so your code can read the simulation but cannot reach in and change it.
Everything is computed once per step and cached, so reading `nearest_idx` in six
places costs what reading it once costs.

Every track-relative quantity mirrors itself for cars driving the other way
round, which is what lets one policy learn the track in both directions. A car
that is not moving has no direction of travel, so below 0.2 m/s the direction
comes from where the car is *pointing* instead — otherwise every standing start
would begin with a mirrored view of the track.

---

## The baseline solution

Everything in this section is `teamcode/env.py`. None of it is a
requirement; it is a worked example that trains.

It is **deliberately the least that works**: three observations, one reward
term, two terminations, every car starting from the same point. It learns to get
round the track and it is slow. That is the floor. Beating it is the exercise,
and the fastest way to is to add to it rather than to adjust it.

Worth being precise about **why** it is slow, because it is not the reward. The
reward already pays for speed — distance per step, summed over a fixed episode,
*is* average speed. It is slow because of what it cannot see and how little of
the future it is trained to care about:

- the policy is told where the car is *now* and nothing about where the track
  goes next, so it cannot prepare for a corner, only react once it is in one;
- `gamma` in `teamcode/ppo_cfg.py` is 0.99 — a horizon of about 3 s at 30 Hz,
  enough to see a wall coming, nowhere near the ~10 s of a lap, so it cannot
  learn to give up speed here for a better exit there;
- the networks are 64×64, and every car starts from the same point.

A policy with no preview has exactly one safe strategy: go slowly enough to
correct whatever appears. Give it a lookahead, lengthen the horizon, and it
stops needing to.

### What the policy sees — 3 numbers

| Index | Signal | Scaled by |
|-------|--------|-----------|
| 0 | `car.speed_forward` | top speed (6.7 m/s) |
| 1 | `car.signed_cross_track_error` | 0.3 m |
| 2 | `car.heading_error` | π/2 |

`car.lookahead([10, 20, 40, 70, 100])` returns the centerline ahead in the car's
own frame with the curvature at each point, and is the single biggest thing
missing. The offsets are counted in centerline points, not metres, so how far
ahead they actually sit depends on how densely that stretch of centerline was
exported — worth knowing before you tune them.

Also on `car` and unused by the baseline: `wheel_speed` and `slip`,
`speed_lateral` and `yaw_rate`, `dist_to_wall` (it is blind to how close the
walls are), `dist_to_next_corner` and `next_corner_curvature`, the last action,
`steer_angle`, the per-wheel arrays, `suspension_travel` and `progress_m`.

### What the policy controls — 2 numbers

- **Throttle** `[-1, 1]` — positive accelerates, negative brakes
- **Steering** `[-1, 1]` — scales to ±0.488 rad (±28°) of steering angle

These two are locked. Every team's policy drives the car the same way.

### How it is rewarded

One term: **how far it got.**

```python
car.speed_forward / car.max_speed_m_s * car.step_dt
```

The principle worth keeping even if you replace everything else: **only covering
ground quickly earns real reward**, and anything else belongs as a small penalty
shaping *how*. A term that pays for something other than progress tends to get
farmed — the classic failure is an alive bonus big enough that stopping in a
safe corner beats racing.

There are no penalties here at all, and it shows: the car saws at the steering,
spins its wheels off the line and leans into corners hard enough to lift a
wheel. Each of these is a term to add, an order of magnitude below the distance
term:

| Candidate | Reads |
|-----------|-------|
| Wheel slip | `car.slip` — spinning or locking the wheels |
| Body roll | `car.roll` — leaning, which precedes a roll-over |
| Steering rate | `car.steer_cmd - car.steer_cmd_prev` — jerky steering |
| Throttle rate | `car.throttle_cmd - car.throttle_cmd_prev` |
| Steering usage | `car.steer_cmd` beyond a small deadzone |

Log each one separately with `self.log(...)`: when the policy does something
strange, that is the fastest way to see which term paid for it.

**Lap time is never a reward.** It is measured, and a measurement the policy can
influence is not a measurement. To reward progress use `car.speed_forward` or
`car.progress_m`.

---

## Terminations, and the two modes

While you **train**, `compute_terminations` is the only thing that ends an
episode early. Nothing is forced on you. Ending an episode when a car has
stopped saves samples; ending one when it drifts wide teaches a tighter line and
may also teach timidity. Your call. The baseline ends on a wall hit and on a
stall — below 0.04 of top speed once the car has had 45 steps to get going.

The stall rule earns its place: a car that shuffles back and forth on the spot
earns almost nothing and never crashes, so without it that is a *stable* thing
for the policy to settle into, and it will. Terminating gives the state a value
of zero, which makes standing still strictly worse than driving. Note that the
baseline reads the speed instantaneously, where the SDK's own rule wants 45
*consecutive* slow steps: a jittering car clears the threshold half the time, so
a consecutive count never fires.

A roll-over (`car.up_axis < 0.3`) still sits out the rest of its episode
producing nothing, which is free training speed to reclaim.

While a lap is being **measured** (`benchmark`, or `enforce_official_rules`), the
SDK's crash rules apply *instead of* yours and a car that hits a wall freezes,
so every team's cars fail for exactly the same reasons:

| | |
|---|---|
| Wall contact | centre within 0.15 m of a wall → freeze, episode ends |
| Roll-over | up-axis below 0.3, about 73° |
| Stall (measured runs only) | barely moving for 45 consecutive steps → episode ends |

The stall rule is separate because whether a stalled car is worth terminating is
a training decision (the baseline terminates on it, with its own threshold).
During a measured run it is on, so a car pinned against a wall fails in a moment
instead of sitting out the whole attempt window.

Note that the crash radius is measured from the car's **centre**. A car that
wedges its nose against a wall can sit just outside 0.15 m and never register as
a crash — the stall rule is what ends that attempt.

One rule holds in **both** modes: **a lap during which the car touched a wall is
not a valid lap.** That is a property of the lap rather than a termination, so
it applies however you end your episodes — which is why you are free to let a
crashed car keep driving and learn to recover.

---

## Where cars start

Where cars start is **yours**, and it is not a property of the track.

The SDK's default is deliberately the dullest one there is: `SpawnManager` puts
every car on the **world origin**, facing along the track — one hardcoded point,
the same one `benchmark` scores from, with no opinion about which corners are
worth practising. **The baseline takes that default**, so it only ever practises
the track from one place and in one direction, and the corners it keeps losing
get no more attention than the ones it has learned.

`PresetSpawnManager(points=[...])` spreads cars over a list of poses you choose,
in world metres, each usable facing either way. Set `cfg.spawn_manager` to one
in `teamcode/env.py` — or subclass either manager and do something else
entirely. Randomising along the centerline, or starting cars at speed rather
than from rest, are both reasonable things to try.

Since a scored attempt always starts at (0, 0), it is worth keeping a start
there among whatever else you train on.

### Spawn direction balancing

If each preset is used in both directions, one direction can end up easier than
the other and dominate the learning signal. `PresetSpawnManager` tracks a
rolling average episode length per direction and spawns more cars into whichever
is currently doing *worse*. As the two even out, the split converges back to
roughly 50/50. Machinery, not a rule — turn it off with
`balance_directions=False`.

What you cannot move is where the lap clock starts and stops.

---

## Lap timing

Lap timing is built into the SDK and is the same for everyone. There are two
timers, anchored differently on purpose.

### While training — `LapTimer`

A lap runs from the **track's** start/finish line back to itself. That line is a
plane through one centerline point, at right angles to the track, and it spans
the full width of the track, so a wide racing line crosses it just as a car on
the centerline does. Anchoring to the track rather than to wherever you spawn is
what makes two teams' times comparable when they train from different places.

Three rules keep the numbers honest:

- **Arming** — the car must leave the gate window (2 m of track either side of
  the line) before a crossing counts, so sitting on the line does not tick over
  laps.
- **Travel** — a *timed* lap must reach the far side of the loop, so nudging
  over the line and reversing back is not a lap.
- **Out-lap** — the first crossing after a spawn only starts the clock. Cars
  start wherever they start, so that first partial loop is never timed. The
  travel rule does not apply to it: a car spawned a few metres before the line
  has genuinely reached the line, and the timed lap that follows still has to go
  all the way round.

Both directions round the track are timed. In TensorBoard you get
`Lap/best_lap_time_s`, `Lap/mean_lap_time_s` and `Lap/laps_per_min`. The first
two only appear once some car has completed a lap — before that there is nothing
to average, and a placeholder would just be a misleading flat line.

### Why your episodes have to be longer than a lap

The out-lap rule sets a floor on `episode_length_s`, and it catches people out.
A car has to cross the line **twice** before anything is recorded: once to end
the out-lap and start the clock, once to close the timed lap. From the origin
that is 14.7 m to the line plus a full 50 m lap — **65 m before the first
recordable lap**, which at 1.5–2 m/s is 30–45 s.

The lap timer is reset on *every* episode boundary, including a timeout, so a
too-short episode does not accumulate progress across episodes — it restarts the
out-lap each time. The failure mode is silent and easy to misread: the cars
visibly drive laps in the GUI and `benchmark` reports times, while
`Lap/best_lap_time_s` never appears in TensorBoard at all. `benchmark` is
unaffected because `AttemptTimer` has no out-lap to pay for.

If you want short episodes, spawn cars a couple of metres *before* the line
(just outside the 2 m gate window) so the out-lap costs almost nothing. Spawning
*on* the line is the worst case, not the best: the car is inside the window, has
to leave it to arm, and its first crossing a full lap later is the out-lap.

### While being scored — `AttemptTimer`

A scored attempt is one car from one known place, so it can do better: the gate
is the plane through the **spawn point**, and the clock starts the instant the
car is put down. One lap of driving for one lap of time, with no untimed
out-lap, and the attempt ends the moment the car comes back over the line it
started on. Comparability survives because `benchmark` fixes the spawn point for
everyone.

The arming and travel rules apply here too.

---

## The official score

```bash
benchmark                              # ten cars, one timed lap each
benchmark --headless                   # no window, and quicker for it
benchmark --checkpoint logs/<run>/model_1000.pt
benchmark --spawn 3.9 3.3              # start somewhere other than (0, 0)
benchmark --spawn-yaw 180              # face the other way round the track
benchmark --agents 1 --spawn-jitter 0  # one car, exactly straight
```

Ten cars each get a **single attempt**: placed on the official track, driving
until they cross the line they started on having gone all the way round — or
failing, by crashing into a wall, rolling over, or stalling. **The fastest of
the ten laps is the team's submission.** Not the mean, not the median: a team is
judged on the best lap it can produce, the way a qualifying session works.

They start at the **world origin (0, 0)**, facing along the track. `--spawn X Y`
moves that point and `--spawn-yaw DEG` turns it.

**Why ten, and why they are not identical.** The policy is deterministic, so ten
cars from the exact same pose would drive ten identical laps to the millisecond
and tell you nothing one car would not. Each therefore starts within
`--spawn-jitter` degrees of straight, five either way by default. That is far
too small to make one attempt easier than another, but driving a track is
chaotic: a couple of degrees at the line decides whether a car makes a corner
most of a lap later, and a policy that only survives one exact opening has not
really learned the track. On a trained baseline, ±5° of jitter spreads the ten
laps over more than a second. The jitter is seeded (`--seed`), so a rerun
repeats the same ten starts.

The benchmark runs under the SDK's own rules and **ignores your
`compute_terminations`** — otherwise a team with an aggressive stall rule and a
team with none would be running different sessions. Your observations are still
your own; the policy could not run otherwise.

The result is printed and written to `submission.json` next to the checkpoint,
stamped with the SDK fingerprint and the exact spawn headings used, so a time
can be traced to the rules it was set under and reproduced. It is also
published, together with the policy that set it — see
[the leaderboard](#the-leaderboard).

```
════════════════════════════════════════════════════════
  agent  0    -1.0°     15.867 s   lap
  agent  1    +0.2°     15.700 s   lap
  agent  2    -4.8°     16.167 s   lap
  ...
════════════════════════════════════════════════════════
  SUBMISSION      15.200 s      ← fastest of 10
════════════════════════════════════════════════════════
  average speed     3.29 m/s   over 50.0 m
  completed        10/10       100% of attempts
  median lap      15.933 s
  slowest lap     16.567 s
  spawn         world origin (0, 0), facing along the track
  sdk           24129718c513
════════════════════════════════════════════════════════
```

`benchmark` is `play.py` with the watching taken out and a stopwatch put in: the
same environment, the same loaded checkpoint, the same step loop. It lives in
`lituanicax_sdk/benchmark.py` because it is the scoring rules, not a project
script.

---

## The leaderboard

Every `benchmark` run that produces a lap publishes it. Two things go up: **the
lap** — your team name, the time and the SDK's fingerprint — and **the policy
that drove it**, as a zip of the checkpoint, your `teamcode/` and the run's
config.

The board is at **<https://isaacleaderboard.netlify.app/>**. Tell the SDK who
you are, once:

```bash
export LITUANICAX_TEAM="Wingless Wonders"
benchmark --headless
```

```
[submit] policy bundle 3.4 MB, 9 files
[submit] 15.200 s   awaiting verification   would be P3
[submit] Policy bundle stored (3.4 MB). The lap is on the board as awaiting
         verification until the organisers re-run it.
[submit] it counts once the organisers reproduce it on their own machine.
[submit] https://isaacleaderboard.netlify.app/
```

Put that in your shell profile and forget about it, or write it to
`.lituanicax.json` in the project root (gitignored, so a fork does not carry
your name to somebody else):

```json
{"team": "Wingless Wonders"}
```

There is no password, and there is no sign-up: the first lap you publish
creates your team. Nothing is gained by submitting under someone else's name —
a lap slower than theirs does not move them, and a lap faster than theirs is
one you want your own name on.

### A lap reaches the board in two steps

**1. Published.** Your row appears immediately, greyed out, marked *awaiting
verification*. It sits at the time it claims, so you can see where you would be,
and it counts for nothing: it holds no position and displaces nobody.

**2. Verified.** The organisers download the policy you sent, run it themselves
on the official SDK on their own machine, and compare. If it produces your lap
again, the row turns green and takes its position. If it does not, the row is
rejected and leaves the board.

That is the whole of what a green row means, and it is why the board is worth
something. **The fingerprint alone could never do this job.** The SDK is in your
repository — you can read and edit every line of it, including the file that
computes the fingerprint — so a team willing to edit `_locked.py` could send any
twelve characters it liked. A lap re-driven on somebody else's computer cannot
be edited into existence.

A lap has to reproduce **within 0.1 s or 1% of the claim, whichever is larger**
— tens of milliseconds of drift between two machines running the same policy is
expected; a second is a different policy.

The fingerprint still decides whether a lap is *worth checking*: a lap that does
not carry the official one is recorded and never queued. If your submission
comes back ineligible and you have not touched `lituanicax_sdk/`, you are on an
old SDK: pull. Nobody types that fingerprint in either — every push to
`lituanicax_sdk/` on upstream recomputes it and tells the board
([`publish-sdk-hash.yml`](.github/workflows/publish-sdk-hash.yml)), so the board
gates on whatever `git pull` gives you.

### What is in the bundle

Built by [`lituanicax_sdk/bundle.py`](lituanicax_sdk/bundle.py), a few megabytes,
and nothing surprising:

| In the zip | Why it is needed |
| --- | --- |
| `checkpoint/model_*.pt` | the policy that was scored |
| `teamcode/` | your observations and network shape — the checkpoint will not even load without them |
| `params/` | the env and agent config the run was launched with |
| `submission.json` | the report, exactly as it was written to your run |
| `manifest.json` | a hash of every entry, and the command that reproduces the lap |

Nothing else from `logs/` goes anywhere: not the TensorBoard events, not the
videos, not your other checkpoints. Files nothing needs to *run* the policy are
left out and named in the run's output — a `.blend` is how a track was made, not
what the simulator loads.

Yes, this means **your `teamcode/` is on the organisers' disk.** It is not
published, the download needs the organisers' admin token, and it is deleted
with the row if a submission is removed. If you would rather not send it, pass
`--no-submit`: you keep the score and the board never hears about the lap. There
is no third option where a lap counts and nobody may check it.

### The board

One counted row per team — that team's fastest verified lap — plus any faster
lap of theirs still being checked, shown greyed out and without a position.
Sortable by lap time or by when it was set.

Publishing cannot cost you a run. No network, no team name, board down: the
score is still printed and still written to `submission.json`, and one line says
it was not sent. An upload that fails costs you verification, not the run.
Nothing here retries and nothing here raises. `--no-submit` scores a policy
without publishing anything at all, and `--team NAME` overrides the name for one
run.

### Verifying laps — for the organisers

This is the other half, and it needs the board's admin token, which only the
organisers have. It lives here rather than on the website because it needs the
official SDK and a simulator, which is exactly what this repository is.

```bash
export LEADERBOARD_ADMIN_TOKEN="…"

python -m lituanicax_sdk.verify             # the whole queue, one by one
python -m lituanicax_sdk.verify --list      # what is waiting, run nothing
python -m lituanicax_sdk.verify <id>        # just this one
python -m lituanicax_sdk.verify <id> --dry-run --windowed   # watch it drive
```

With no arguments it does the whole job: every lap waiting, downloaded, re-run
and marked, in one sitting. Each one is checked against its own hash, unpacked
into `.verify/<id>/`, and driven — *their* checkpoint and `teamcode/` through
*this* checkout of the SDK, as a subprocess whose working directory is the
workspace, so their code shadows this repository's rather than replacing it. It
posts the lap the machine produced; the board applies the tolerance and decides,
so the rule is one rule, on the server, the same for everyone. One team's broken
bundle is reported and the queue carries on. Laps published without a policy are
skipped and counted at the end — there is nothing to re-run.

**The admin token is the whole gate, and this file being public changes
nothing.** Listing the queue, downloading a bundle and posting a verdict are all
`/api/admin/` endpoints behind `ADMIN_TOKEN`. Without it the board answers 401
to the first request and there is no second one, so a team can read every line
of `verify.py` — and should — and still not move a single row. There is no local
verdict to forge either: this posts *the lap the machine measured*, and the
server decides what that means.

**It runs code a competitor wrote.** It has to: `teamcode/` defines the
observations the checkpoint expects, and importing it runs whatever is in it.
The run's `runtime_fingerprint` is compared against a clean process here, so an
SDK patched in memory is flagged in the note that goes with the verdict — but
that is a tripwire, not a sandbox. Verify on a machine you would be willing to
hand to a stranger, and read a remarkable lap's `teamcode/` before you run it.

---

## Training

```bash
train --num_envs 200 --headless
```

`--headless` skips on-screen rendering, which trains considerably faster, and
`--resume` continues from the most recent checkpoint. Those three flags are all
`train` has.

How long it trains and every other learning setting lives in
`teamcode/ppo_cfg.py`. To change one for a single run without editing the
file, pass it through: `train --num_envs 200 agent.max_iterations=200`.

`teamcode/train.py` starts TensorBoard for you and prints the URL (port
6006, or the next free one). You can also start it yourself:

```bash
tensorboard --logdir logs
```

Metrics worth watching:

| Metric | What it means |
|--------|---------------|
| `Train/mean_reward` | Average episode reward — should climb |
| `Train/mean_episode_length` | How long cars survive (2700 = a full episode) |
| `Train/mean_lr` | Adaptive learning rate (falls as KL rises) |
| `Info/kl` | How much the policy changed — should hover near 0.01 |
| `Lap/best_lap_time_s` | Appears once any car completes a lap |
| `Spawn/mean_ep_len_*` | How the two directions are doing (`PresetSpawnManager` only) |

Anything you pass to `self.log(...)` shows up too, averaged over the iteration.

### PPO hyperparameters

All of these live in `teamcode/ppo_cfg.py`, and all of them are yours.

| Parameter | Value |
|-----------|-------|
| Actor hidden layers | [64, 64] (ELU) |
| Critic hidden layers | [64, 64] (ELU) |
| Initial noise std | 0.8 |
| Learning rate | 1e−4 (adaptive) |
| PPO epochs | 6 |
| Mini-batches | 6 |
| GAE λ | 0.95 |
| Discount γ | 0.99 — a ~3 s horizon; 0.9965 puts a whole lap inside it |
| Entropy coefficient | 0.004 |
| Clip parameter | 0.2 |
| Desired KL | 0.01 |
| Max gradient norm | 0.75 |
| Steps per env per iteration | 800 |

---

## Playing a checkpoint

```bash
play                                # the newest checkpoint
play --num_envs 1                   # one car
play --checkpoint logs/<timestamp>/model_1000.pt
play --video                        # record a clip into the run folder
```

Playback runs at wall-clock speed — it is for watching — and `--enable_cameras`
is set for you, since Isaac Sim will not produce frames without it and video
recording fails with a confusing error.

`play` also exports the policy into the same run folder, at
`logs/<timestamp>/exported/`, as `policy.pt` (TorchScript) and `policy.onnx`,
ready to run on the real car. The observation normaliser is baked in, so the
exported network takes raw observations.

For a lap time, use `benchmark` — that is the official measurement.

---

## Runs and logs

Every run is self-contained in a single timestamped folder, so you can copy,
compare or delete a run by moving one directory:

```
logs/2026-08-01_01-05-23/
├── model_0.pt … model_480.pt    checkpoints, saved every 10 iterations
├── events.out.tfevents.*        TensorBoard data
├── params/env.yaml              the exact task config this run used
├── params/agent.yaml            the exact PPO config this run used
├── git/                         a snapshot of the code at the time
├── exported/                    policy.pt + policy.onnx, written by play
├── submission.json              written by benchmark
└── videos/                      only if you passed --video
```

`train`, `play` and `benchmark` all agree on this layout, and all three default
to the most recently modified checkpoint under `logs/`.

---

## Tracks

A track is a centerline CSV (a closed loop of `x,y` points a few centimetres
apart), a walls USD that becomes the crash boundary, and an optional surface USD
that is pure scenery — cones and tarmac never get collision, so decorating a
track cannot change how it drives.

```python
from lituanicax_sdk.tracks import register, get
from lituanicax_sdk import TrackCfg

register(TrackCfg(
    name="figure_eight",
    walls_usd="teamcode/tracks/eight_walls.usdc",
    surface_usd="teamcode/tracks/eight_surface.usdc",
    centerline_csv="teamcode/tracks/eight_line.csv",
))

track = get("figure_eight")
```

Then set `cfg.track = get("figure_eight")`. Train on whatever you like; lap
times are only compared on official tracks.

**Check a new track before spending a night training on it:**

```python
from lituanicax_sdk import Track
for problem in Track(get("figure_eight")).validate():
    print(problem)
```

`validate()` catches a centerline that is not quite a closed loop, one whose
points are unevenly spaced, and — once the walls are loaded — one that is not
running between them. All three produce curvature and lookahead observations
that are subtly wrong rather than obviously broken, which is the worst kind of
bug to find eight hours into a training run.

`centerline_scale` has to agree with `mesh_scale`: the walls and the centerline
describe the same track, so whatever scales one scales the other. The official
track shipped for a while with a centerline scaled 15% larger than its own
walls — it ran outside them for a quarter of the lap, and every track-relative
observation was measured against a racing line that was not on the track. That
is exactly what the wall check now catches.

---

## Extra objects and sensors

```python
extra_scene_entities = {
    "lidar": RayCasterCfg(...),      # reachable as car.sensors["lidar"]
    "obstacle": RigidObjectCfg(...),
}
```

Sensor, rigid-object and articulation configs are recognised; anything else with
a `.func()` is spawned as plain USD.

---

## Integrity

The SDK hashes its own sources. If a file has changed, every run prints a
warning and the fingerprint is recorded next to the lap times, so a reported time
can always be traced to the rules it was set under.

This is a soft lock: the SDK is in your repository and you can read and edit
every line of it. It is not there to stop you — it is there so that when you
change something, you and everyone else know.

**And a soft lock is all a fingerprint can ever be.** It is computed by your copy
of `_locked.py`. A team willing to edit that file can report the official
fingerprint from an SDK that is nothing of the sort, and no amount of hashing
inside the repository fixes that: the thing doing the checking is the thing
being checked.

So the board does not rely on it. What puts a lap on the leaderboard is the
organisers **re-running your policy on their own machine, against their own
copy of the SDK** — see [the leaderboard](#the-leaderboard). Your fingerprint
decides whether a lap is worth their time; their re-run decides whether it
counts. Editing the SDK to make the car faster therefore does not cost you a
warning, and does not cost you a fingerprint check you might route around: it
costs you the lap, because the car that drives it at the other end is the stock
one and the time will not come back.

Two things are hashed, and they catch different cheats:

| | What it hashes | What it catches |
| --- | --- | --- |
| `sdk_fingerprint()` | the SDK's **files** on disk | an edited SDK |
| `runtime_fingerprint()` | the SDK's competition-critical **objects in memory**, after the driving is done | an SDK patched at runtime — `teamcode/__init__.py` replacing the crash rules or moving `WALL_COLLISION_RADIUS_M`, which changes no file at all |

The second travels in the bundle's manifest, and the organisers compare it with
what a clean process on their own machine produces. Neither is a proof; both are
things a lap has to explain.

---

## Tests

The parts of the SDK that do not need Isaac Sim — the lap timer, the attempt
timer, the track maths, `CarState`, every term, the locks, the policy bundle,
and both leaderboard clients against a real socket — are tested on the CPU in
seconds:

```bash
.venv/bin/python -m pytest tests/ -q
```

Worth running after you change anything in `teamcode/`: a term that returns
the wrong shape, or an observation that stops being finite at speed, is caught
here rather than an hour into training.

---

## Troubleshooting

**Isaac Sim crashes on start-up, somewhere in `librtx`.** Your NVIDIA driver is
probably 590 or newer. See [the driver
section](#the-driver-version-really-matters).

**A run redirected to a file produces an empty log.** Isaac Sim exits the
process without flushing Python's buffers, so anything still in the buffer is
lost — which is the whole report. Run with `python -u`, or check the artefacts
(`submission.json`, the run folder) rather than the log.

**`benchmark` reports `NO SUBMISSION` for every agent.** The car is not
completing a lap. Watch it with `play` first; if it drives fine there but not
under `benchmark`, remember that the benchmark starts at (0, 0) — a pose your
training may never visit unless you keep a spawn point there.

**`[submit] not published`.** The rest of the line says why: usually no team
name (set `LITUANICAX_TEAM`) or a board it could not reach. The lap is not lost
— it is in `submission.json` — so fix the cause and rerun, or ask an organiser
to add it by hand.

**The lap was sent but not ranked.** The fingerprint printed by `benchmark`
does not match the one the board is gating on. Either you have edited
`lituanicax_sdk/` — `git status` will say so — or you are behind: `git pull`.

**Cars behave strangely from a standing start.** Every track-relative
observation depends on which way round the track the car is going. At rest that
comes from the car's heading rather than its velocity; if you have replaced
`going_forward`, check that case.

**Out of GPU memory while training.** Lower `--num_envs`. 200 cars fit in 8 GB
alongside the track; the benchmark's ten always will.
