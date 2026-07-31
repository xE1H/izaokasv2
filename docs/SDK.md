# The LituanicaX SDK

Every team drives the same car on the same clock. What you build is the policy
that drives it, and everything you use to train that policy.

The SDK gives you a vehicle, a ground plane, a track and physics — and tells you
everything measurable about the car at every step. It gives you no rewards and
no observations, because those are the exercise.

```
lituanicax_sdk/     the car, the ground, the physics, the lap clock
                    + every car parameter, readable
  env.py              RaceEnv and its config — the simulation loop
  scene.py            one-time USD scene building
  vehicle.py          the car        dynamics.py   the motor model
  state.py            CarState       timing.py     the lap clock
  rules.py            crash rules    track.py      tracks
  evaluate.py         the official score
  runs.py             where runs live, finding checkpoints

team_solution/      the logic: observations, reward, terminations, training
  env.py              everything you write, plus where the cars drive
  ppo_cfg.py          the learning settings
```

The rule of thumb for the split: **if changing it makes the car faster rather
than the policy better, it is locked.**

---

## What is locked

| | Why |
|---|---|
| The MuSHR nano v2: mass, wheels, USD, scale | Everyone drives the same car |
| Motor torque curve, brakes, steering limits | Same acceleration and same grip |
| Ground and tyre friction | Same grip |
| Policy rate (30 Hz) and decimation (4, so 120 Hz physics) | Lap times are counted in policy steps |
| The action space: throttle and steering, both `[-1, 1]` | One control interface |
| Crash rules *during a measured lap* | A recorded lap is always a clean lap |
| Lap timing | The number the competition is about |
| The official track and its start/finish line | A lap means the same thing for everyone |

Try to change any of these and you get an error naming the parameter, not a
silently different simulation:

```python
>>> from lituanicax_sdk import VEHICLE
>>> VEHICLE.drive_torque_nm = 0.5
LockedParameterError: VehicleCfg.drive_torque_nm cannot be changed.
This parameter is fixed by the competition rules so that every team's car
behaves identically — see docs/SDK.md for what you can change instead.
```

The same applies to Hydra overrides on the command line — `train env.sim.dt=0.001`
is rejected when the environment is built.

The clock is two numbers, `TimingCfg.policy_hz` and `TimingCfg.decimation` in
`lituanicax_sdk/vehicle.py`. Everything else about it — `physics_hz`,
`physics_dt`, `step_dt`, `render_interval` — is derived from those, so whoever
runs the competition changes the rate in one place and nothing can fall out of
step with it. Teams read the values and never set them.

You can **read** every locked value, and you should. That is the other half of
the deal: the car is fixed, so the SDK holds nothing back about it.
`car.max_speed_m_s`, `car.wheel_radius_m`, `car.max_steer_rad`, `car.mass_kg`,
`car.drive_torque_nm`, `car.step_dt` — and `car.vehicle` for the whole spec, down
to the brake fade constants. You should never have to hardcode `6.7` or `0.037`.

## What is yours

Observations · reward · terminations · spawn points and curriculum · episode
length · tracks · extra objects and sensors · number of cars · the RL
algorithm, network and hyperparameters · the trainer itself.

The SDK ships **no rewards and no observations**. Not as an oversight — a
reward the SDK wrote would be a reward every team shared, and picking weights
off a menu is not the same exercise as deciding what matters.

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

`team_solution/env.py` is a complete worked example — 23 observations, 7 reward
terms, three terminations — that trains as-is and is yours to rewrite.

---

## What `car` gives you

`CarState`, one row per car, in raw SI units. How you scale it is your business.

**Motion:** `speed_forward` `speed_lateral` `speed_ground` `yaw_rate`
`lin_vel_b` `lin_vel_w` `ang_vel_b`

**Pose:** `pos_w` `pos_xy` `quat_w` `yaw` `roll` `pitch` `up_axis` `forward_xy`

**Drivetrain:** `wheel_omega` `wheel_speed` `slip` `steer_angle`
`wheel_speeds` `wheel_slips` `applied_wheel_torque` `suspension_travel`
(the per-wheel arrays are `[N, 4]`, ordered as `wheelbase_names`)

**Raw joints:** `joint_pos` `joint_vel` `joint_names` — everything, if the
shaped readouts are not enough

**Commands:** `throttle_cmd` `steer_cmd` and their `_prev`

**On the track:** `nearest_idx` `cross_track_error` `signed_cross_track_error`
`heading_error` `going_forward` `direction_sign` `progress_m`
`dist_to_next_corner` `next_corner_curvature` `dist_to_wall`
`lookahead(offsets)` `track`

**Episode and lap:** `episode_step` `episode_time_s` `wall_touched`
`lap.count` `lap.last_time_s` `lap.best_time_s` `lap.just_finished`

**The locked car, readable in full:** `max_speed_m_s` `wheel_radius_m`
`max_steer_rad` `mass_kg` `drive_torque_nm` `max_wheel_omega` `step_dt`, and
`car.vehicle` for the rest of the spec. You cannot change these, but you should
never have to hardcode `6.7` or `0.037` either.

**Your sensors:** `car.sensors["<name>"]` for anything in
`cfg.extra_scene_entities`.

`CarState` is a view: it hands out numbers, never the articulation they came
from, so your code can read the simulation but cannot reach in and change it.
Everything is computed once per step and cached, so reading `nearest_idx` in six
places costs what reading it once costs.

Every track-relative quantity mirrors itself for cars driving the other way
round, which is what lets one policy learn the track in both directions.

---

## Terminations, and the two modes

While you **train**, `compute_terminations` is the only thing that ends an
episode early. Nothing is forced on you. Ending an episode when a car has
stopped saves samples; ending one when it drifts wide teaches a tighter line and
may also teach timidity. Your call.

While a lap is being **measured** (`evaluate`, or `enforce_official_rules`), the
SDK's crash rules apply *instead of* yours and a car that hits a wall freezes,
so every team's cars fail for exactly the same reasons:

| | |
|---|---|
| Wall contact | centre within 0.15 m of a wall → freeze, episode ends |
| Roll-over | up-axis below 0.3, about 73° |
| Stall (evaluate only) | barely moving for 45 consecutive steps → episode ends |

The stall rule is separate because whether a stalled car is worth terminating
is a training decision — the baseline solution does terminate on it. During a
measured run it is on, so a car pinned against a wall fails in a moment instead
of sitting out the whole attempt window, and a car that just completed a lap is
terminated on the spot (its time is already recorded).

One rule holds in **both** modes: **a lap during which the car touched a wall is
not a valid lap.** That is a property of the lap rather than a termination, so
it applies however you end your episodes — which is why you are free to let a
crashed car keep driving and learn to recover.

**Lap time is never a reward.** It is measured, and a measurement the policy can
influence is not a measurement. To reward progress use `car.speed_forward` or
`car.progress_m`.

## Tracks

A track is a centerline CSV (a closed loop of `x,y` points a few centimetres
apart), a walls USD that becomes the crash boundary, and an optional surface
USD that is pure scenery — cones and tarmac never get collision, so decorating
a track cannot change how it drives.

```python
from lituanicax_sdk.tracks import register, get
from lituanicax_sdk import TrackCfg

register(TrackCfg(
    name="figure_eight",
    walls_usd="team_solution/tracks/eight_walls.usdc",
    surface_usd="team_solution/tracks/eight_surface.usdc",
    centerline_csv="team_solution/tracks/eight_line.csv",
    spawn_points=[(0.0, 0.0, 90.0)],
))

track = get("figure_eight")
```

Check a new track before spending a night training on it:

```python
from lituanicax_sdk import Track
for problem in Track(get("figure_eight")).validate():
    print(problem)
```

A centerline that is not quite a closed loop, or whose points are unevenly
spaced, produces curvature and lookahead observations that are subtly wrong
rather than obviously broken. `validate()` catches that.

Train on whatever you like. Lap times are only compared on official tracks.

## Extra objects and sensors

```python
extra_scene_entities = {
    "lidar": RayCasterCfg(...),      # reachable as state.sensors["lidar"]
    "obstacle": RigidObjectCfg(...),
}
```

Sensor, rigid-object and articulation configs are recognised; anything else
with a `.func()` is spawned as plain USD.

## One constraint worth knowing

`scene.env_spacing` must be `0`. Every car drives the *same* shared track
rather than its own copy, so they all occupy the same space and the track is
loaded once. Cross-track error, lookahead and lap timing are all measured in
world coordinates against that one track. Setting a non-zero spacing is
rejected at start-up rather than silently measuring most of your cars against a
track that is no longer under them.

## What is logged

| | |
|---|---|
| anything you pass to `self.log(...)` | averaged over the iteration |
| `Lap/*` | lap times, once any exist |
| `Spawn/mean_ep_len_forward` / `_reversed` | how the two directions are doing, which is what drives the spawn balancing |

## Spawning

`SpawnManager` places cars at the track's preset points, each usable facing
either way, and biases spawns towards whichever direction is currently doing
worse. Subclass it and set `cfg.spawn_manager` to do something else.

What you cannot move is where the lap clock starts and stops. That is the
track's start/finish line, not your spawn point — which is exactly what makes
times comparable between teams that spawn differently.

---

## Lap timing

A lap runs from the start/finish line back to itself. That line is a plane
through one centerline point, at right angles to the track — and it spans the
full width of the track, so a wide racing line crosses it just as a car on the
centerline does.

Three rules keep the numbers honest:

- **Arming** — the car must leave the gate window (2 m of track either side of
  the line) before a crossing counts, so sitting on the line does not tick over
  laps.
- **Travel** — it must reach the far side of the loop, so nudging over the line
  and reversing back is not a lap.
- **Out-lap** — the first crossing after a spawn only starts the clock. Cars
  start wherever they start, so that first partial loop is never timed.

A lap during which the car touched a wall does not count. Both directions round
the track are timed.

In TensorBoard: `Lap/best_lap_time_s`, `Lap/mean_lap_time_s` and
`Lap/laps_per_min`. The first two only appear once some car has actually
completed a lap — before that there is nothing to average, and a placeholder
would just be a misleading flat line.

### The official score

```bash
evaluate                              # 100 agents, one attempt each
evaluate --agents 100 --batch-size 25 # lower the batch if 25 cars will not fit
evaluate --spawn-presets              # the track's preset points, not (0, 0)
```

One hundred agents each get a **single attempt**: drive until you complete a lap
or fail. **The fastest of the hundred is the team's submission** — a team is
judged on the best lap it can produce, the way a qualifying session works.

Every agent starts at the **world origin (0, 0)**, facing along the track, so
all hundred attempts begin from the same point. The origin sits on the
centerline a little before the start/finish line, so an attempt is a short
out-lap to the line, then one timed lap back to it — and the episode ends the
step the lap is recorded. Pass `--spawn-presets` to start from the track's
preset points instead. The hundred run in rounds of `--batch-size` because they
will not all fit on an 8 GB card at once; the rounds are independent attempts
in one session.

The evaluation uses the SDK's own rules and **ignores `compute_terminations`**
— otherwise a team with an aggressive stall rule and a team with none would be
running different sessions. Its rules are the crash rules (wall contact,
roll-over) plus a stall rule that cuts off a car staying barely moving for too
long, which is how a car pinned against a wall fails instead of sitting out the
whole attempt window. Your observations are still your own; the policy could
not run otherwise.

Results go to `submission.json` next to the checkpoint, stamped with the SDK
fingerprint so a time can be traced to the rules it was set under.

## Integrity

The SDK hashes its own sources. If a file has changed, every run prints a
warning and the fingerprint is recorded next to the lap times, so a reported
time can always be traced to the rules it was set under.

This is a soft lock: the SDK is in your repository and you can read and edit
every line of it. It is not there to stop you — it is there so that when you
change something, you and everyone else know.

---

## Running the tests

The parts of the SDK that do not need Isaac Sim — the lap timer, the track
maths, `CarState`, every term, and the locks — are tested on the CPU in
seconds:

```bash
.venv/bin/python -m pytest tests/ -q
```

Worth running after you change anything in `team_solution/`: a term that
returns the wrong shape or an observation that stops being finite at speed is
caught here rather than an hour into training.
