"""``RaceEnv`` and its config — the car, the ground and the physics.

The SDK simulates the vehicle and the track and hands you its complete state.
What the policy *sees*, what it is *paid for*, and when an episode *ends* are
yours to write — subclass :class:`RaceEnv` and implement three methods:

    from lituanicax_sdk import RaceEnv

    class TeamEnv(RaceEnv):
        def compute_observations(self, car):
            return torch.stack([...], dim=-1)

        def compute_reward(self, car):
            return car.speed_forward / car.max_speed_m_s * car.step_dt

        def compute_terminations(self, car):
            return car.speed_forward < 0.2

``car`` is a :class:`~lituanicax_sdk.state.CarState`: every measurable property
of the vehicle and its position on the track, in SI units. There is no menu of
ready-made rewards to pick from, because a reward the SDK wrote would be a
reward every team shared.

Everything below those three methods is sealed. Overriding it would mean a
different car or a different clock, which is the one thing the competition
fixes. Hundreds of copies of the car run in parallel on the GPU, so nearly
every variable here is a tensor with one row per car.

Scene construction lives in :mod:`lituanicax_sdk.scene`; it is a one-time USD
job and reading it is rarely what you want when looking at how a step works.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from . import rules, scene, tracks
from ._locked import (
    LockedParameterError,
    SealedMeta,
    assert_unchanged,
    sdk_fingerprint,
    sealed,
    verify_integrity,
)
from .dynamics import clamp_actions, steer_target, wheel_torque
from .spawn import SpawnManager
from .state import CarState, ControlHistory, LapView
from .timing import LapTimer
from .track import Track, TrackCfg
from .tracks import OFFICIAL
from .vehicle import TIMING, VEHICLE, build_articulation_cfg, build_sim_cfg

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════


@configclass
class RaceEnvCfg(DirectRLEnvCfg):
    """Where the cars drive and for how long.

    Deliberately short. The SDK's job is to simulate the car, the ground and
    the track; what the policy sees and what it is paid for is code you write
    in :meth:`RaceEnv.compute_observations` and friends, not options here.

    ``sim``, ``decimation`` and ``robot_cfg`` appear because Isaac Lab requires
    them on the config object, but they are the locked vehicle and clock:
    setting them raises rather than silently taking effect.
    """

    # ── Locked, and re-checked when the environment is built ───────────────
    sim = build_sim_cfg()
    decimation: int = TIMING.decimation
    robot_cfg = build_articulation_cfg()
    action_space: int = 2  # throttle, steering — fixed
    state_space: int = 0  # the critic reads the same observations as the actor
    ui_window_class_type = None

    # ── Yours ─────────────────────────────────────────────────────────────

    #: How many numbers :meth:`RaceEnv.compute_observations` returns per car.
    #: This sizes the policy's input before the simulation starts, so it has to
    #: be declared rather than discovered; the first step checks that the two
    #: agree.
    observation_space: int = 0

    #: How long an episode may last. Longer episodes mean more laps per
    #: episode, and a value estimate that has to look further ahead.
    episode_length_s: float = 90.0

    #: Every car shares one track, so there is no grid offset between them.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200, env_spacing=0.0, replicate_physics=True
    )

    #: Which track to drive. Official tracks are locked; register your own with
    #: :func:`lituanicax_sdk.tracks.register`.
    track: TrackCfg = OFFICIAL

    #: Where cars start, and how the two directions are balanced. Subclass
    #: :class:`~lituanicax_sdk.spawn.SpawnManager` to do something else.
    spawn_manager: SpawnManager = SpawnManager()

    #: Run under the official rules: :mod:`lituanicax_sdk.rules` decides when an
    #: episode ends *instead of* your ``compute_terminations``, and a car that
    #: hits a wall freezes on the spot.
    #:
    #: It also makes an attempt a car's only life. A car whose episode has
    #: ended is not respawned: it stops being driven, and it is hidden, so the
    #: cars on the track are exactly the cars still being scored. Training does
    #: the opposite, and should: there a respawn is the next sample.
    #:
    #: Off while you train — what ends an episode is your decision there. The
    #: benchmark turns it on, so that every team's cars fail for the same
    #: reasons and a lap time compares like with like.
    enforce_official_rules: bool = False

    #: Terminate a car the step it completes a valid lap, instead of letting it
    #: keep driving (and respawning) once it has banked a time.
    #:
    #: Off while you train, where running laps is itself the learning signal.
    #: The benchmark turns it on, so an attempt is exactly one out-lap and one
    #: timed lap, never more.
    terminate_on_lap: bool = False

    #: On top of the crash rules, cut off a car that stays barely moving for
    #: too long (:func:`lituanicax_sdk.rules.stalled`).
    #:
    #: Off while you train — whether a stalled car is worth terminating is a
    #: training decision (the baseline solution terminates on it). The
    #: benchmark turns it on: a car pinned against a wall can sit there for the
    #: whole attempt window while its centre stays just outside the crash
    #: radius, and a failed attempt should not take 90 seconds to fail.
    official_stall_rule: bool = False

    #: Extra things to put in the scene: obstacles, sensors, lights. Values are
    #: Isaac Lab entity configs, keyed by the name you want to reach them under
    #: in ``car.sensors``.
    extra_scene_entities: dict = {}

    #: Brightness of the default dome light.
    light_intensity: float = 2000.0


# ═══════════════════════════════════════════════════════════════════════════
#  The environment
# ═══════════════════════════════════════════════════════════════════════════


class RaceEnv(DirectRLEnv, metaclass=SealedMeta):
    """Simulates ``num_envs`` copies of the locked car on one shared track."""

    cfg: RaceEnvCfg

    #: The loaded track. Built in :meth:`_setup_scene`, which ``super().__init__``
    #: calls, so it exists from the end of ``__init__`` onwards — declared rather
    #: than assigned here so it never reads as optional.
    track: Track

    def __init__(self, cfg: RaceEnvCfg, render_mode: str | None = None, **kwargs):
        verify_integrity()
        self._enforce_locked_cfg(cfg)

        super().__init__(cfg, render_mode, **kwargs)

        # ── Joint indices, looked up once by name ──────────────────────────
        throttle_ids, _ = self.robot.find_joints(list(VEHICLE.throttle_joints))
        steer_ids, _ = self.robot.find_joints(list(VEHICLE.steer_joints))
        suspension_ids, _ = self.robot.find_joints(list(VEHICLE.suspension_joints))
        self._throttle_ids = torch.tensor(
            throttle_ids, dtype=torch.long, device=self.device
        )
        self._steer_ids = torch.tensor(steer_ids, dtype=torch.long, device=self.device)
        self._suspension_ids = torch.tensor(
            suspension_ids, dtype=torch.long, device=self.device
        )

        self._rescale_to_real_mass()

        # ── Per-car state ──────────────────────────────────────────────────
        self._wheel_torque = torch.zeros(self.num_envs, 4, device=self.device)
        self._steer_tan = torch.zeros(self.num_envs, device=self.device)
        self._commands = ControlHistory(self.num_envs, self.device)

        # Sticky for the episode. Always tracked, because a lap with a wall
        # touch in it is invalid however you have chosen to end your episodes.
        self._wall_touched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Consecutive slow steps per car, for the benchmark's stall rule.
        self._stall_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Under official rules a car gets one attempt: once its episode has
        # ended it is left where it stopped rather than respawned. Cleared by
        # :meth:`reset`, which is what starts a session.
        self._attempt_over = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.track.load_walls()
        for problem in self.track.validate():
            print(f"[track:{self.track.cfg.name}] warning: {problem}")
        print(self.track.describe())

        self.lap_timer = LapTimer(
            track=self.track,
            num_envs=self.num_envs,
            step_dt=TIMING.step_dt,
            device=self.device,
            grace_steps=rules.SPAWN_GRACE_STEPS,
        )
        self._lap_view = LapView(self.lap_timer)
        self.cfg.spawn_manager.setup(self.track, self.num_envs, self.device)

        self._car: CarState | None = None
        self._log: dict[str, torch.Tensor] = {}

        mode = "OFFICIAL RULES" if cfg.enforce_official_rules else "training"
        print(
            f"[lituanicax] SDK {sdk_fingerprint()} | {mode} | "
            f"{cfg.observation_space} observations, "
            f"{TIMING.describe()}, track '{cfg.track.name}'"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  What you write
    # ══════════════════════════════════════════════════════════════════════

    def compute_observations(self, car: CarState) -> torch.Tensor:
        """What the policy sees. Return ``[num_envs, cfg.observation_space]``.

        Build it from anything on ``car``. Networks train better on inputs of a
        similar, bounded scale, so divide by something meaningful and clamp —
        ``car.max_speed_m_s``, ``car.max_steer_rad`` and the rest of the locked
        constants are there so you never have to hardcode a number.

        The width must match ``cfg.observation_space``, which sizes the policy
        before the simulation starts. A mismatch is reported on the first step.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_observations(car). "
            "The SDK simulates the car; what the policy sees is yours to decide. "
            "See README.md and teamcode/env.py."
        )

    def compute_reward(self, car: CarState) -> torch.Tensor:
        """What the policy is paid for. Return ``[num_envs]``.

        Entirely yours — the SDK ships no rewards, because a reward it wrote
        would be a reward every team shared.

        Use :meth:`log` to record the parts separately; when a policy does
        something strange, the fastest way to find out why is to see which term
        paid for it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_reward(car). "
            "See README.md and teamcode/env.py."
        )

    def compute_terminations(self, car: CarState) -> torch.Tensor:
        """When an episode ends early. Return ``[num_envs]`` boolean.

        Optional — the default never terminates, and episodes simply run to
        ``cfg.episode_length_s``.

        This is a training decision, not a rule, which is why the SDK leaves it
        to you. Ending an episode when a car has stopped stops it wasting
        samples; ending one when it drifts wide teaches a tighter line and may
        also teach timidity.

        Ignored entirely during a measured run — see ``enforce_official_rules``.
        """
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def log(self, name: str, value: torch.Tensor | float) -> None:
        """Record a scalar for TensorBoard, averaged over the iteration.

        Call it from :meth:`compute_reward` to trace each part of your reward::

            self.log("Rewards/forward", forward)

        Anything with a ``/`` in the name keeps it; anything without is filed
        under ``Episode/``.
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value), device=self.device)
        self._log[name] = value.detach().float().mean().unsqueeze(0)

    # ══════════════════════════════════════════════════════════════════════
    #  Locked below here
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _enforce_locked_cfg(cfg: RaceEnvCfg) -> None:
        """Re-check the locked values that must live on the Isaac Lab config.

        ``sim.dt``, ``decimation`` and the action space sit in a config tree
        that Hydra can write to from the command line (``env.sim.dt=...``), so
        checking them here covers command-line overrides, loaded YAML and plain
        assignment in one place.
        """
        assert_unchanged(cfg.sim.dt, TIMING.physics_dt, "sim.dt")
        assert_unchanged(cfg.decimation, TIMING.decimation, "decimation")
        assert_unchanged(cfg.action_space, VEHICLE.action_dim, "action_space")

        # Swapping the spawner for a different kind is itself a way of changing
        # the car, so the type is checked before the values are.
        spawn = cfg.robot_cfg.spawn
        if not isinstance(spawn, UsdFileCfg):
            raise LockedParameterError(
                f"robot_cfg.spawn is a {type(spawn).__name__}, but the "
                "competition car is spawned from a USD file."
            )
        assert_unchanged(spawn.usd_path, VEHICLE.usd_path, "robot_cfg.spawn.usd_path")
        assert_unchanged(
            tuple(spawn.scale or ()),
            (VEHICLE.usd_scale,) * 3,
            "robot_cfg.spawn.scale",
        )

        # Not a lock, but a hard constraint of the design worth catching early:
        # every car shares one track at the world origin, so cross-track error,
        # lookahead and lap timing are all measured in world coordinates. Spread
        # the environments out on a grid and every car but the first would be
        # measured against a track that is no longer under it.
        if cfg.scene.env_spacing != 0.0:
            raise ValueError(
                f"scene.env_spacing is {cfg.scene.env_spacing}, but it must be 0. "
                "Every car drives the same shared track, so they all occupy the "
                "same space rather than a grid of copies."
            )

        if cfg.observation_space < 1:
            raise ValueError(
                "cfg.observation_space must say how many numbers "
                "compute_observations() returns; it sizes the policy's input "
                "before the simulation starts."
            )

        # Official tracks are locked: their geometry, spawn points and
        # start/finish line are what make a lap time mean the same thing for
        # everyone. Your own tracks are free to be anything.
        if tracks.is_official(cfg.track.name):
            official = tracks.get(cfg.track.name)
            for field_name, expected in vars(official).items():
                assert_unchanged(
                    getattr(cfg.track, field_name),
                    expected,
                    f"track '{official.name}'.{field_name}",
                )

    def _rescale_to_real_mass(self) -> None:
        """Scale the imported car to the real vehicle's mass."""
        view = self.robot.root_physx_view
        masses = view.get_masses()
        factor = VEHICLE.mass_kg / masses[0].sum().item()
        cpu_ids = torch.arange(self.num_envs, dtype=torch.int, device="cpu")
        view.set_masses(masses * factor, cpu_ids)
        view.set_inertias(view.get_inertias() * factor, cpu_ids)

    # ──────────────────────────────────────────────────────────────────────
    #  1. Scene construction — runs once, from __init__
    # ──────────────────────────────────────────────────────────────────────

    @sealed
    def _setup_scene(self):
        """Spawn the ground, the track, the walls and one car per environment."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.track = Track(self.cfg.track, device=self.device)

        scene.spawn_track(self.track)
        scene.spawn_ground()
        scene.prepare_car_physics(self.num_envs)

        rubber = scene.define_tyre_material()
        self.scene.clone_environments(copy_from_source=False)
        scene.bind_tyre_material(self.num_envs, rubber)

        self.scene.articulations["robot"] = self.robot
        for name, entity_cfg in self.cfg.extra_scene_entities.items():
            scene.spawn_extra_entity(self.scene, name, entity_cfg)

        scene.spawn_light(self.cfg.light_intensity)

    # ──────────────────────────────────────────────────────────────────────
    #  2. Reset — put the cars back on the track
    # ──────────────────────────────────────────────────────────────────────

    @sealed
    def retire(self, cars: torch.Tensor) -> None:
        """Take cars out of the session: no more driving, no respawn, hidden.

        The environment knows an attempt is over when the car *fails* — it
        ends the episode itself. It cannot know when one *succeeds*: the
        benchmark times attempts against its own gate at the spawn point, not
        the track's start/finish line, so the caller is the only one holding
        that fact. This is how it hands it back.

        Without it the cars that get round first — the good ones — carry on
        lapping while the rest are still on their attempt.

        Args:
            cars: ``[N]`` boolean, true for each car that is finished.

        Ignored while training: a car that has just done something worth
        learning from is one you want more samples from, not fewer.
        """
        if self.cfg.enforce_official_rules:
            self._retire(cars)

    def _retire(self, cars: torch.Tensor) -> None:
        """Mark cars as finished and take them off the screen.

        Idempotent, and it has to be: a retired car goes on reporting done
        every step, so this is called with the same cars over and over. Only
        the newly retired ones are hidden, which keeps the USD work to the
        handful of steps where something actually changes.
        """
        newly_done = cars & ~self._attempt_over
        if not bool(newly_done.any()):
            return
        self._attempt_over |= newly_done
        for env_id in newly_done.nonzero().flatten().tolist():
            scene.set_car_visible(env_id, False)

    @sealed
    def reset(self, seed: int | None = None, options: dict | None = None):
        """Put every car back on the grid and start a fresh set of attempts.

        Under official rules :meth:`_reset_idx` refuses to respawn a car whose
        attempt is over, so something has to say when a *new* session begins.
        This is that: the one call that legitimately starts everything again.
        Inferring it from "every car is being reset at once" would not work —
        under official rules the cars that have already settled keep reporting
        done, so the step the last one times out looks exactly the same.
        """
        for env_id in self._attempt_over.nonzero().flatten().tolist():
            scene.set_car_visible(env_id, True)
        self._attempt_over[:] = False
        return super().reset(seed=seed, options=options)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        # Isaac Lab types env_ids as a Sequence[int] but indexes tensors with it
        # throughout, so a tensor is what it actually wants — and what every
        # caller below needs.
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # One attempt per car under official rules. Isaac Lab respawns whatever
        # terminated, which would put a car that has already crashed back on the
        # grid to drive the track again while the others are still on their
        # attempt — scored correctly (the benchmark banks the first result and
        # ignores the car afterwards) but wrong to watch, and wrong in the
        # replay. Dropping those cars here leaves them where they stopped.
        if self.cfg.enforce_official_rules:
            ids = ids[~self._attempt_over[ids]]
            if len(ids) == 0:
                return

        # Must happen before super() clears the episode counters.
        self.cfg.spawn_manager.observe_episode_lengths(
            ids, self.episode_length_buf[ids].float()
        )

        # Isaac Lab's own write_*_to_sim take the same tensor under a
        # Sequence[int] annotation; one alias keeps the calls below readable.
        sim_ids = cast(Sequence[int], ids)
        super()._reset_idx(sim_ids)

        xy, yaw = self.cfg.spawn_manager.sample(ids)

        root_state = self.robot.data.default_root_state[ids].clone()
        root_state[:, :3] = self.scene.env_origins[ids]
        root_state[:, 0] += xy[:, 0]
        root_state[:, 1] += xy[:, 1]
        root_state[:, 2] += self.cfg.spawn_manager.height_m
        zeros = torch.zeros(len(ids), device=self.device)
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root_state[:, 7:] = 0.0

        self.robot.write_root_pose_to_sim(root_state[:, :7], sim_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], sim_ids)

        joint_pos = self.robot.data.default_joint_pos[ids]
        self.robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), None, sim_ids
        )

        self._commands.reset(ids)
        self._wall_touched[ids] = False
        self._stall_steps[ids] = 0
        self.lap_timer.reset(ids)

    # ──────────────────────────────────────────────────────────────────────
    #  3. Actions — two numbers become wheel torque and steering
    # ──────────────────────────────────────────────────────────────────────

    @sealed
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert throttle and steering into physical targets, at the policy rate."""
        commands = clamp_actions(actions)
        throttle, steer = commands[:, 0], commands[:, 1]

        omega = self.robot.data.joint_vel[:, self._throttle_ids]
        self._wheel_torque = wheel_torque(throttle, omega)
        self._steer_tan = steer_target(steer)
        self._commands.push(throttle, steer)

    @sealed
    def _apply_action(self) -> None:
        """Push the targets into the simulator, every physics step."""
        # Wall contact is tracked here as well as in _get_dones so that it is
        # caught the instant it happens, rather than up to one policy step late.
        # It is always recorded, because a lap with a wall touch in it is not a
        # valid lap however you have chosen to end your episodes.
        past_grace = self.episode_length_buf >= rules.SPAWN_GRACE_STEPS
        touching = (
            self.track.distance_to_walls(self.robot.data.root_pos_w[:, :2])
            < rules.WALL_COLLISION_RADIUS_M
        )
        self._wall_touched |= touching & past_grace

        torque = self._wheel_torque
        steer = self._steer_tan.unsqueeze(-1).expand(-1, 2)

        if self.cfg.enforce_official_rules:
            # A crashed car freezes on the spot, so it cannot scrape its way to
            # a better time. Only during a measured run: while training, what a
            # car does after touching a wall is your business.
            #
            # A car whose attempt is over freezes too. It is not respawned, so
            # without this the policy would go on driving it round the track
            # from wherever it stopped, for the rest of the session.
            done_driving = self._wall_touched | self._attempt_over
            torque = torque.clone()
            steer = steer.clone()
            torque[done_driving] = 0.0
            steer[done_driving] = 0.0

        # joint_ids is typed Sequence[int], but it is used as a tensor index —
        # which is what these are, looked up by name once in __init__.
        throttle_ids = cast(Sequence[int], self._throttle_ids)
        steer_ids = cast(Sequence[int], self._steer_ids)
        self.robot.set_joint_effort_target(torque, joint_ids=throttle_ids)
        self.robot.set_joint_position_target(steer, joint_ids=steer_ids)

    # ──────────────────────────────────────────────────────────────────────
    #  4. The state you build on
    # ──────────────────────────────────────────────────────────────────────

    def _build_car_state(self) -> CarState:
        """Snapshot the simulation for this policy step.

        One :class:`CarState` is shared by observations, reward and
        terminations, and caches what it computes — so a quantity read in six
        places costs what reading it once costs.
        """
        car = CarState(
            robot=self.robot,
            track=self.track,
            vehicle=VEHICLE,
            step_dt=TIMING.step_dt,
            throttle_ids=self._throttle_ids,
            steer_ids=self._steer_ids,
            suspension_ids=self._suspension_ids,
            episode_step=self.episode_length_buf,
            commands=self._commands,
            applied_wheel_torque=self._wheel_torque,
            sensors=self.scene.sensors,
            # A live reference: _apply_action updates it in place during the
            # step, and a caller should see the current value.
            wall_touched=self._wall_touched,
            lap=self._lap_view,
        )
        self._car = car
        return car

    # ──────────────────────────────────────────────────────────────────────
    #  5. The step
    # ──────────────────────────────────────────────────────────────────────

    @sealed
    def _get_observations(self) -> dict:
        # Rebuilt rather than reused: observations are produced *after* the
        # respawn, so a snapshot taken in _get_dones would hand freshly reset
        # cars their pre-crash positions.
        obs = self.compute_observations(self._build_car_state())
        expected = (self.num_envs, self.cfg.observation_space)
        if tuple(obs.shape) != expected:
            raise ValueError(
                f"compute_observations() returned {tuple(obs.shape)}, but "
                f"cfg.observation_space says it should be {expected}. The "
                "policy's input was sized from the config before the simulation "
                "started, so the two have to agree — fix whichever is wrong."
            )
        return {"policy": obs}

    @sealed
    def _get_rewards(self) -> torch.Tensor:
        # _get_dones runs first in the step and leaves its snapshot behind, so
        # reward and terminations see exactly the same numbers.
        car = self._car if self._car is not None else self._build_car_state()
        reward = self.compute_reward(car)

        if reward.shape != (self.num_envs,):
            raise ValueError(
                f"compute_reward() returned shape {tuple(reward.shape)}, but it "
                f"must be one number per car: ({self.num_envs},)."
            )

        # Published here rather than in _get_dones because reward is computed
        # *after* terminations, and RSL-RL fixes the set of logged keys from the
        # first step of each iteration — assembled any earlier and everything
        # logged from compute_reward() would be missing for the whole run.
        self.extras["log"] = {
            **self.lap_timer.log_dict(),
            **self.cfg.spawn_manager.log_dict(),
            **self._log,
        }
        return reward

    @sealed
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(failed, ran_out_of_time)``.

        While training, your terminations are the only thing that ends an
        episode early. During a measured run they are *replaced* by the crash
        rules — not added to — so that every team's cars fail for exactly the
        same reasons and a lap time compares like with like. The benchmark's
        two extra options also apply then: ``official_stall_rule`` cuts off a
        car that stops making progress, and ``terminate_on_lap`` ends the
        episode the step a valid lap is recorded.

        The lap clock is ticked here because it is measurement: it must not
        depend on how you have shaped your reward.
        """
        car = self._build_car_state()

        if self.cfg.enforce_official_rules:
            failed = rules.official_terminations(car)
            # The stall rule needs a running count of slow steps per car.
            if self.cfg.official_stall_rule:
                self._stall_steps, stalled = rules.stalled(car, self._stall_steps)
                failed |= stalled
        else:
            failed = self.compute_terminations(car)
            if failed.dtype != torch.bool:
                failed = failed.bool()

        # A lap with a wall touch in it never counts, in either mode.
        self.lap_timer.invalidate(self._wall_touched)
        self.lap_timer.update(car.pos_xy, car.nearest_idx, self.episode_length_buf)

        # During a measured run a valid lap is a finished attempt: the clock
        # has recorded it, so the car is done. The benchmark alone turns this
        # on, so training runs keep lapping.
        if self.cfg.enforce_official_rules and self.cfg.terminate_on_lap:
            failed |= self.lap_timer.just_finished

        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        # Retire whoever is out, before Isaac Lab acts on this and tries to
        # respawn them. Done here rather than in _reset_idx because a car whose
        # attempt ends on the same step it would be reset has to be caught by
        # that very reset — _get_dones runs first, so this is in time.
        if self.cfg.enforce_official_rules:
            self._retire(failed | timed_out)

        return failed, timed_out
