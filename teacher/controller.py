"""The control law: a deterministic driver, vectorized over thousands of cars.

One function of state and parameters, no memory between steps, no branching on
Python-level state. Everything is a torch tensor with one row per car, because
CMA-ES scores a whole population in one simulation.

**It reads a** :class:`~lituanicax_sdk.state.CarState` **and nothing else.** That
is the constraint that makes the whole approach worth anything: the point of the
controller is to teach a network that will only ever see an observation vector, so
a controller that peeks at the simulator would be teaching something
unlearnable. Rather than trusting discipline, the signature enforces it — the SDK
hands ``compute_observations`` exactly the surface a policy is allowed to see, and
that is the only input here. ``tests/test_controller.py`` pins the accessed
attributes with a recording proxy.

Steering is three terms added together, each of which fails in a different place,
which is why the search gets to weight them:

* **pure pursuit** aims at a point some distance ahead on the reference line. It
  is stable and it cuts corners, and its lookahead has to grow with speed or it
  oscillates.
* **curvature feedforward** sets the steering the corner needs before any error
  has built up. It is exactly right on a constant-radius corner and useless in a
  transition.
* **cross-track PD** removes whatever error the other two leave. Alone it is
  always a step behind.

Throttle is the profile's own gradient as feedforward plus a proportional
correction, with separate accelerate and brake gains because the car's limits are
asymmetric — it brakes at roughly 8 m/s² and accelerates at 6.

Those three steering terms are all *kinematic*: each assumes the car goes where
its wheels point. Measured, it does not — the steering saturates for 39% of a lap
and the wheels reach only about 75% of the angle they are commanded at cornering
speed, so the car under-rotates through exactly the corners that decide the lap.
Three more terms carry the dynamic state that notices:

* **yaw-rate shortfall** ``v·κ_ref − ψ̇`` into steering, which asks for more lock
  when the corner is not being turned tightly enough;
* **sideslip** ``β`` subtracted from steering, which *is* counter-steering — it
  falls out of the sign rather than needing a mode switch;
* **yaw-rate shortfall into braking**, gated on the steering already being at the
  stop, for the case where there is no more lock to ask for.

All three start at zero, so the law reduces exactly to the kinematic one and the
search decides whether they are worth anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lituanicax_sdk.state import CarState
from lituanicax_sdk.vehicle import VEHICLE
from tools.geometry import TrackGeometry
from tools.profile import (
    offset_path,
    periodic_basis,
    steering_ceiling,
    three_pass_profile,
)

from .params import ControllerParams


@dataclass
class Reference:
    """What the controller steers at: a line and a speed for every arc length.

    All tables are ``[M]`` torch tensors indexed by the track's own uniform
    samples, so a lookup is an interpolation rather than a search.

    Attributes:
        offset: lateral offset of the racing line, metres, positive left.
        kappa: signed curvature *of the racing line*, 1/m — not of the
            centerline, which is a different and slower curve.
        speed: target speed, m/s.
        speed_gradient: ``dv/ds`` along the racing line, 1/s. Feeds the
            feedforward acceleration ``v · dv/ds``.
    """

    offset: torch.Tensor
    kappa: torch.Tensor
    speed: torch.Tensor
    speed_gradient: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.offset.device

    def to(self, device) -> "Reference":
        return Reference(
            self.offset.to(device),
            self.kappa.to(device),
            self.speed.to(device),
            self.speed_gradient.to(device),
        )

    def as_arrays(self) -> dict[str, np.ndarray]:
        """For ``reference.npz`` — the artifact the distilled student reads."""
        return {
            "offset": self.offset.cpu().numpy(),
            "kappa": self.kappa.cpu().numpy(),
            "speed": self.speed.cpu().numpy(),
            "speed_gradient": self.speed_gradient.cpu().numpy(),
        }


#: Distance the speed gradient is measured over, metres. Short against the
#: corner scale (the tightest radius is ~0.4 m and corners run 1-2 m), long
#: against the 20 mm sample spacing.
GRADIENT_WINDOW_M = 0.15

#: Most sideslip the reference will ever ask for, radians. About 23°.
#:
#: A cap rather than a bound on the gain, because what matters is the angle the
#: car is asked to hold and that scales with curvature. Beyond this the car is
#: not cornering quickly, it is spinning, and the lap ends against a wall.
MAX_SLIP_RAD = 0.40


def _periodic_smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Moving average around a closed loop."""
    if width <= 1:
        return values
    kernel = np.ones(width) / width
    padded = np.concatenate([values[-width:], values, values[:width]])
    return np.convolve(padded, kernel, mode="same")[width : width + len(values)]


def _speed_gradient(speed: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """``dv/ds`` along the path, smoothed enough to be differentiable.

    The naive central difference between neighbouring samples does not survive
    contact with a real track. Curvature comes from a cubic spline, whose second
    derivative ripples by a fraction of a percent between knots; speed inherits
    that as ``√(a_lat/κ)``, and differentiating it over a 20 mm baseline
    amplifies it into ±0.47 1/s of alternating noise. Since the throttle
    feedforward is ``v · dv/ds``, that is about 3 m/s² of garbage oscillating at
    every sample — injected straight into the throttle command at 30 Hz.

    So the speed is smoothed over the measurement window before being
    differentiated, and the difference is taken across it rather than between
    neighbours. On a constant-radius circle this returns ~0, which is the right
    answer and what ``tests/test_controller.py`` checks.
    """
    spacing = float(np.mean(segment))
    half = max(1, int(round(GRADIENT_WINDOW_M / max(spacing, 1e-9))))

    smoothed = _periodic_smooth(speed, half)
    arc = np.concatenate([[0.0], np.cumsum(segment)])[:-1]
    total = float(segment.sum())
    # Modular difference, so the window wraps across the start/finish line.
    span = (np.roll(arc, -half) - np.roll(arc, half)) % total
    return (np.roll(smoothed, -half) - np.roll(smoothed, half)) / np.maximum(span, 1e-9)


def build_reference(
    geometry: TrackGeometry,
    params: ControllerParams,
    *,
    v_max: float,
    wheelbase_m: float = 0.224,
) -> Reference:
    """Turn a parameter vector into the line and speed profile it describes.

    Runs once per candidate on the CPU, not once per simulation step.

    The speed profile is *derived* from the line rather than searched directly —
    see :mod:`teacher.params` for why — and then scaled by the candidate's
    ``speed_scale`` knots, which is what lets the search trail-brake and pick late
    apexes without ever proposing a physically impossible profile.
    """
    count = geometry.num_samples
    line = periodic_basis(count, len(params.line)) @ np.asarray(
        params.line, dtype=np.float64
    )
    scale = periodic_basis(count, len(params.speed_scale)) @ np.asarray(
        params.speed_scale, dtype=np.float64
    )

    _, segment, kappa = offset_path(geometry, line)
    # What the *steering* allows, alongside what the tyres allow. Without it the
    # profile specifies radii the wheels cannot be pointed round at the speed
    # asked, and the controller spends half the lap saturated and running wide.
    ceiling = steering_ceiling(
        kappa,
        wheelbase_m=wheelbase_m,
        max_steer_rad=VEHICLE.max_steer_rad,
        steer_ratio=params.steer_ratio_eff,
        v_max=v_max,
    )
    speed = three_pass_profile(
        segment,
        kappa,
        a_lat=params.a_lat_eff,
        a_accel=params.a_accel_eff,
        a_brake=params.a_brake_eff,
        v_max=v_max,
        steer_ceiling=ceiling,
    )
    speed = np.clip(speed * scale, 0.1, v_max)
    gradient = _speed_gradient(speed, segment)

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=geometry.device)  # noqa: E731
    return Reference(
        offset=to_tensor(line),
        kappa=to_tensor(kappa),
        speed=to_tensor(speed),
        speed_gradient=to_tensor(gradient),
    )


#: Parameters the control law reads at every step. The line and speed profile are
#: already baked into the :class:`Reference`; these are the feedback half, plus
#: ``a_accel_eff`` which normalizes the throttle feedforward.
GAIN_NAMES = (
    "k_v",
    "L_0",
    "L_min",
    "L_max",
    "w_pp",
    "w_ff",
    "k_e",
    "k_d",
    "k_p_accel",
    "k_p_brake",
    "k_ff",
    "a_accel_eff",
    "k_r",
    "k_beta",
    "k_rotate",
    "slip_gain",
)


def stack_references(references: list[Reference]) -> Reference:
    """Combine per-candidate references into ``[C, M]`` tables."""
    return Reference(
        offset=torch.stack([r.offset for r in references]),
        kappa=torch.stack([r.kappa for r in references]),
        speed=torch.stack([r.speed for r in references]),
        speed_gradient=torch.stack([r.speed_gradient for r in references]),
    )


class Controller:
    """A deterministic driver. Call it with a ``CarState``, get actions back.

    Args:
        geometry: the track, resampled.
        reference: the line and speed profile from :func:`build_reference`, or a
            ``[C, M]`` stack from :func:`stack_references`.
        params: one :class:`~teacher.params.ControllerParams`, or a list of ``C``
            of them when driving a whole population at once.
        wheelbase_m: measured by the Phase 0 probe, not taken from a spec sheet.
        max_steer_rad: the locked steering limit, ``VEHICLE.max_steer_rad``.
        rows: ``[N]`` which candidate each car is driving. Required when ``params``
            is a list, and it is what lets CMA-ES score 256 candidates across
            2560 environments in one simulation.

    One implementation serves both cases. The alternative — a separate batched
    controller — would be two copies of the control law to keep in step, and the
    one that drifts is the one that produced the lap time.
    """

    def __init__(
        self,
        geometry: TrackGeometry,
        reference: Reference,
        params: ControllerParams | list[ControllerParams],
        *,
        wheelbase_m: float,
        max_steer_rad: float,
        rows: torch.Tensor | None = None,
    ):
        self.geometry = geometry
        self.reference = reference
        self.wheelbase_m = float(wheelbase_m)
        self.max_steer_rad = float(max_steer_rad)
        self.rows = rows

        if isinstance(params, ControllerParams):
            if rows is not None:
                raise ValueError("rows only makes sense with a list of params.")
            self.params = params
            self.gains = {name: float(getattr(params, name)) for name in GAIN_NAMES}
        else:
            if rows is None:
                raise ValueError(
                    "a list of params needs rows to say which car drives which."
                )
            if reference.offset.dim() != 2:
                raise ValueError(
                    "a list of params needs stacked [C, M] reference tables; "
                    "see stack_references()."
                )
            self.params = params[0]
            device = geometry.device
            self.gains = {
                name: torch.tensor(
                    [float(getattr(p, name)) for p in params],
                    dtype=torch.float32,
                    device=device,
                )[rows]
                for name in GAIN_NAMES
            }

    # ──────────────────────────────────────────────────────────────────────

    def __call__(self, car: CarState) -> torch.Tensor:
        """Throttle and steering for every car, ``[N, 2]`` in ``[-1, 1]``."""
        geometry, reference, gains = self.geometry, self.reference, self.gains

        # ── Where am I, relative to the line I meant to be on ─────────────
        s, offset = geometry.project(car.pos_xy)
        error = offset - self._at(reference.offset, s)

        # How fast that error is growing. Read from the velocity rather than
        # differenced between steps, so the controller stays memoryless — which
        # is what lets a whole CMA-ES population share one environment without
        # any per-candidate state to reset.
        normal = geometry.interp_vec(s, geometry.normal)
        error_rate = (car.lin_vel_w[:, :2] * normal).sum(dim=-1)

        # ── Steering ──────────────────────────────────────────────────────
        speed = car.speed_forward
        lookahead = torch.clamp(
            gains["k_v"] * speed + gains["L_0"], gains["L_min"], gains["L_max"]
        )

        ahead = s + lookahead
        target = geometry.point_at(ahead, self._at(reference.offset, ahead))
        to_target = target - car.pos_xy
        cos_yaw, sin_yaw = torch.cos(car.yaw), torch.sin(car.yaw)
        forward = to_target[:, 0] * cos_yaw + to_target[:, 1] * sin_yaw
        left = -to_target[:, 0] * sin_yaw + to_target[:, 1] * cos_yaw
        alpha = torch.atan2(left, forward)

        pursuit = torch.atan2(
            2.0 * self.wheelbase_m * torch.sin(alpha), lookahead.clamp(min=1e-3)
        )
        feedforward = torch.atan(self.wheelbase_m * self._at(reference.kappa, s))
        feedback = -(gains["k_e"] * error + gains["k_d"] * error_rate)

        # ── What the car is actually doing, as opposed to where it is ─────
        # Everything above is kinematic: it assumes the car goes where the wheels
        # point. It does not, and the diagnostic says so — the steering is
        # saturated for 39% of a lap and the wheels only reach about 75% of the
        # angle they are asked for, so the car under-rotates through exactly the
        # corners that decide the lap. These two terms are the whole dynamic
        # state the controller has, and both are free in `CarState`.

        # Is the car turning as fast as the corner needs? A corner of curvature
        # kappa taken at v needs a yaw rate of v*kappa. Any shortfall is rotation
        # the geometry did not deliver, and asking for more steering is the first
        # thing to try.
        yaw_wanted = speed * self._at(reference.kappa, s)
        yaw_short = yaw_wanted - car.yaw_rate

        # How sideways the car is. Positive when the rear has stepped out, and
        # subtracting it *is* counter-steering — it falls out of the sign rather
        # than needing a mode switch, so there is no separate "now we are
        # drifting" branch to get wrong. k_beta at zero recovers the old law
        # exactly, so the search decides whether any of this is worth using.
        sideslip = torch.atan2(car.speed_lateral, speed.abs().clamp(min=0.05))

        # How sideways the car is *meant* to be, and this is the part that stops
        # the law being purely kinematic.
        #
        # Driving the sideslip to zero says the car should always go where its
        # wheels point. That is what a well-behaved road car does and it is not
        # what a fast lap here looks like: the diagnostic has the steering at the
        # stop for 41% of the lap, which means the wheels cannot be pointed round
        # the corner at all, and the only remaining way to rotate is to let the
        # car slide. Asking for a slip angle proportional to how tight the corner
        # is, and in the direction it turns, makes that a *plan* rather than an
        # error to suppress — the same feedback then builds the slide going in
        # and catches it coming out, with no mode switch.
        #
        # One parameter rather than a profile, because the population that
        # reproduces is small and twenty more knots would not be searchable. At
        # slip_gain = 0 this is exactly the law that produced the 15.067 s
        # baseline, so the search cannot do worse than where it started.
        slip_target = torch.clamp(
            gains["slip_gain"] * self._at(reference.kappa, s),
            -MAX_SLIP_RAD,
            MAX_SLIP_RAD,
        )

        steer = (
            gains["w_pp"] * pursuit
            + gains["w_ff"] * feedforward
            + feedback
            + gains["k_r"] * yaw_short
            - gains["k_beta"] * (sideslip - slip_target)
        ).clamp(-self.max_steer_rad, self.max_steer_rad)

        # ── Throttle ──────────────────────────────────────────────────────
        wanted = self._at(reference.speed, s)
        error_speed = wanted - speed
        # v * dv/ds is the acceleration the profile itself is asking for.
        wanted_accel = wanted * self._at(reference.speed_gradient, s)
        # ...but that describes a car already *on* the profile. A car 2.9 m/s
        # below it, sitting in a braking zone because there is a corner coming,
        # gets a feedforward of -2.9 against a proportional term of +2.9, and the
        # two cancel to a throttle of 0.03. Measured, in Gate 1: cars crawling at
        # 0.15 m/s for forty steps, unable to start, until the stall rule retired
        # them 5.5 m into the lap.
        #
        # So the feedforward may help the speed error but never fight it. When
        # the car is slow it does not need to hear that it should be slowing
        # down; the profile dropping below its speed will say so soon enough,
        # through the proportional term, which is the part that knows where the
        # car actually is.
        wanted_accel = torch.where(
            error_speed > 0.0, wanted_accel.clamp(min=0.0), wanted_accel.clamp(max=0.0)
        )
        gain = torch.where(
            error_speed > 0.0,
            torch.as_tensor(gains["k_p_accel"], device=speed.device).expand_as(speed),
            torch.as_tensor(gains["k_p_brake"], device=speed.device).expand_as(speed),
        )
        reference_accel = torch.as_tensor(
            gains["a_accel_eff"], device=speed.device
        ).clamp(min=1e-3)
        throttle = gain * error_speed + gains["k_ff"] * wanted_accel / reference_accel

        # ── Rotating the car when the steering cannot ─────────────────────
        # Once the wheels are at the stop and the car still is not turning
        # enough, steering has nothing left to give and the only actuator with
        # any authority over yaw is the longitudinal one. Lifting or braking
        # moves load onto the front axle and sheds the speed that made the corner
        # too tight in the first place, so the deficit closes from both sides.
        #
        # Gated on the steering actually being saturated, so on a car with
        # authority in hand this term is inert and the law is the one that
        # produced the baseline. k_rotate at zero disables it outright, which is
        # where the warm start puts it — the search decides whether it earns its
        # place. Braking rather than throttle is deliberate: this car drives all
        # four wheels, so opening the throttle mid-corner consumes front grip and
        # pushes the nose wide, which is the opposite of what is wanted.
        at_the_stop = (steer.abs() >= self.max_steer_rad * 0.99).to(steer.dtype)
        throttle = throttle - gains["k_rotate"] * at_the_stop * yaw_short.abs()

        return torch.stack(
            [throttle.clamp(-1.0, 1.0), steer / self.max_steer_rad], dim=-1
        )

    def _at(self, table: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Interpolate a reference table at arc length ``s``, wrapping the loop.

        Picks the right row per car when the tables are a ``[C, M]`` population
        stack, using the two-index trick in
        :meth:`~tools.geometry.TrackGeometry.interp`.
        """
        return self.geometry.interp(s, table, rows=self.rows)
