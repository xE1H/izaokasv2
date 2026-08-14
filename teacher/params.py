"""The controller's parameters: what CMA-ES searches over.

Eighty-nine numbers in five groups:

======================  =====  ===============================================
group                   count  what it does
======================  =====  ===============================================
``line``                  40   racing line, as lateral offset control points
``speed_scale``           30   local multiplier on the quasi-static profile
effective limits           4   a_lat, a_accel, a_brake, kappa_max
gains                     11   lookahead, steering blend, feedback
dynamic terms              4   yaw-rate, sideslip, braking, steering ceiling
======================  =====  ===============================================

**The dynamic terms start at zero.** With ``k_r``, ``k_beta`` and ``k_rotate``
all zero the law is exactly the kinematic one that produced the 17.1 s baseline,
so the search is never handed a worse starting point than it had — it decides
whether any of them earn their place. They exist because the diagnostic says the
steering is saturated for 39% of a lap: the geometric terms have run out of
authority there, and these three are the only ones that can notice.

**Why the speed profile is not parameterized directly.** A free speed at every
point is mostly physically infeasible, and a search that starts there spends its
first generations discovering that rather than driving. Instead the profile is
*derived* from the line by :func:`~tools.profile.three_pass_profile`, and the
search only scales it locally. Every point in this space is a sane driver, and so
is every point near it — which is the property that makes a 70-dimensional
evolutionary search tractable at all.

**Why the four limits are searched and not just measured.** They start at the
Phase 0 probe's values, but the quasi-static model they feed is wrong in ways the
probe cannot see: no load transfer, no combined-slip ellipse, no throttle
rotation. Letting the search move them lets it absorb that error. ``kappa_max``
in particular is a rigid-kinematic bound (``L_wb / tan(max_steer)``) that a real
car beats when the rear slips and misses when it understeers, so pinning it would
be pinning the wrong number.

CMA-ES works in a normalized space where every parameter spans ``[0, 1]``, so one
initial ``sigma`` is meaningful for all seventy at once — see
:meth:`ControllerParams.to_normalized`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

#: Racing-line control points. Matches the resolution the warm-start solve uses.
LINE_POINTS = 40

#: Knots in the speed multiplier.
#:
#: Doubled from 15 once the search showed it needed the resolution: at 15 knots a
#: 50 m lap gets one every 3.3 m, and the corners that decide this lap are 1-2 m
#: long, so a single knot had to serve a corner and the straight either side of
#: it. The quasi-static profile it multiplies is wrong in ways that vary corner
#: by corner — it has no load transfer and no rotation — so the correction has to
#: be able to vary corner by corner too.
SPEED_POINTS = 30


@dataclass(frozen=True)
class Bound:
    """An inclusive range, and the value the warm start uses inside it."""

    low: float
    high: float

    def clip(self, value):
        return np.clip(value, self.low, self.high)


#: Bounds for every scalar. The line and speed blocks are bounded separately
#: below, since they are arrays.
SCALAR_BOUNDS: dict[str, Bound] = {
    # ── Effective limits, in SI. Wide enough to absorb model error in both
    #    directions without letting the search invent a different car.
    "a_lat_eff": Bound(3.0, 20.0),
    "a_accel_eff": Bound(1.0, 15.0),
    "a_brake_eff": Bound(1.0, 20.0),
    # Curvature the controller believes it can hold, 1/m. 1/0.9 to 1/0.25 m.
    "kappa_max_eff": Bound(1.1, 4.0),
    # ── Lookahead. Speed-scaled, because a fixed lookahead is either twitchy at
    #    speed or lazy in the slow corners.
    # Seconds of travel. Widened from 0.6 after the search pinned itself at
    # 0.575 — a bound the optimizer is pressed against is a bound that is
    # choosing the answer.
    "k_v": Bound(0.0, 1.2),
    "L_0": Bound(0.02, 1.0),  # metres at a standstill
    "L_min": Bound(0.02, 0.6),
    "L_max": Bound(0.2, 3.0),
    # ── Steering blend. Pure pursuit and curvature feedforward both produce a
    #    steering angle; the search decides how much to trust each.
    #    Widened from 2.5: the wheels only reach about 75% of the angle they are
    #    commanded at cornering speed, so a gain above one is the car's physics
    #    rather than a mistuning, and the search had pushed w_ff to 2.40 of 2.5.
    "w_pp": Bound(0.0, 4.0),
    "w_ff": Bound(0.0, 4.0),
    # ── Cross-track PD, in radians of steer per metre and per m/s.
    "k_e": Bound(0.0, 8.0),
    "k_d": Bound(0.0, 3.0),
    # ── Longitudinal. Separate gains because the car's limits are asymmetric:
    #    it brakes harder than it accelerates.
    "k_p_accel": Bound(0.0, 6.0),
    "k_p_brake": Bound(0.0, 6.0),
    "k_ff": Bound(0.0, 3.0),
    # ── The dynamic terms. Everything above treats the car as going where its
    #    wheels point; these three are what notice that it does not.
    #    Yaw-rate shortfall into steering, radians of steer per rad/s.
    "k_r": Bound(0.0, 1.5),
    #    Sideslip into steering — this is the counter-steer, and it has to be
    #    allowed to be strong enough to catch a slide it did not intend.
    "k_beta": Bound(0.0, 3.0),
    #    Yaw-rate shortfall into *braking*, once the steering is at the stop.
    #    Zero disables it, which is where the warm start starts.
    "k_rotate": Bound(0.0, 2.0),
    #    What fraction of the commanded steering angle the profile believes the
    #    wheels reach at top speed. Measured at 0.62; searched because the linear
    #    model of the loss is a simplification and because a car that is willing
    #    to slide does not need the wheels to point all the way round the corner.
    #    At 1.0 the ceiling is inert and the profile is the one that came before.
    "steer_ratio_eff": Bound(0.35, 1.0),
}

#: Corridor for the line, metres either side of the centerline. Matches
#: :data:`tools.profile.DEFAULT_HALF_WIDTH_M`.
LINE_BOUND = Bound(-0.18, 0.18)

#: Speed multiplier on the quasi-static profile.
#:
#: Widened from (0.7, 1.15) because the search pressed against *both* ends — its
#: knots ran 0.704 to 1.136 — and a parameter pinned at its bounds is a bound
#: choosing the answer. The asymmetry is gone with it: the original reasoning was
#: that being optimistic ends an attempt while being pessimistic only costs time,
#: which is true of a single attempt and false of the competition, where the
#: fastest of ten counts and nine failures are free.
SPEED_BOUND = Bound(0.5, 1.5)

#: Total number of searched parameters.
DIMENSION = LINE_POINTS + SPEED_POINTS + len(SCALAR_BOUNDS)


@dataclass
class ControllerParams:
    """One candidate driver.

    ``line`` and ``speed_scale`` are control points, not per-sample values: they
    are resampled onto the track's own discretization by
    :func:`teacher.controller.build_reference`.
    """

    line: np.ndarray = field(default_factory=lambda: np.zeros(LINE_POINTS))
    speed_scale: np.ndarray = field(default_factory=lambda: np.ones(SPEED_POINTS))

    a_lat_eff: float = 8.0
    a_accel_eff: float = 6.0
    a_brake_eff: float = 8.0
    kappa_max_eff: float = 1.0 / 0.45

    k_v: float = 0.15
    L_0: float = 0.2
    L_min: float = 0.1
    L_max: float = 1.5

    w_pp: float = 1.0
    w_ff: float = 1.0

    k_e: float = 2.0
    k_d: float = 0.5

    k_p_accel: float = 2.0
    k_p_brake: float = 2.0
    k_ff: float = 1.0
    k_r: float = 0.0
    k_beta: float = 0.0
    k_rotate: float = 0.0
    steer_ratio_eff: float = 1.0

    # ──────────────────────────────────────────────────────────────────────
    #  Vector conversion
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def scalar_names() -> list[str]:
        """Scalar field names, in the order they occupy the vector."""
        return [
            f.name
            for f in fields(ControllerParams)
            if f.name not in ("line", "speed_scale")
        ]

    def to_vector(self) -> np.ndarray:
        """Flatten to ``[DIMENSION]`` in SI units."""
        return np.concatenate(
            [
                np.asarray(self.line, dtype=np.float64).ravel(),
                np.asarray(self.speed_scale, dtype=np.float64).ravel(),
                np.array([getattr(self, name) for name in self.scalar_names()]),
            ]
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "ControllerParams":
        """Rebuild from :meth:`to_vector`, clipping each value into its bounds.

        Clipping rather than raising: CMA-ES samples a Gaussian and will step
        outside the box, and the honest handling is to evaluate the nearest legal
        driver rather than to reject the sample and bias the distribution.
        """
        vector = np.asarray(vector, dtype=np.float64).ravel()
        if vector.size != DIMENSION:
            raise ValueError(f"expected {DIMENSION} parameters, got {vector.size}.")

        cut = LINE_POINTS + SPEED_POINTS
        params = cls(
            line=LINE_BOUND.clip(vector[:LINE_POINTS]),
            speed_scale=SPEED_BOUND.clip(vector[LINE_POINTS:cut]),
        )
        for name, value in zip(cls.scalar_names(), vector[cut:]):
            setattr(params, name, float(SCALAR_BOUNDS[name].clip(value)))
        return params

    # ──────────────────────────────────────────────────────────────────────
    #  Normalized space, for the optimizer
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def bounds_vector() -> tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` for every searched parameter, ``[DIMENSION]`` each."""
        low = np.concatenate(
            [
                np.full(LINE_POINTS, LINE_BOUND.low),
                np.full(SPEED_POINTS, SPEED_BOUND.low),
                np.array(
                    [SCALAR_BOUNDS[n].low for n in ControllerParams.scalar_names()]
                ),
            ]
        )
        high = np.concatenate(
            [
                np.full(LINE_POINTS, LINE_BOUND.high),
                np.full(SPEED_POINTS, SPEED_BOUND.high),
                np.array(
                    [SCALAR_BOUNDS[n].high for n in ControllerParams.scalar_names()]
                ),
            ]
        )
        return low, high

    def to_normalized(self) -> np.ndarray:
        """Map into ``[0, 1]`` per parameter, so one ``sigma`` suits all of them.

        Without this the covariance would start wildly anisotropic — the line
        offsets span 0.36 m while ``a_lat_eff`` spans 17 m/s², and CMA-ES would
        spend its early generations learning the scaling instead of the track.
        """
        low, high = self.bounds_vector()
        return (self.to_vector() - low) / (high - low)

    @classmethod
    def from_normalized(cls, normalized: np.ndarray) -> "ControllerParams":
        low, high = cls.bounds_vector()
        clipped = np.clip(np.asarray(normalized, dtype=np.float64).ravel(), 0.0, 1.0)
        return cls.from_vector(low + clipped * (high - low))

    # ──────────────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        out: dict = {
            "line": np.asarray(self.line).tolist(),
            "speed_scale": np.asarray(self.speed_scale).tolist(),
        }
        out.update({name: float(getattr(self, name)) for name in self.scalar_names()})
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ControllerParams":
        params = cls(
            line=np.asarray(data["line"], dtype=np.float64),
            speed_scale=np.asarray(data["speed_scale"], dtype=np.float64),
        )
        for name in cls.scalar_names():
            if name in data:
                setattr(params, name, float(data[name]))
        return params

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ControllerParams":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def describe(self) -> str:
        return (
            f"[params] line +/-{np.abs(self.line).max():.3f} m, "
            f"speed x[{np.min(self.speed_scale):.2f},{np.max(self.speed_scale):.2f}], "
            f"a_lat {self.a_lat_eff:.1f}, a_accel {self.a_accel_eff:.1f}, "
            f"a_brake {self.a_brake_eff:.1f}, R_min {1.0 / self.kappa_max_eff:.3f} m"
        )
