"""What the Phase 0 probe measured about the car.

The schema shared by the writer (:mod:`tools.probe`, which needs Isaac Sim) and
the readers (:mod:`teacher.warmstart` and :mod:`teacher.optimize`, which do not).
Kept in its own module so neither has to duplicate the field names, and so the
warm start can be developed and tested without a simulator.

Not to be confused with :mod:`lituanicax_sdk.dynamics`, which is the *locked*
drive model — the torque curve the competition fixes. This is the car's observed
behaviour, which is a different thing and has to be measured rather than read off
the config: the SDK publishes a motor no-load speed and a drive torque, not a
cornering limit or a rollover threshold.

Every default here is a **guess**, present only so the rest of the pipeline can be
exercised before a GPU is rented. They are deliberately conservative, and
:attr:`Measured.measured` says whether a real probe has run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("artifacts/dynamics.json")


@dataclass
class Measured:
    """The seven Phase 0 probe results, plus what they imply.

    Attributes:
        wheelbase_m: front-to-rear axle distance, from geometry and confirmed by
            driving a full-lock circle.
        a_accel_m_s2: representative forward acceleration. The real curve falls
            off with speed (the motor follows a torque-speed curve), so this is
            the figure the quasi-static profile uses, not a peak.
        a_brake_m_s2: braking deceleration, positive.
        a_lat_max_m_s2: the highest lateral acceleration the car **sustained
            without tipping**. A steering angle that put the car on its roof
            contributes to :attr:`rollover_a_lat_m_s2` instead — counting its peak
            here as well would make the two equal and hide which one binds.
        rollover_a_lat_m_s2: the lowest lateral acceleration at which the up-axis
            dropped below 0.3. ``None`` means the car never tipped at any steering
            angle tried.
        steer_lag_steps: control steps for the steering angle to reach its
            commanded value. Feeds lookahead tuning.
        v_max_m_s: top speed on the flat. The SDK publishes this
            (``VEHICLE.motor_no_load_speed_m_s``), so it is a cross-check rather
            than a discovery.
        accel_curve: optional ``[(speed, accel), ...]`` samples, for reference.
        measured: False while these are the built-in guesses.
    """

    wheelbase_m: float = 0.26
    a_accel_m_s2: float = 6.0
    a_brake_m_s2: float = 8.0
    a_lat_max_m_s2: float = 8.0
    rollover_a_lat_m_s2: float | None = None
    steer_lag_steps: float = 2.0
    v_max_m_s: float = 6.7
    accel_curve: list[tuple[float, float]] = field(default_factory=list)
    full_lock_radius_m: float | None = None
    steer_ratio_at_speed: float = 1.0
    measured: bool = False

    # ──────────────────────────────────────────────────────────────────────

    @property
    def r_min_m(self) -> float:
        """Minimum turn radius: the circle the car actually drove, if it drove one.

        The number that decides whether the tightest corner is drivable by
        steering alone. Compare against
        :func:`tools.profile.widest_achievable_radius`.

        The geometric ``L_wb / tan(max_steer)`` is the fallback, not the answer,
        because it assumes the wheels reach the commanded angle. They do not: the
        steering is an effort-limited servo and at full lock the measured car held
        0.447 rad of the 0.488 it was asked for. The difference is not academic —
        the geometric figure says 0.422 m against a corridor that can deliver
        0.545 m, comfortable; the circle the car actually drove is 0.523 m, which
        passes by four percent.
        """
        if self.full_lock_radius_m is not None:
            return self.full_lock_radius_m

        import math

        from lituanicax_sdk.vehicle import VEHICLE

        return self.wheelbase_m / math.tan(VEHICLE.max_steer_rad)

    @property
    def geometric_r_min_m(self) -> float:
        """``L_wb / tan(max_steer)`` — what the kinematics alone would give."""
        import math

        from lituanicax_sdk.vehicle import VEHICLE

        return self.wheelbase_m / math.tan(VEHICLE.max_steer_rad)

    @property
    def a_lat_effective_m_s2(self) -> float:
        """The cornering limit that actually binds.

        **If the car tips before it slides, the rollover threshold replaces the
        grip limit everywhere.** On a 1.27 kg chassis that is a live possibility,
        and getting it the wrong way round would put every speed target above
        what the car survives.
        """
        if self.rollover_a_lat_m_s2 is None:
            return self.a_lat_max_m_s2
        return min(self.a_lat_max_m_s2, self.rollover_a_lat_m_s2)

    @property
    def rollover_limited(self) -> bool:
        """Whether the car was seen to tip at all.

        Deliberately *not* ``rollover < a_lat_max``. The probe sweeps steering
        monotonically, so lateral acceleration rises with steering angle and a car
        that tips does so at the *highest* accelerations it reached — meaning the
        tip threshold is always at or above the highest figure it sustained, and
        that comparison could never fire.

        What the flag has to mean is "the ceiling here is tipping, not sliding":
        the grip limit was never found because the car went over first, so pushing
        past :attr:`a_lat_max_m_s2` risks a roll rather than a slide. That changes
        how much margin the speed targets need, which is the decision this informs.
        """
        return self.rollover_a_lat_m_s2 is not None

    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path = DEFAULT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        # Derived, but worth having in the file a human reads.
        payload["r_min_m"] = self.r_min_m
        payload["a_lat_effective_m_s2"] = self.a_lat_effective_m_s2
        payload["rollover_limited"] = self.rollover_limited
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "Measured":
        """Read a probe result, or return the guesses if none exists yet."""
        path = Path(path)
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def describe(self) -> str:
        source = "measured" if self.measured else "GUESSED — run tools.probe"
        limit = "rollover" if self.rollover_limited else "grip"
        return (
            f"[car:{source}] L_wb {self.wheelbase_m:.3f} m "
            f"-> R_min {self.r_min_m:.3f} m | "
            f"a_lat {self.a_lat_effective_m_s2:.1f} ({limit}-limited), "
            f"a_accel {self.a_accel_m_s2:.1f}, a_brake {self.a_brake_m_s2:.1f} m/s^2 | "
            f"v_max {self.v_max_m_s:.2f} m/s, "
            f"steer lag {self.steer_lag_steps:.1f} steps"
        )
