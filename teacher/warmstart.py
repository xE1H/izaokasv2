"""The starting parameter vector, so CMA-ES begins from a driver rather than noise.

    python -m teacher.warmstart            # writes artifacts/teacher-warmstart.json
    python -m teacher.warmstart --report   # and prints what it decided, and why

Nothing here needs a simulator. That is the point: the warm start can be built and
checked before any GPU time is bought, and Gate 1 — *does the unoptimized
controller complete a lap* — is then a single run rather than a debugging session.

Three groups of parameters, three sources:

* **the line** comes from :func:`tools.profile.racing_line_offsets`, bounded to a
  radius the car can actually steer;
* **the effective limits** come from the Phase 0 probe (:mod:`tools.measured`);
* **the gains** come from closed-form control design, not from taste — see
  :func:`critically_damped_gains`.

All of them are then free to move: the search treats every one as a parameter, and
the quasi-static model behind the first two is wrong in ways only the simulator
can reveal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lituanicax_sdk.track import Track
from lituanicax_sdk.vehicle import TIMING
from lituanicax_sdk.tracks import OFFICIAL
from tools.geometry import TrackGeometry
from tools.measured import Measured
from tools.profile import (
    DEFAULT_HALF_WIDTH_M,
    offset_path,
    periodic_basis,
    profile_lap_time,
    racing_line_offsets,
    widest_achievable_radius,
)

from .params import (
    LINE_POINTS,
    SCALAR_BOUNDS,
    SPEED_POINTS,
    ControllerParams,
)

DEFAULT_PATH = Path("artifacts/teacher-warmstart.json")

#: Speed the feedback gains are designed at. The middle of the car's range, so
#: the closed loop is neither sluggish in the slow corners nor twitchy on the
#: straight — the search retunes it either way.
DESIGN_SPEED_M_S = 3.0

#: How fast the cross-track loop should settle, rad/s. About 0.5 s to correct,
#: which is fast against a 30 Hz control rate and slow against the ~2 s it takes
#: to cross a corner.
DESIGN_FREQUENCY_RAD_S = 4.0


def critically_damped_gains(
    wheelbase_m: float,
    *,
    speed: float = DESIGN_SPEED_M_S,
    frequency: float = DESIGN_FREQUENCY_RAD_S,
) -> tuple[float, float]:
    """``(k_e, k_d)`` for a critically damped cross-track response.

    For a kinematic bicycle at speed ``v``, a steering angle ``δ`` turns the body
    at ``v/L``, so the cross-track error obeys ``ë = (v²/L) δ``. Feeding back
    ``δ = -(k_e e + k_d ė)`` gives

        ``ë + (v² k_d / L) ė + (v² k_e / L) e = 0``

    which is a damped oscillator with ``ω² = v² k_e / L`` and damping
    ``2ζω = v² k_d / L``. Solving for ``ζ = 1`` at the design speed:

        ``k_e = ω² L / v²``     ``k_d = 2 ω L / v²``

    Derived rather than guessed because the two gains are not independent — a
    plausible-looking pair that is badly under-damped makes the car weave, and
    the search would then spend generations undoing it.
    """
    scale = wheelbase_m / max(speed**2, 1e-9)
    return frequency**2 * scale, 2.0 * frequency * scale


def build(
    geometry: TrackGeometry,
    car: Measured,
    *,
    half_width: float = DEFAULT_HALF_WIDTH_M,
) -> tuple[ControllerParams, dict]:
    """The warm-start parameters, and a report on how they were chosen.

    The report is the interesting half: it records whether the car can steer this
    track at all, which is the question Gate 0 exists to answer.
    """
    widest, _ = widest_achievable_radius(geometry, half_width=half_width)
    centerline_radius = 1.0 / float(geometry.kappa.abs().max())
    steerable = car.r_min_m <= widest

    # Ask for the tightest radius the car can hold. If that is beyond what the
    # corridor can deliver, ask for the best the corridor has and record that the
    # track needs more than kinematic steering.
    requested = car.r_min_m if steerable else widest
    # Solved in exactly the basis the search moves, so the warm start is
    # representable without loss. Refitting an 80-point solution onto 40 control
    # points afterwards smooths the apexes off — 137 mm of error at worst against
    # a 180 mm corridor, which can also break the curvature bound the solve just
    # went to the trouble of enforcing.
    line, line_report = racing_line_offsets(
        geometry,
        a_lat=car.a_lat_effective_m_s2,
        v_max=car.v_max_m_s,
        half_width=half_width,
        kappa_max=1.0 / requested,
        control_points=LINE_POINTS,
    )
    basis = periodic_basis(geometry.num_samples, LINE_POINTS)
    control = np.clip(
        line_report.get("coefficients", np.zeros(LINE_POINTS)), -half_width, half_width
    )

    steer_lag_s = float(car.steer_lag_steps) * TIMING.step_dt
    # The cross-track loop cannot be faster than the actuator it drives. A 4 rad/s
    # design against a 0.33 s dead time is 76 degrees of phase lag at crossover,
    # which is not a stability margin, it is an oscillator. Keep the loop slow
    # enough that the lag costs about 0.3 rad of phase.
    frequency = min(DESIGN_FREQUENCY_RAD_S, 0.3 / max(steer_lag_s, 1e-3))
    k_e, k_d = critically_damped_gains(car.wheelbase_m, frequency=frequency)
    params = ControllerParams(
        line=control,
        speed_scale=np.ones(SPEED_POINTS),
        a_lat_eff=car.a_lat_effective_m_s2,
        a_accel_eff=car.a_accel_m_s2,
        a_brake_eff=car.a_brake_m_s2,
        kappa_max_eff=1.0 / requested,
        # Lookahead, in seconds of travel plus a floor. The seconds are the
        # measured steering dead time: the wheels take about ten control steps to
        # reach a commanded angle, so by the time they get there the car has
        # moved on, and aiming at where it is now is aiming at the past. Gate 1
        # failed with k_v = 0.15 -- half the lag -- and the trace shows exactly
        # that failure: the steering command swinging full-scale between
        # consecutive steps while the wheels chase it, saturated 28% of the time,
        # and the line missed by up to 285 mm in a 200 mm corridor.
        k_v=min(steer_lag_s, SCALAR_BOUNDS["k_v"].high),
        L_0=2.0 * car.wheelbase_m,
        L_min=car.wheelbase_m,
        # Room for the lag at top speed: 6.9 m/s x 0.33 s is 2.3 m on its own.
        L_max=min(
            steer_lag_s * car.v_max_m_s + 2.0 * car.wheelbase_m,
            SCALAR_BOUNDS["L_max"].high,
        ),
        # Left at 1.0 deliberately. The wheels only reach about 0.62 of the
        # commanded angle at speed, but scaling both blend weights up by 1/0.62
        # to compensate made Gate 1 worse, not better: the two terms already
        # double-count the reference curvature, so the sum saturates against the
        # 0.488 rad limit in every tight corner and the controller loses the
        # ability to steer *more* when it needs to. Compensating for the servo is
        # a job for one knob, and CMA-ES has both of these.
        w_pp=1.0,
        w_ff=1.0,
        k_e=k_e,
        k_d=k_d,
        # Proportional speed control, sized so a 1 m/s error asks for most of the
        # available command. Asymmetric because braking is stronger.
        k_p_accel=1.0,
        k_p_brake=1.5,
        k_ff=1.0,
    )

    fitted = basis @ control
    evaluation = dict(
        a_lat=car.a_lat_effective_m_s2,
        a_accel=car.a_accel_m_s2,
        a_brake=car.a_brake_m_s2,
        v_max=car.v_max_m_s,
    )
    report = {
        "car": car.describe(),
        "car_measured": car.measured,
        "r_min_m": car.r_min_m,
        "geometric_r_min_m": car.geometric_r_min_m,
        "steer_ratio_at_speed": float(car.steer_ratio_at_speed),
        "centerline_min_radius_m": centerline_radius,
        "widest_achievable_radius_m": widest,
        "track_is_steerable": bool(steerable),
        "requested_radius_m": requested,
        "line": {k: v for k, v in line_report.items() if k != "coefficients"},
        "line_fit_error_mm": float(np.abs(fitted - line).max() * 1000.0),
        "peak_offset_m": float(np.abs(fitted).max()),
        # The bound the solve enforced must survive being expressed in the
        # search's own basis, or the controller starts from a line it cannot steer.
        "fitted_peak_radius_m": float(
            1.0 / max(float(np.abs(offset_path(geometry, fitted)[2]).max()), 1e-9)
        ),
        "quasi_static_lap_s": {
            "centerline": float(
                profile_lap_time(geometry, np.zeros(geometry.num_samples), **evaluation)
            ),
            "solved_line": float(profile_lap_time(geometry, line, **evaluation)),
            "fitted_line": float(profile_lap_time(geometry, fitted, **evaluation)),
        },
        "gains": {"k_e": k_e, "k_d": k_d},
        "steer_lag_s": steer_lag_s,
        "design_frequency_rad_s": frequency,
        "lookahead_s": float(params.k_v),
    }
    return params, report


def format_report(report: dict) -> str:
    lap = report["quasi_static_lap_s"]
    lines = [
        report["car"],
        "",
        f"  centerline tightest radius   {report['centerline_min_radius_m']:.3f} m",
        f"  widest the corridor allows   {report['widest_achievable_radius_m']:.3f} m",
        f"  car's minimum turn radius    {report['r_min_m']:.3f} m "
        f"(kinematics alone say {report['geometric_r_min_m']:.3f} m)",
        f"  wheels reach                 "
        f"{report['steer_ratio_at_speed']:.0%} of the commanded angle at speed",
        f"  steering dead time           {report['steer_lag_s'] * 1000:.0f} ms, so the "
        f"cross-track loop is designed at\n                               "
        f"{report['design_frequency_rad_s']:.1f} rad/s and looks "
        f"{report['lookahead_s'] * 1000:.0f} ms ahead",
        "",
    ]
    if report["track_is_steerable"]:
        lines.append(
            "  STEERABLE: the corridor can hold the car's minimum radius, so a "
            "kinematic\n             line exists. Pure pursuit has a chance."
        )
    else:
        lines.append(
            "  NOT STEERABLE BY GEOMETRY ALONE: the tightest corner needs a radius\n"
            "             smaller than the car can turn. The teacher will have to "
            "rotate the\n             car with the throttle, which pure pursuit "
            "cannot discover — expect\n             Gate 1 to fail and read this "
            "before blaming the gains."
        )
    lines += [
        "",
        f"  racing line: R_min {report['line']['peak_radius_m']:.3f} m, "
        f"path {report['line'].get('path_length_m', float('nan')):.2f} m, "
        f"peak offset {report['peak_offset_m']:.3f} m",
        f"  as {LINE_POINTS} control points: "
        f"R_min {report['fitted_peak_radius_m']:.3f} m, "
        f"{report['line_fit_error_mm']:.1f} mm from the solved line",
        "",
        "  quasi-static lap time (a point-mass estimate, not a prediction):",
        f"    centerline   {lap['centerline']:6.2f} s",
        f"    solved line  {lap['solved_line']:6.2f} s",
        f"    fitted line  {lap['fitted_line']:6.2f} s"
        "   <- what the controller starts from",
    ]
    if not report["car_measured"]:
        lines += [
            "",
            "  WARNING: these are the built-in guesses, not measurements. Run",
            "           `python -m tools.probe` on a GPU box before trusting any",
            "           of the numbers above.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m teacher.warmstart",
        description="Build the controller's starting parameters. Needs no simulator.",
    )
    parser.add_argument("--dynamics", default=None, help="Probe output to read.")
    parser.add_argument("--out", default=str(DEFAULT_PATH), help="Where to write.")
    parser.add_argument(
        "--spacing", type=float, default=0.02, help="Track resample spacing, metres."
    )
    parser.add_argument(
        "--report", action="store_true", help="Also write <out>.report.json."
    )
    args = parser.parse_args(argv)

    car = Measured.load(args.dynamics) if args.dynamics else Measured.load()
    geometry = TrackGeometry.from_track(
        Track(OFFICIAL, device="cpu"), spacing_m=args.spacing
    )
    print(geometry.describe())
    for problem in geometry.validate():
        print(f"[geometry] warning: {problem}")

    params, report = build(geometry, car)
    print()
    print(format_report(report))

    path = params.save(args.out)
    print(f"\n[warmstart] parameters written to {path}")
    if args.report:
        report_path = Path(str(args.out) + ".report.json")
        report_path.write_text(json.dumps(report, indent=2, default=float))
        print(f"[warmstart] report written to {report_path}")

    # A line that is not steerable is not a failure of this script, but it is the
    # single most important thing to know before Gate 1.
    return 0 if report["track_is_steerable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
