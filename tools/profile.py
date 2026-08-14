"""Racing lines and speed profiles — the quasi-static model the teacher starts from.

Two things live here, both pure numpy on the CPU, both computed once per
candidate rather than once per simulation step:

* :func:`three_pass_profile` — the minimum-time speed profile on a *fixed* path,
  the standard forward/backward sweep against a lateral-acceleration limit.
* :func:`racing_line_offsets` — the racing line, as a lateral offset from the
  centerline, by Adam on the exact gradient of a cornering-limited lap time.

**The line objective, and two that do not work.** On a track this tight the
corridor is a large fraction of the corner radius — ±0.18 m against roughly
0.4 m — which breaks the assumption the textbook racing-line formulation rests
on. Measured on the official track at a_lat = 8, a_accel = 6:

==============================  ========  =======  =====================
line                            path      R_min    accel-limited lap
==============================  ========  =======  =====================
centerline                       50.00 m   0.366           14.45 s
min-Σκ², linearized curvature    46.30 m   0.281           13.49 s
min-Σκ², true curvature          53.34 m   0.413           14.64 s
min-time, bounded R ≥ 0.41       46.86 m   0.409           13.88 s
min-time, unbounded              45.77 m   0.249           13.78 s
==============================  ========  =======  =====================

The linearized min-Σκ² line is the fastest-looking and is not real: at its own
optimum the linearization reports a peak curvature of 1.26 1/m where the path
actually reaches 3.55, so it claims no bound violations against 118 real ones,
and its 0.281 m apex is far inside anything the car can steer. Minimizing Σκ² on
*true* curvature is honest and lengthens the path to 53.3 m, which makes it
slower than leaving the line alone. Neither is a lap-time objective; only the
last two are, and the bound is what makes one of them drivable.

**Why the curvature bound.** ``R_min = L_wb / tan(0.488)`` — 0.41 m at a 0.22 m
wheelbase, 0.57 m at 0.30 m. The unbounded line's 0.249 m apex is under all of
them, so the fastest line on paper is one the car physically cannot steer. Pass
``kappa_max = 1 / R_min`` from the Phase 0 probe.
"""

from __future__ import annotations

import numpy as np
import torch

#: Half the width the car's *centre* may use. The track is 0.70 m wide, so the
#: centerline sits 0.351 m from either wall, and the crash rule fires when the
#: centre comes within :data:`~lituanicax_sdk.rules.WALL_COLLISION_RADIUS_M`
#: (0.15 m) of one — leaving 0.201 m. 0.18 m keeps 2 cm of working room for the
#: controller to be imperfect in.
DEFAULT_HALF_WIDTH_M = 0.18


# ═══════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════


def offset_path(geometry, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The path ``centerline + n * normal``, and its true geometry.

    Args:
        geometry: a :class:`~tools.geometry.TrackGeometry`.
        n: ``[M]`` or ``[B, M]`` lateral offsets in metres, one per centerline
            sample, positive to the left.

    Returns:
        ``(pos, seg, kappa)`` — positions ``[..., M, 2]``, the length of the
        segment leaving each point ``[..., M]``, and **signed** curvature
        ``[..., M]``.

    Curvature is measured on the resulting polyline rather than carried over from
    the centerline, because moving the line changes it — that is the entire
    point of moving it. Menger curvature is fine here: the samples are uniform by
    construction, which is the condition the raw centerline failed.
    """
    n = np.asarray(n, dtype=np.float64)
    pos = geometry.pos.cpu().numpy() + n[..., None] * geometry.normal.cpu().numpy()
    return (pos, *_polyline_geometry(pos))


def _polyline_geometry(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment lengths and signed Menger curvature of a closed polyline."""
    ahead = np.roll(pos, -1, axis=-2)
    behind = np.roll(pos, 1, axis=-2)
    seg = np.linalg.norm(ahead - pos, axis=-1)

    to_here = pos - behind
    to_next = ahead - pos
    across = ahead - behind
    cross = to_here[..., 0] * to_next[..., 1] - to_here[..., 1] * to_next[..., 0]
    denominator = (
        np.linalg.norm(to_here, axis=-1)
        * np.linalg.norm(to_next, axis=-1)
        * np.linalg.norm(across, axis=-1)
    )
    return seg, 2.0 * cross / np.maximum(denominator, 1e-12)


# ═══════════════════════════════════════════════════════════════════════════
#  Speed profile
# ═══════════════════════════════════════════════════════════════════════════


#: Fraction of its standing-start acceleration the car still has at top speed.
#: Not zero, because a profile that plans on exactly zero acceleration at ``v_max``
#: can never quite reach it and the forward sweep crawls the last fraction.
FALLOFF_FLOOR = 0.05


def _falloff(speed: np.ndarray, v_max: float) -> np.ndarray:
    """How much of its standing acceleration the car still has at ``speed``.

    A DC motor's torque falls linearly with speed (``dynamics.py:44``), so the
    acceleration it can produce does too. The probe measured exactly that on this
    car: 4.9 m/s² near rest, 0.29 m/s² at 6.75 m/s, against a top speed of 6.92.

    Modelling it as one constant — which is what a single ``a_accel`` is — gets
    the car wrong at both ends. At 2.74 m/s², the average, the profile
    under-drives it out of every slow corner where it really has 4.9, and plans
    to reach speeds on the straights that it really cannot. On a car this
    power-limited that is the largest error in the whole quasi-static model, and
    correcting it costs no parameters: the shape is fixed and ``a_accel`` simply
    becomes the standing-start figure rather than an average.
    """
    return np.clip(1.0 - speed / max(v_max, 1e-9), FALLOFF_FLOOR, 1.0)


#: Slowest the steering ceiling may order the car to go, m/s.
#:
#: Comfortably above the stall rule's threshold — 4% of top speed, 0.28 m/s on
#: this car — because a reference that asks for less than that is asking the car
#: to be disqualified. 1.5 m/s is also roughly what the tightest corner here is
#: taken at, so the floor binds rarely and never fatally.
STEERING_FLOOR_M_S = 1.5


def steering_ceiling(
    kappa: np.ndarray,
    *,
    wheelbase_m: float,
    max_steer_rad: float,
    steer_ratio: float,
    v_max: float,
) -> np.ndarray:
    """Fastest speed at which the car can still *steer* each curvature, m/s.

    The cornering ceiling asks whether the tyres can hold the corner. This asks
    the question that turned out to bind harder on this car: whether the wheels
    can be pointed round it at all.

    They often cannot. The steering is an effort-limited servo fighting the
    tyres, and what it achieves falls with speed — measured at 78% of the
    commanded angle at 2.4 m/s and 62% at 6.8. So a reference that specifies a
    radius without specifying a speed at which that radius is reachable is
    specifying something the car cannot do: the diagnostic found the best
    candidate saturated for 53% of a lap, asking for R = 0.386 m while the wheels
    could deliver 0.56 m.

    Modelling the loss as linear in speed — 1.0 at rest falling to
    ``steer_ratio`` at ``v_max`` — the achievable angle is
    ``ratio(v) · max_steer``, the corner needs ``atan(L·κ)``, and the largest
    speed that satisfies it falls out in closed form.
    """
    needed = np.arctan(wheelbase_m * np.abs(kappa)) / max(max_steer_rad, 1e-9)
    loss = max(1.0 - float(steer_ratio), 1e-6)
    # ratio(v) = 1 - loss·v/v_max  >=  needed
    ceiling = v_max * (1.0 - needed) / loss
    # Where the corner needs more than full lock the expression goes negative,
    # and the floor is doing real work: this track's centerline asks for 1.12 of
    # full lock at its tightest, so *every* profile hits it. A floor of 0.1 m/s
    # is below the stall rule's own threshold (4% of top speed, 0.28 m/s here),
    # so the reference would have been ordering the car to fail. And a corner
    # that kinematics cannot steer is not a corner the car cannot take — it can
    # slide through it, which is why steer_ratio_eff is searched rather than
    # believed.
    return np.clip(ceiling, STEERING_FLOOR_M_S, v_max)


def three_pass_profile(
    seg: np.ndarray,
    kappa: np.ndarray,
    *,
    a_lat: float,
    a_accel: float,
    a_brake: float,
    v_max: float,
    steer_ceiling: np.ndarray | None = None,
    iterations: int = 6,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Minimum-time speed profile on a fixed path.

    Args:
        seg: ``[..., M]`` length of the segment leaving each point, metres.
        kappa: ``[..., M]`` curvature at each point, 1/m. Sign is ignored.
        a_lat: cornering limit, m/s².
        a_accel: forward acceleration **from a standing start**, m/s². What the
            car has at ``speed`` is this times :func:`_falloff` — the motor's
            torque falls off linearly and so does the acceleration.
        a_brake: braking limit, m/s² (positive).
        v_max: top speed, m/s. Also sets where the acceleration falls to nothing.
        steer_ceiling: optional ``[..., M]`` per-point speed cap from
            :func:`steering_ceiling`. Without it the profile will happily specify
            a radius the car cannot point its wheels round at the speed asked.
        iterations: how many times to sweep forward and back. The passes wrap
            around the loop, so one sweep is not enough — a braking zone that
            starts before the start/finish line has to propagate across it.
        tolerance: stop early once a sweep changes no speed by more than this.

    Returns:
        ``[..., M]`` speed, m/s.

    Three passes, the textbook construction: a cornering ceiling, then forward
    propagation of what the engine can add, then backward propagation of what the
    brakes must take away.

    Batched over any leading dimensions, and the sequential sweeps loop over
    ``M`` only — so scoring a whole CMA-ES population costs the same number of
    Python iterations as scoring one candidate.
    """
    seg = np.asarray(seg, dtype=np.float64)
    kappa = np.abs(np.asarray(kappa, dtype=np.float64))
    if seg.shape != kappa.shape:
        raise ValueError(f"seg {seg.shape} and kappa {kappa.shape} must match.")

    count = seg.shape[-1]
    # 1. The ceilings: what the tyres allow, and what the steering allows.
    speed = np.minimum(v_max, np.sqrt(a_lat / np.maximum(kappa, 1e-9)))
    if steer_ceiling is not None:
        speed = np.minimum(speed, steer_ceiling)

    for _ in range(iterations):
        before = speed.copy()
        # 2. Forward: bounded by what the car can accelerate to.
        for i in range(count):
            j = (i + 1) % count
            here = a_accel * _falloff(speed[..., i], v_max)
            reachable = np.sqrt(speed[..., i] ** 2 + 2.0 * here * seg[..., i])
            speed[..., j] = np.minimum(speed[..., j], reachable)
        # 3. Backward: bounded by what it can still brake from.
        for i in range(count - 1, -1, -1):
            j = (i + 1) % count
            stoppable = np.sqrt(speed[..., j] ** 2 + 2.0 * a_brake * seg[..., i])
            speed[..., i] = np.minimum(speed[..., i], stoppable)
        if np.max(np.abs(speed - before)) < tolerance:
            break

    return speed


def lap_time(seg: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Time to cover a path at a given speed profile, seconds.

    Trapezoidal in ``1/v`` rather than ``seg/v``: on a track where speed halves
    through a hairpin, taking the speed at the segment's start overstates the
    time by a few percent.
    """
    ahead = np.roll(speed, -1, axis=-1)
    mean_inverse = 0.5 * (1.0 / speed + 1.0 / ahead)
    return np.sum(seg * mean_inverse, axis=-1)


def profile_lap_time(
    geometry,
    n: np.ndarray,
    *,
    a_lat: float,
    a_accel: float,
    a_brake: float,
    v_max: float,
) -> np.ndarray:
    """Quasi-static lap time of the line ``n``. Convenience for the two above."""
    _, seg, kappa = offset_path(geometry, n)
    speed = three_pass_profile(
        seg, kappa, a_lat=a_lat, a_accel=a_accel, a_brake=a_brake, v_max=v_max
    )
    return lap_time(seg, speed)


# ═══════════════════════════════════════════════════════════════════════════
#  The racing line
# ═══════════════════════════════════════════════════════════════════════════


#: Control points the racing line is solved in. One per ~0.6 m on a 50 m track.
#: The line is solved in a smooth basis rather than per-sample for two reasons:
#: a few hundred free offsets at 20 mm spacing can encode millimetre ripple that
#: the speed profile reads as a series of hairpins, and a normalised non-negative
#: basis makes the corridor a plain box constraint (see :func:`periodic_basis`).
DEFAULT_CONTROL_POINTS = 80


def _torch_curvature(pos: torch.Tensor) -> torch.Tensor:
    """Signed Menger curvature of a closed polyline, differentiably.

    The same formula as :func:`_polyline_geometry`, in torch so that autograd can
    supply exact gradients of the *true* curvature with respect to the offsets.

    That matters more than it sounds. The textbook racing-line QP linearizes
    curvature as ``κ_c + n''``, which is valid when the corridor is small against
    the corner radius. Here it is 0.18 m against 0.4 m, and at the optimum the
    linearization reports a peak of 1.26 1/m where the real path reaches 3.55 —
    it claims zero bound violations against 118 real ones. So the bound is
    enforced against the real thing.
    """
    ahead = torch.roll(pos, -1, dims=-2)
    behind = torch.roll(pos, 1, dims=-2)
    to_here = pos - behind
    to_next = ahead - pos
    across = ahead - behind
    cross = to_here[..., 0] * to_next[..., 1] - to_here[..., 1] * to_next[..., 0]
    denominator = (
        to_here.norm(dim=-1) * to_next.norm(dim=-1) * across.norm(dim=-1)
    ).clamp(min=1e-12)
    return 2.0 * cross / denominator


def periodic_basis(count: int, control_points: int) -> np.ndarray:
    """Periodic smooth basis, ``[M, K]``, rows non-negative and summing to one.

    That last property is what makes the corridor cheap: ``n = B w`` is then a
    weighted average of ``w``, so ``|w_j| <= half_width`` for every ``j`` implies
    ``|n_i| <= half_width`` everywhere. The corridor becomes a box constraint on
    the coefficients and the whole solve stays L-BFGS-B rather than needing a
    general QP solver.
    """
    index = np.arange(count)
    centres = np.linspace(0.0, count, control_points, endpoint=False)
    width = count / control_points * 0.75
    # Periodic distance, so the basis wraps at the start/finish line.
    distance = np.abs(index[:, None] - centres[None, :])
    distance = np.minimum(distance, count - distance)
    basis = np.exp(-0.5 * (distance / width) ** 2)
    return basis / basis.sum(axis=1, keepdims=True)


def widest_achievable_radius(
    geometry,
    *,
    half_width: float = DEFAULT_HALF_WIDTH_M,
    control_points: int = DEFAULT_CONTROL_POINTS,
    iterations: int = 1500,
    sharpness: float = 40.0,
) -> tuple[float, np.ndarray]:
    """The largest minimum radius any line in the corridor can achieve, metres.

    Returns ``(radius, n)`` — the radius and the line that achieves it.

    **This is the number that decides whether the car can drive the track at
    all.** A car steering kinematically cannot corner tighter than
    ``R_min = L_wb / tan(max_steer)``, so if ``R_min`` exceeds this, the tightest
    corner is not merely slow — it is impossible without rotating the car with
    the throttle or running closer to the wall than the corridor allows.

    On the official track this measures about 0.50-0.55 m against a centerline
    minimum of 0.366 m, so the corridor buys roughly 0.15 m of radius. Compare it
    against ``R_min`` from the Phase 0 probe:

    * ``L_wb <= 0.26 m`` gives ``R_min <= 0.49 m`` — drivable by pure steering.
    * ``L_wb >= 0.28 m`` gives ``R_min >= 0.53 m`` — not drivable by pure
      steering, and a pure-pursuit teacher will fail Gate 1.

    Maximizes the minimum radius by minimizing a soft maximum of ``|κ|``, which is
    one gradient solve rather than a binary search over
    :func:`racing_line_offsets`.
    """
    coefficients = widest_line_coefficients(
        geometry,
        half_width=half_width,
        control_points=control_points,
        iterations=iterations,
        sharpness=sharpness,
    )
    basis = torch.tensor(
        periodic_basis(geometry.num_samples, control_points), dtype=torch.float64
    )
    n = (basis @ coefficients).detach().numpy()
    _, _, kappa = offset_path(geometry, n)
    return 1.0 / max(float(np.abs(kappa).max()), 1e-9), n


def widest_line_coefficients(
    geometry,
    *,
    half_width: float,
    control_points: int,
    iterations: int = 1500,
    sharpness: float = 40.0,
) -> torch.Tensor:
    """The basis coefficients of the flattest line the corridor allows.

    Public because it is wanted three times over — as the answer to "how much
    radius can this corridor buy", as one of the two starting points for the
    min-time solve, and as the fallback when no line meets the bound — and it is
    the most expensive solve in this module. Callers that need more than one of
    those should compute it once and pass it along.
    """
    basis = torch.tensor(
        periodic_basis(geometry.num_samples, control_points), dtype=torch.float64
    )
    centerline = geometry.pos.detach().cpu().double()
    normal = geometry.normal.detach().cpu().double()
    coefficients = torch.zeros(control_points, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([coefficients], lr=0.02)
    for _ in range(iterations):
        optimizer.zero_grad()
        path = centerline + (basis @ coefficients).unsqueeze(-1) * normal
        kappa = _torch_curvature(path).abs()
        # Soft maximum: a hard max would give gradient to one sample only, and
        # the apex would chase itself around the corner.
        loss = torch.logsumexp(sharpness * kappa, dim=0) / sharpness
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            coefficients.clamp_(-half_width, half_width)
    return coefficients.detach()


def racing_line_offsets(
    geometry,
    *,
    a_lat: float,
    v_max: float,
    half_width: float = DEFAULT_HALF_WIDTH_M,
    kappa_max: float | None = None,
    control_points: int = DEFAULT_CONTROL_POINTS,
    iterations: int = 800,
    learning_rate: float = 0.02,
    flattest: torch.Tensor | None = None,
) -> tuple[np.ndarray, dict]:
    """The fastest line in the corridor that the car can actually steer.

    Args:
        geometry: a :class:`~tools.geometry.TrackGeometry`.
        a_lat: cornering limit, m/s², from the Phase 0 probe.
        v_max: top speed, m/s.
        half_width: corridor the offset may use, metres either side.
        kappa_max: hard cap on the path's curvature, 1/m — pass ``1 / R_min``,
            the car's geometric minimum turn radius. ``None`` leaves it
            unbounded, which produces a faster line the car cannot follow.
        control_points: coefficients solved for. Fewer means a smoother line.
        iterations: Adam steps per penalty stage.
        flattest: the coefficients of the flattest line this corridor allows, if
            the caller already has them. One of the two starting points, and by
            far the most expensive thing here — a caller that has just run
            :func:`widest_achievable_radius` at the same width should pass them
            rather than pay for the solve twice.

    Returns:
        ``(n, report)`` — offsets ``[M]`` in metres, positive left, and a dict
        recording peak curvature, path length, and whether the bound holds.

    Minimizes the **cornering-limited lap time** ``Σ ds / min(v_max, √(a_lat/|κ|))``
    over a smooth periodic basis, by Adam on the exact autograd gradient. Both
    quantities that decide lap time are in that expression — how far the path is
    and how tight it is — so the optimizer trades them against each other
    correctly instead of proxying one of them.

    Accelerating and braking are deliberately left out of *this* objective. They
    matter (they cost about 1.5 s on the official track) but including them means
    differentiating the sequential sweeps of :func:`three_pass_profile`, and the
    resulting line differs little. The warm start does not need to be optimal —
    CMA-ES optimizes real lap time in the real simulator afterwards. It needs to
    be feasible, smooth, and better than the centerline.

    **Two objectives that look reasonable and are not.** Minimizing Σκ² on the
    *linearized* curvature — the textbook racing-line QP — produces the fastest
    line here on paper, but the linearization is invalid at this corridor width
    (see :func:`_torch_curvature`) and its apex comes out at R = 0.25 m, well
    inside anything the car can steer. Minimizing Σκ² on the *true* curvature is
    honest and lengthens the path from 50.0 m to 53.3 m, which makes it slower
    than doing nothing. Measured on the official track at a_lat = 8, a_accel = 6:

    ==============================  ========  =======  ======
    line                            path      R_min    lap
    ==============================  ========  =======  ======
    centerline                       50.00 m   0.366    14.45
    min-Σκ², linearized              46.30 m   0.281    13.49   (unsteerable)
    min-Σκ², true curvature          53.34 m   0.413    14.64   (slower)
    **this, bounded to R ≥ 0.41**     46.86 m   0.409    13.88
    ==============================  ========  =======  ======
    """
    count = geometry.num_samples
    basis = torch.tensor(periodic_basis(count, control_points), dtype=torch.float64)
    centerline = geometry.pos.detach().cpu().double()
    normal = geometry.normal.detach().cpu().double()
    ceiling = torch.tensor(v_max, dtype=torch.float64)

    report: dict = {"kappa_max_requested": kappa_max, "control_points": control_points}
    best_n: np.ndarray | None = None
    best_time = float("inf")
    requested = kappa_max

    # Two starting points, because neither wins on its own and which one wins
    # depends on how hard the bound bites.
    #
    # From the **centerline**, a penalty continuation works well when the bound
    # is loose: the unpenalized stage finds the aggressive short line and the
    # escalating penalty pulls its apexes back out. But the centerline is the
    # *tightest* thing in the corridor (R = 0.366 m against a corridor that can
    # manage 0.545 m), so a bound near what the car actually needs starts badly
    # violated and the min-time objective pulls the wrong way — tighter corners
    # are a shorter path. At the measured R_min of 0.523 m this found nothing
    # feasible at any weight.
    #
    # From the **flattest line the corridor allows**, the bound holds at step
    # zero and the penalty only has to keep it — which is what rescues the tight
    # case. But with a loose bound it stays out near the walls and returns a
    # 53.6 m path, slower than the centerline.
    #
    # Running both costs a few seconds and the better feasible one wins.
    if flattest is None:
        flattest = widest_line_coefficients(
            geometry, half_width=half_width, control_points=control_points
        )
    zeros = torch.zeros(control_points, dtype=torch.float64)
    attempts = (
        [(zeros, [0.0])]
        if kappa_max is None
        # No unpenalized stage from the flattest start: the stages share one set
        # of coefficients, so it would spend its whole budget undoing the start.
        else [(zeros, [0.0, 1e2, 1e3, 1e4, 1e5, 1e6]), (flattest, [1e2, 1e3, 1e4])]
    )

    for start, stages in attempts:
        coefficients = start.clone().requires_grad_(True)
        # A penalty continuation, then a gentle tightening of the bound itself if
        # the penalty alone cannot close the last fraction of a percent.
        # Escalating the weight is much cheaper in lap time than over-tightening.
        working = kappa_max
        found = False
        spare = 2  # penalty stages still worth running once feasible
        for _tighten in range(3):
            for weight in stages:
                optimizer = torch.optim.Adam([coefficients], lr=learning_rate)
                for _ in range(iterations):
                    optimizer.zero_grad()
                    # No clamp on the offsets: the basis rows are non-negative
                    # and sum to one, so keeping the coefficients in the corridor
                    # keeps the line in it too.
                    offsets = basis @ coefficients
                    path = centerline + offsets.unsqueeze(-1) * normal
                    segment = (torch.roll(path, -1, dims=0) - path).norm(dim=-1)
                    kappa = _torch_curvature(path)
                    speed = torch.minimum(
                        ceiling, (a_lat / kappa.abs().clamp(min=1e-6)).sqrt()
                    )
                    loss = (segment / speed).sum()
                    if weight:
                        excess = (kappa.abs() - working).clamp(min=0.0)
                        loss = loss + weight * (excess**2).mean()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        coefficients.clamp_(-half_width, half_width)

                n = (basis @ coefficients).detach().numpy()
                _, segments, true_kappa = offset_path(geometry, n)
                peak = float(np.abs(true_kappa).max())
                if requested is not None and peak > requested * 1.001:
                    continue  # still not steerable; escalate
                found = True
                elapsed = float(
                    lap_time(
                        segments,
                        np.minimum(
                            v_max,
                            np.sqrt(a_lat / np.maximum(np.abs(true_kappa), 1e-9)),
                        ),
                    )
                )
                if elapsed < best_time:
                    best_time, best_n = elapsed, n
                    report.update(
                        {
                            "peak_kappa": peak,
                            "peak_radius_m": 1.0 / max(peak, 1e-9),
                            "path_length_m": float(segments.sum()),
                            "penalty_weight": weight,
                            "cornering_limited_lap_s": elapsed,
                            "started_from": (
                                "centerline" if start is zeros else "flattest line"
                            ),
                            # The basis coefficients, so a caller searching in
                            # this same basis can take them directly. Refitting
                            # the per-sample line onto a coarser basis afterwards
                            # loses the apexes — 137 mm at worst on the official
                            # track, against a 180 mm corridor.
                            "coefficients": coefficients.detach().numpy().copy(),
                        }
                    )

                # One more stage after the first feasible one, then stop. The
                # escalation exists to *reach* feasibility; past that a heavier
                # penalty only flattens the line further, which is slower, and
                # the best so far is already recorded. Running all five stages
                # regardless was most of the cost of building a warm start.
                spare -= 1 if found else 0
                if spare <= 0:
                    break

            if found or requested is None:
                break
            working = working * 0.97
            report["tightened_to"] = working

    if best_n is None:
        # Nothing satisfied the bound. Fall back to the flattest line the
        # corridor allows, which is the closest to satisfying it that exists —
        # not the centerline, which is the furthest.
        basis_np = periodic_basis(count, control_points)
        best_n = basis_np @ flattest.numpy()
        _, segments, true_kappa = offset_path(geometry, best_n)
        peak = float(np.abs(true_kappa).max())
        report.update(
            {
                "peak_kappa": peak,
                "peak_radius_m": 1.0 / max(peak, 1e-9),
                "path_length_m": float(segments.sum()),
                "coefficients": flattest.numpy().copy(),
                "fell_back_to_widest_line": True,
            }
        )

    report["within_kappa_max"] = (
        requested is None or report["peak_kappa"] <= requested * 1.001
    )
    return best_n, report
