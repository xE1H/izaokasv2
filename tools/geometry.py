"""Track geometry: uniform arc length, a periodic smoothing spline, and projection.

The SDK's :class:`~lituanicax_sdk.track.Track` is enough to drive against, but not
enough to *optimize* against. Two of its properties get in the way:

* its centerline points are **not uniformly spaced** — on the official track they
  run from 17 mm to 115 mm apart, mean 41 mm;
* its curvature is Menger curvature on those raw points, which on uneven spacing
  is dominated by spacing noise.

Both matter because every downstream quantity — the racing line, the speed
profile, the lookahead the controller steers at — is a function of curvature.
Measured on the official track: Menger curvature on the raw points peaks at
2.95 1/m, an interpolating spline peaks at 4.13 1/m (ringing off the uneven
spacing), and a *smoothing* spline that stays within 2.2 mm of the raw points
peaks at 2.4-2.5 1/m. The last one is the track; the other two are artefacts.

So this resamples to uniform arc length, fits a periodic smoothing spline, and
takes tangent and curvature from the spline's own derivatives rather than from
differences of samples.

Construction is scipy on the CPU and happens once. Everything on the hot path
(:meth:`TrackGeometry.project`, :meth:`TrackGeometry.at`) is torch tensor work
over the environment dimension, because it runs for thousands of cars every
step.

No Isaac Sim dependency, so it is tested directly — see ``tests/test_geometry.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import splev, splprep

#: Spacing of the resampled centerline. Projection recovers arc length to
#: second order in the distance to the nearest sample, so halving this quarters
#: the round-trip error; 20 mm keeps it well under a millimetre on the official
#: track's tightest corner.
DEFAULT_SPACING_M = 0.02

#: How far the smoothing spline may sit from the raw centerline points, RMS, in
#: millimetres. Small enough that the spline is still *this* track; large enough
#: to absorb the export noise that makes an interpolating spline ring. 0.7 mm
#: corresponds to scipy's ``s`` of about 6e-4 on the official track's 1216
#: points, and measures 2.2 mm at its worst point.
DEFAULT_SMOOTH_RMS_MM = 0.7


class TrackGeometry:
    """A track resampled to uniform arc length, with spline tangent and curvature.

    All arrays are torch tensors on ``device``, indexed by sample, and every
    sample is ``spacing_m`` of arc length from its neighbours. Arc length ``s``
    runs from 0 to :attr:`length` in the order the centerline is stored, which
    is the direction :class:`~lituanicax_sdk.spawn.SpawnManager` faces a car
    when no heading is given — and therefore the direction a scored attempt
    starts in.

    Attributes:
        s: ``[M]`` arc length of each sample, metres.
        pos: ``[M, 2]`` position.
        tangent: ``[M, 2]`` unit tangent, in the stored direction.
        normal: ``[M, 2]`` unit left normal, i.e. ``(-ty, tx)``.
        kappa: ``[M]`` **signed** curvature, 1/m. Positive turns left.
        length: total arc length, metres.
        spacing_m: arc length between samples.
    """

    def __init__(
        self,
        points: np.ndarray | torch.Tensor,
        *,
        spacing_m: float = DEFAULT_SPACING_M,
        smooth_rms_mm: float = DEFAULT_SMOOTH_RMS_MM,
        device: torch.device | str = "cpu",
    ):
        raw = np.asarray(
            points.detach().cpu().numpy()
            if isinstance(points, torch.Tensor)
            else points,
            dtype=np.float64,
        )
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError(f"points must be [P, 2]; got {raw.shape}.")
        if len(raw) < 8:
            raise ValueError(f"a closed loop needs at least 8 points; got {len(raw)}.")

        self.device = torch.device(device)
        self.smooth_rms_mm = float(smooth_rms_mm)

        tck, chord_length = self._fit(raw, smooth_rms_mm)
        self._tck = tck
        pos, tangent, kappa, length = self._resample(tck, chord_length, spacing_m)

        self.length = float(length)
        self.num_samples = len(pos)
        #: The spacing actually used, which is ``length / num_samples`` and not
        #: quite what was asked for — the sample count has to be a whole number
        #: for the loop to close on itself. Everything that converts an arc
        #: length into a sample index divides by *this*; dividing by the
        #: requested value instead drifts by up to half a sample over a lap,
        #: which lands as a discontinuity at the start/finish seam.
        self.spacing_m = self.length / self.num_samples
        self.s = torch.arange(self.num_samples, dtype=torch.float32, device=self.device)
        self.s *= self.spacing_m
        self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        self.tangent = torch.tensor(tangent, dtype=torch.float32, device=self.device)
        self.normal = torch.stack([-self.tangent[:, 1], self.tangent[:, 0]], dim=-1)
        self.kappa = torch.tensor(kappa, dtype=torch.float32, device=self.device)

        # Precomputed for project(): ||pos||^2, so the distance expansion below
        # never materialises an [N, M, 2] tensor.
        self._pos_sq = (self.pos * self.pos).sum(dim=-1)

    # ──────────────────────────────────────────────────────────────────────
    #  Construction
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_track(cls, track, **kwargs) -> "TrackGeometry":
        """Build from a loaded :class:`~lituanicax_sdk.track.Track`.

        Takes the track's already-scaled centerline points, so whatever
        ``centerline_scale`` the track config applies is respected.
        """
        return cls(track.points, device=track.device, **kwargs)

    @staticmethod
    def _fit(raw: np.ndarray, smooth_rms_mm: float):
        """Fit a periodic smoothing spline, parameterised by normalised chord length.

        scipy's ``s`` is a total squared residual in metres squared, which scales
        with how many points a track happens to have. Specifying an RMS
        deviation instead makes the setting mean the same thing on any layout.
        """
        # A closed loop: repeat the first point so splprep's periodic fit has a
        # full period to work with.
        closed = np.vstack([raw, raw[:1]])
        steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        chord_length = float(steps.sum())
        u = np.concatenate([[0.0], np.cumsum(steps)]) / chord_length

        residual = len(raw) * (smooth_rms_mm / 1000.0) ** 2
        tck, _ = splprep([closed[:, 0], closed[:, 1]], u=u, s=residual, per=1)
        return tck, chord_length

    @staticmethod
    def _resample(tck, chord_length: float, spacing_m: float):
        """Resample the spline at uniform arc length.

        The spline is parameterised by normalised chord length, which is not arc
        length, so this integrates ``|dP/du|`` on a dense grid and inverts it.
        """
        dense = np.linspace(0.0, 1.0, max(20000, int(40 * chord_length / spacing_m)))
        dx, dy = splev(dense, tck, der=1)
        speed = np.hypot(dx, dy)
        # Cumulative arc length by trapezoid — the integrand is smooth, so this
        # is far more accurate than summing chords.
        arc = np.concatenate(
            [[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(dense))]
        )
        length = float(arc[-1])

        count = max(8, int(round(length / spacing_m)))
        spacing = length / count  # exact, so the loop closes on itself
        wanted = np.arange(count) * spacing
        u = np.interp(wanted, arc, dense)

        x, y = splev(u, tck)
        dx, dy = splev(u, tck, der=1)
        ddx, ddy = splev(u, tck, der=2)

        speed = np.hypot(dx, dy)
        tangent = np.stack([dx / speed, dy / speed], axis=-1)
        kappa = (dx * ddy - dy * ddx) / np.maximum(speed**3, 1e-12)
        return np.stack([x, y], axis=-1), tangent, kappa, length

    # ──────────────────────────────────────────────────────────────────────
    #  Queries — the hot path
    # ──────────────────────────────────────────────────────────────────────

    def project(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest point on the centerline, as arc length and signed offset.

        Args:
            xy: ``[N, 2]`` world positions.

        Returns:
            ``(s, n)``, each ``[N]`` — arc length in ``[0, length)`` and lateral
            offset in metres, **positive to the left** of the direction of
            travel, matching :attr:`normal`.

        A coarse nearest-sample search followed by a first-order refinement
        along the local tangent. Deliberately searched against this class's own
        uniform samples rather than through
        :meth:`~lituanicax_sdk.track.Track.nearest`: the SDK's points are up to
        115 mm apart, and a coarse estimate that bad costs more accuracy in the
        refinement than the shared code saves.
        """
        xy = xy.to(self.pos.dtype)
        # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, so the [N, M, 2] difference
        # is never built. The ||a||^2 term is constant per row and drops out of
        # the argmin.
        d2 = self._pos_sq.unsqueeze(0) - 2.0 * (xy @ self.pos.T)
        index = d2.argmin(dim=1)

        offset = xy - self.pos[index]
        along = (offset * self.tangent[index]).sum(dim=-1)
        n = (offset * self.normal[index]).sum(dim=-1)
        s = torch.remainder(self.s[index] + along, self.length)
        return s, n

    def at(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Position, unit tangent and signed curvature at arc length ``s``.

        Args:
            s: any shape; wrapped into ``[0, length)`` so lookahead past the
                start/finish line needs no special case.

        Returns:
            ``(pos, tangent, kappa)`` of shapes ``s.shape + (2,)``,
            ``s.shape + (2,)`` and ``s.shape``.

        Linear interpolation between samples. At 20 mm spacing the interpolation
        error on the official track's tightest corner is about 25 microns, which
        is four orders of magnitude below the 0.15 m crash radius.
        """
        pos = self.interp_vec(s, self.pos)
        tangent = self.interp_vec(s, self.tangent)
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return pos, tangent, self.interp(s, self.kappa)

    def kappa_at(self, s: torch.Tensor) -> torch.Tensor:
        """Signed curvature at arc length ``s``, 1/m."""
        return self.interp(s, self.kappa)

    def heading_at(self, s: torch.Tensor) -> torch.Tensor:
        """Track heading at arc length ``s``, radians in ``[-pi, pi]``."""
        tangent = self.interp_vec(s, self.tangent)
        return torch.atan2(tangent[..., 1], tangent[..., 0])

    def point_at(self, s: torch.Tensor, n: torch.Tensor | float = 0.0) -> torch.Tensor:
        """World position at arc length ``s``, offset ``n`` metres to the left."""
        pos = self.interp_vec(s, self.pos)
        if isinstance(n, (int, float)):
            if n == 0.0:
                return pos
            n = torch.full_like(s, float(n))
        return pos + n.unsqueeze(-1) * self.interp_vec(s, self.normal)

    # ── Interpolation helpers ─────────────────────────────────────────────

    def _weights(self, s: torch.Tensor):
        position = torch.remainder(s, self.length) / self.spacing_m
        low = torch.floor(position).long() % self.num_samples
        high = (low + 1) % self.num_samples
        return low, high, (position - torch.floor(position)).unsqueeze(-1)

    def interp(
        self, s: torch.Tensor, table: torch.Tensor, rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Interpolate a ``[M]`` table, or one row each of a ``[C, M]`` table.

        ``rows`` is what makes a whole CMA-ES population affordable. Each car
        follows a different candidate's reference, so the naive approach is to
        gather ``table[rows]`` into ``[N, M]`` and index that — 6.4 million floats
        per table per step at 2560 cars. Indexing with both ``rows`` and the
        sample index at once returns ``[N]`` directly and never builds it.
        """
        low, high, frac = self._weights(s)
        frac = frac.squeeze(-1)
        if rows is None:
            return table[low] * (1.0 - frac) + table[high] * frac
        return table[rows, low] * (1.0 - frac) + table[rows, high] * frac

    def interp_vec(self, s: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        low, high, frac = self._weights(s)
        return table[low] * (1.0 - frac) + table[high] * frac

    # ──────────────────────────────────────────────────────────────────────
    #  Checking
    # ──────────────────────────────────────────────────────────────────────

    def total_turning(self) -> float:
        """``∮κ ds``, which is ``±2π`` for any simple closed loop.

        A value that is not close to a whole turn means the centerline crosses
        itself or the spline has gone somewhere the track does not.
        """
        return float((self.kappa * self.spacing_m).sum())

    def closure_error(self) -> float:
        """How far reintegrating the tangents from ``κ`` misses the start, metres.

        Curvature is the whole of what the racing line and the speed profile are
        built from, so a loop that does not close under its own curvature is a
        loop whose curvature is wrong.
        """
        heading = torch.cumsum(self.kappa * self.spacing_m, dim=0)
        step = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        return float((step * self.spacing_m).sum(dim=0).norm())

    def describe(self) -> str:
        kappa = self.kappa.abs()
        peak = float(kappa.max())
        return (
            f"[geometry] {self.num_samples} samples at {self.spacing_m * 1000:.0f} mm, "
            f"{self.length:.3f} m round, |k|max {peak:.3f} 1/m "
            f"(R {1.0 / max(peak, 1e-9):.3f} m), "
            f"turning {self.total_turning() / (2 * np.pi):+.4f} turns"
        )

    def validate(self) -> list[str]:
        """Sanity-check the resampled geometry; empty list is good."""
        problems = []

        turning = self.total_turning() / (2 * np.pi)
        if abs(abs(turning) - 1.0) > 0.02:
            problems.append(
                f"total turning is {turning:+.4f} turns, not +/-1. The centerline "
                "is not a simple closed loop, or the spline has left the track."
            )

        closure = self.closure_error()
        if closure > 0.1:
            problems.append(
                f"reintegrating the tangents from curvature misses the start by "
                f"{closure:.3f} m; the curvature is not self-consistent."
            )

        spacing = (self.pos - torch.roll(self.pos, 1, dims=0)).norm(dim=-1)
        drift = float((spacing - self.spacing_m).abs().max())
        if drift > 0.1 * self.spacing_m:
            problems.append(
                f"resampled spacing varies by {drift * 1000:.1f} mm against a "
                f"target of {self.spacing_m * 1000:.0f} mm; the arc-length "
                "inversion is not converged."
            )
        return problems
