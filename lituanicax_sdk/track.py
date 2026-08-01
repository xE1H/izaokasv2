"""Tracks: the racing line, the walls, and where the start/finish line sits.

Tracks are **open** — build as many as you like and train on your own.  A track
is three things:

* a **centerline CSV**, a closed loop of ``x,y`` points a few centimetres apart
  (any extra columns are ignored), which defines the racing line, the curvature
  and the start/finish line;
* a **walls USD**, whose meshes become the crash boundary;
* an optional **surface USD**, which is scenery — cones, tarmac, kerbs.  It is
  never given collision, so decorating a track cannot change how it drives.

The one locked part is :class:`~lituanicax_sdk.timing.LapTimer`'s use of
``start_finish_index``: a lap is measured from that point on the centerline back
to itself, so lap times mean the same thing on every track and for every team.

Only :meth:`Track.load_walls` needs Isaac Sim; everything else is CSV and
tensor maths, so a track can be checked with :meth:`Track.validate` before you
ever start the simulator.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrackCfg:
    """Everything that describes one track.

    A track is geometry: where the walls are, where the middle is, and where
    the lap is measured from.  It says nothing about where cars start — that is
    a training decision and lives with the team, in
    :class:`~lituanicax_sdk.spawn.SpawnManager` and its subclasses.

    ``centerline_scale`` has to agree with ``mesh_scale``: the walls and the
    centerline describe the same track, so whatever scales one scales the
    other.  An earlier version let a spawn-only fudge factor onto the
    centerline, which scaled the official one 15% too large — it then ran
    through the walls for a quarter of the lap, and every track-relative
    observation was measured against a racing line that was not on the track.
    :meth:`Track.validate` now checks the result against the walls.
    """

    name: str
    walls_usd: str
    centerline_csv: str
    surface_usd: str | None = None

    # Geometry
    mesh_scale: float = 0.85
    centerline_scale: tuple[float, float] = (-0.85, -0.85)

    # Lap timing
    start_finish_index: int = 0  # index into the centerline
    # How near the start/finish point a car must be, measured *along the track*,
    # for a crossing to count. The gate spans the full width of the track, so a
    # wide racing line still trips it.
    lap_gate_window_m: float = 2.0

    # Shape analysis
    corner_curvature_threshold: float = 0.5  # curvature (1/m) that counts as a corner

    def resolve(self, root: Path | str | None = None) -> TrackCfg:
        """Return a copy with relative asset paths resolved against ``root``."""
        if root is None:
            return self
        root = Path(root)

        def fix(p: str | None) -> str | None:
            if p is None:
                return None
            return str(p if Path(p).is_absolute() else root / p)

        return TrackCfg(
            **{
                **self.__dict__,
                "walls_usd": fix(self.walls_usd),
                "centerline_csv": fix(self.centerline_csv),
                "surface_usd": fix(self.surface_usd),
            }
        )


class Track:
    """A loaded track: the centerline and everything precomputed from it.

    Built once at start-up and shared by every car in the scene.  All tensors
    live on ``device`` and are indexed by centerline point.
    """

    def __init__(self, cfg: TrackCfg, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.points = self._load_centerline()  # [P, 2]
        self.num_points = self.points.shape[0]

        # Neighbours, wrapping round because the track is a closed loop.
        previous = torch.roll(self.points, shifts=1, dims=0)
        following = torch.roll(self.points, shifts=-1, dims=0)
        to_here = self.points - previous
        to_next = following - self.points
        across = following - previous
        len_to_here = to_here.norm(dim=1)
        len_to_next = to_next.norm(dim=1)
        len_across = across.norm(dim=1)

        # Menger curvature: the circle through each point and its neighbours.
        cross = to_here[:, 0] * to_next[:, 1] - to_here[:, 1] * to_next[:, 0]
        self.curvature = (
            2.0 * cross.abs() / (len_to_here * len_to_next * len_across).clamp(min=1e-8)
        )
        self.tangents = across / len_across.unsqueeze(-1).clamp(min=1e-8)

        self.arc_length = torch.zeros(self.num_points, device=self.device)
        self.arc_length[1:] = torch.cumsum(len_to_here[1:], dim=0)
        self.track_length = float(self.arc_length[-1] + len_to_here[0])
        self.point_spacing_m = self.track_length / self.num_points

        self.corner_idx = self._find_corners()

        # Start/finish line: a plane through one centerline point, at right
        # angles to the track there.
        gate = cfg.start_finish_index % self.num_points
        self.gate_index = gate
        self.gate_point = self.points[gate]  # [2]
        self.gate_normal = self.tangents[gate]  # [2]
        self.gate_arc = self.arc_length[gate]

        # Filled in by load_walls(), which needs a running simulator.
        self.wall_seg_a: torch.Tensor | None = None
        self.wall_seg_b: torch.Tensor | None = None

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_centerline(self) -> torch.Tensor:
        path = Path(self.cfg.centerline_csv)
        if not path.is_file():
            raise FileNotFoundError(
                f"Track '{self.cfg.name}': centerline file not found: {path}"
            )

        scale_x, scale_y = self.cfg.centerline_scale
        rows = []
        with open(path, newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 2:
                    continue
                try:
                    rows.append([float(row[0]) * scale_x, float(row[1]) * scale_y])
                except ValueError:
                    continue  # a header line, most likely
        if len(rows) < 4:
            raise ValueError(
                f"Track '{self.cfg.name}': centerline {path} has {len(rows)} usable "
                "rows; a closed loop needs at least 4."
            )
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _find_corners(self) -> torch.Tensor:
        """Indices of curvature peaks — the corners the car should anticipate."""
        curvature = self.curvature
        is_peak = (curvature > torch.roll(curvature, 1)) & (
            curvature >= torch.roll(curvature, -1)
        )
        sharp = curvature > self.cfg.corner_curvature_threshold
        idx = torch.where(is_peak & sharp)[0]
        if idx.numel() == 0:
            # A track with no peaks above the threshold would leave the
            # "next corner" observations with nothing to point at.
            idx = curvature.argmax().reshape(1)
        return idx

    def load_walls(self) -> None:
        """Extract the crash boundary from the walls USD on the live stage.

        Only near-horizontal triangle edges are kept: those are the top and
        bottom rails of each wall panel, which is all that is needed to measure
        how far the car is from a wall horizontally.

        Unlike the original implementation this raises rather than returning
        empty on failure.  Wall segments decide both crashes and lap validity,
        so silently continuing without them would produce a training run and a
        set of lap times that quietly mean nothing.
        """
        import omni.usd  # pyright: ignore[reportMissingImports]
        from pxr import Gf, Usd, UsdGeom  # pyright: ignore[reportMissingImports]

        stage = omni.usd.get_context().get_stage()
        walls_root = stage.GetPrimAtPath(self.walls_prim_path)
        if not walls_root.IsValid():
            raise RuntimeError(
                f"Track '{self.cfg.name}': no wall prim at {self.walls_prim_path}. "
                "The walls USD defines the crash boundary and is required."
            )

        xform_cache = UsdGeom.XformCache()
        segments: list[list[list[float]]] = []

        for prim in Usd.PrimRange(walls_root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            face_sizes = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()
            if points is None or face_sizes is None or face_indices is None:
                continue

            to_world = xform_cache.GetLocalToWorldTransform(prim)
            world = [
                to_world.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                for p in points
            ]

            cursor = 0
            for face_size in face_sizes:
                face = [int(face_indices[cursor + i]) for i in range(face_size)]
                cursor += face_size
                # Fan-triangulate, then keep the horizontal edges.
                for k in range(1, face_size - 1):
                    corners = (world[face[0]], world[face[k]], world[face[k + 1]])
                    for e in range(3):
                        p0, p1 = corners[e], corners[(e + 1) % 3]
                        if abs(p0[2] - p1[2]) > 0.05:
                            continue  # a vertical edge — not useful here
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                        if dx * dx + dy * dy < 1e-4:
                            continue  # zero length
                        segments.append(
                            [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]
                        )

        if not segments:
            raise RuntimeError(
                f"Track '{self.cfg.name}': the walls mesh produced no horizontal "
                "edges, so there is no crash boundary. Check that "
                f"{self.cfg.walls_usd} contains wall panels modelled as meshes."
            )

        seg = torch.tensor(segments, dtype=torch.float32, device=self.device)
        self.wall_seg_a = seg[:, 0, :].contiguous()
        self.wall_seg_b = seg[:, 1, :].contiguous()

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def walls_prim_path(self) -> str:
        return f"/World/track/{self.cfg.name}/walls"

    @property
    def surface_prim_path(self) -> str:
        return f"/World/track/{self.cfg.name}/surface"

    def nearest(self, pos_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest centerline point to each position.

        Args:
            pos_xy: ``[N, 2]`` world positions.

        Returns:
            ``(index, distance)``, each ``[N]``.
        """
        sq = (pos_xy.unsqueeze(1) - self.points.unsqueeze(0)).pow(2).sum(dim=-1)
        best = sq.min(dim=1)
        return best.indices, best.values.sqrt()

    def distance_to_walls(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Distance from each position to the nearest wall, ``[N]``.

        Measured against the wall segments directly rather than through physics
        contact reports: one small tensor operation, and it responds the instant
        the car gets too close.
        """
        seg_a, seg_b = self.wall_seg_a, self.wall_seg_b
        if seg_a is None or seg_b is None:
            raise RuntimeError(
                f"Track '{self.cfg.name}': walls have not been loaded. "
                "load_walls() runs during scene setup."
            )
        p = pos_xy.unsqueeze(1)  # [N, 1, 2]
        a = seg_a.unsqueeze(0)  # [1, S, 2]
        ab = seg_b.unsqueeze(0) - a
        t = ((p - a) * ab).sum(dim=-1) / (ab * ab).sum(dim=-1).clamp(min=1e-8)
        closest = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
        return (p - closest).pow(2).sum(dim=-1).min(dim=1).values.sqrt()

    def loop_distance_from_gate(self, nearest_idx: torch.Tensor) -> torch.Tensor:
        """How far round the loop each car is from the start/finish line.

        Measured the short way, so it rises to half the track length at the
        far side and falls again — the same either way round the track.
        """
        along = (self.arc_length[nearest_idx] - self.gate_arc) % self.track_length
        return torch.minimum(along, self.track_length - along)

    # ── Checking ──────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Sanity-check a track, returning a list of problems (empty is good).

        Worth calling on any track you build yourself: a centerline that is not
        a closed loop, or whose points are unevenly spaced, produces curvature
        and lookahead observations that are subtly wrong rather than obviously
        broken.
        """
        problems = []

        gap = float((self.points[0] - self.points[-1]).norm())
        if gap > 5.0 * self.point_spacing_m:
            problems.append(
                f"centerline is not a closed loop: first and last points are "
                f"{gap:.3f} m apart, versus {self.point_spacing_m:.3f} m typical "
                "spacing. The lap timer assumes a loop."
            )

        steps = (self.points - torch.roll(self.points, 1, dims=0)).norm(dim=1)[1:]
        if steps.numel() and float(steps.max()) > 10.0 * float(steps.median()):
            problems.append(
                f"centerline spacing is very uneven (largest gap "
                f"{float(steps.max()):.3f} m, typical {float(steps.median()):.3f} m); "
                "curvature will be noisy. Re-export at a uniform spacing."
            )

        if self.num_points < 50:
            problems.append(
                f"only {self.num_points} centerline points; lookahead "
                "observations need a denser line."
            )

        problems += self._wall_alignment_problems()

        if self.cfg.lap_gate_window_m >= 0.25 * self.track_length:
            problems.append(
                f"lap_gate_window_m ({self.cfg.lap_gate_window_m} m) is a large "
                f"fraction of the {self.track_length:.1f} m lap; the start/finish "
                "gate should be a small window, not a quarter of the track."
            )

        return problems

    #: A centerline point should have wall on both sides no further away than
    #: this multiple of the track's typical clearance. Generous, because a
    #: hairpin's outside rail is legitimately further off than a straight's.
    ENCLOSURE_TOLERANCE = 5.0

    def _wall_alignment_problems(self) -> list[str]:
        """Check that the centerline actually runs between the walls.

        The centerline is the middle of the track, so by definition it has wall
        on either side of it. When it does not, the transform that brought it
        out of Blender does not match the one the wall mesh got — and nothing
        else notices, because every track-relative observation is computed from
        the centerline alone and looks perfectly reasonable while being measured
        against a line that is not on the track.

        That is not hypothetical: the official centerline shipped scaled 15%
        larger than its own walls, ran outside them for a quarter of the lap,
        and took every cross-track error, lookahead point and curvature reading
        with it.
        """
        if self.wall_seg_a is None or self.wall_seg_b is None:
            return []  # nothing to compare against until the walls are loaded

        # A sample is plenty: this catches a systematic mismatch, not a single
        # stray point, and the comparison is every point against every segment.
        step = max(1, self.num_points // 200)
        points = self.points[::step]
        tangents = self.tangents[::step]
        left_normal = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)

        middles = 0.5 * (self.wall_seg_a + self.wall_seg_b)
        to_wall = middles.unsqueeze(0) - points.unsqueeze(1)  # [P, S, 2]
        distance = to_wall.norm(dim=-1)
        sideways = (to_wall * left_normal.unsqueeze(1)).sum(dim=-1)

        far = torch.full_like(distance, float("inf"))
        to_the_left = torch.where(sideways > 0, distance, far).min(dim=1).values
        to_the_right = torch.where(sideways < 0, distance, far).min(dim=1).values

        clearance = torch.minimum(to_the_left, to_the_right)
        typical = float(clearance.median())
        stranded = torch.maximum(to_the_left, to_the_right) > (
            self.ENCLOSURE_TOLERANCE * typical
        )
        outside = float(stranded.float().mean())

        problems = []
        if outside > 0.1:
            problems.append(
                f"{outside:.0%} of the centerline has wall on one side only, so it "
                "is not running between the walls. That is a coordinate mismatch "
                f"rather than a track: check that centerline_scale "
                f"{self.cfg.centerline_scale} agrees with the mesh_scale "
                f"{self.cfg.mesh_scale} the walls were built at."
            )

        tightest = float(self.distance_to_walls(points).min())
        if tightest < 0.05:
            problems.append(
                f"the centerline comes within {tightest:.3f} m of a wall, which "
                "leaves no room for a car; the centerline should be the middle of "
                "the track, not touching the edge of it."
            )
        return problems

    def describe(self) -> str:
        walls = 0 if self.wall_seg_a is None else self.wall_seg_a.shape[0]
        return (
            f"[track:{self.cfg.name}] {self.num_points} centerline points, "
            f"{self.track_length:.2f} m round, {self.corner_idx.numel()} corners, "
            f"{walls} wall segments"
        )
