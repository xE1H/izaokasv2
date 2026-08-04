"""The official tracks, and the registry your own tracks go in.

The official tracks are locked: their geometry and start/finish line are fixed
so that a lap time on one means the same thing for everyone. Where cars *start*
is not part of a track — that is your curriculum, and it lives with your spawn
manager in ``teamcode``.

Your own tracks are not locked. Register as many as you like and train on them::

    from lituanicax_sdk.tracks import register, TrackCfg

    register(TrackCfg(
        name="my_track",
        walls_usd="teamcode/tracks/my_walls.usdc",
        surface_usd="teamcode/tracks/my_surface.usdc",
        centerline_csv="teamcode/tracks/my_line.csv",
    ))

Then set ``cfg.track = get("my_track")``. Call ``Track(cfg).validate()`` on a
new track before you spend a night training on it — a centerline that is not
quite a closed loop produces curvature and lookahead observations that are
subtly wrong rather than obviously broken.
"""

from __future__ import annotations

from pathlib import Path

from ..track import Track, TrackCfg

_HERE = Path(__file__).resolve().parent

# ── The official track ────────────────────────────────────────────────────
# The cone-lined loop the competition is run on. Locked.

OFFICIAL = TrackCfg(
    name="official",
    surface_usd=str(_HERE / "official" / "Track.usdc"),
    walls_usd=str(_HERE / "official" / "Walls.usdc"),
    centerline_csv=str(_HERE / "official" / "centerline.csv"),
    # The track mesh is modelled 1 / 0.85 too large.
    mesh_scale=0.85,
    # Blender to Isaac Sim. The centerline gets the same scale as the mesh it
    # describes — measured against the walls this puts it 0.351 m from either
    # side the whole way round, on a track 0.70 m wide.
    centerline_scale=(-0.85, -0.85),
    start_finish_index=0,
    lap_gate_window_m=2.0,
    corner_curvature_threshold=0.5,
)

#: Tracks that lap times are officially compared on.
OFFICIAL_TRACKS: dict[str, TrackCfg] = {OFFICIAL.name: OFFICIAL}

_REGISTRY: dict[str, TrackCfg] = dict(OFFICIAL_TRACKS)


def register(cfg: TrackCfg, *, overwrite: bool = False) -> TrackCfg:
    """Add a track to the registry so it can be looked up by name."""
    if cfg.name in OFFICIAL_TRACKS:
        raise ValueError(
            f"'{cfg.name}' is an official track and cannot be redefined. "
            "Pick another name — you can register as many of your own as you like."
        )
    if cfg.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"A track named '{cfg.name}' is already registered. "
            "Pass overwrite=True if you meant to replace it."
        )
    _REGISTRY[cfg.name] = cfg
    return cfg


def get(name: str) -> TrackCfg:
    """Look up a registered track by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"No track named '{name}'. Registered: {sorted(_REGISTRY)}. "
            "Use lituanicax_sdk.tracks.register() to add your own."
        )
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


def is_official(name: str) -> bool:
    return name in OFFICIAL_TRACKS


__all__ = [
    "OFFICIAL",
    "OFFICIAL_TRACKS",
    "Track",
    "TrackCfg",
    "get",
    "is_official",
    "names",
    "register",
]
