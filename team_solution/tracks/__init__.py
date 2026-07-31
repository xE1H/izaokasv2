"""Your own tracks.

Training on more than the official layout is one of the more reliable ways to
stop a policy memorising a single track. Lap times are only ever *compared* on
official tracks, but nothing stops you training anywhere.

A track needs three things, and only the second is required:

* ``<name>_line.csv`` — the centerline: a closed loop of ``x,y`` points a few
  centimetres apart. This defines the racing line, the curvature the policy
  sees, and the start/finish line.
* ``<name>_walls.usdc`` — the walls, which become the crash boundary.
* ``<name>_surface.usdc`` — optional scenery: cones, tarmac, kerbs. Never given
  collision, so decorating a track cannot change how it drives.

Drop the files in this directory, register the track, and use it::

    from lituanicax_sdk import TrackCfg
    from lituanicax_sdk.tracks import register

    register(TrackCfg(
        name="figure_eight",
        walls_usd=str(HERE / "figure_eight_walls.usdc"),
        surface_usd=str(HERE / "figure_eight_surface.usdc"),
        centerline_csv=str(HERE / "figure_eight_line.csv"),
        # Blender to Isaac Sim, if your export needs it.
        centerline_scale=(1.0, 1.0),
        spawn_points=[(0.0, 0.0, 90.0)],  # (x, y, heading_deg) in CSV coords
    ))

then set ``track = get("figure_eight")`` in ``env_cfg.py``.

**Check it before you train on it.** A centerline that is not quite a closed
loop, or whose points are unevenly spaced, produces curvature and lookahead
observations that are subtly wrong rather than obviously broken — and you will
not notice for several hours::

    from lituanicax_sdk import Track
    from lituanicax_sdk.tracks import get

    track = Track(get("figure_eight"))
    for problem in track.validate():
        print(problem)
    print(track.describe())

This needs no simulator, so it is a two-second check.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

# Register your tracks here — this module is imported by team_solution/__init__.py.
