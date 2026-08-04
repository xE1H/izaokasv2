"""Where training runs live, and how to find a checkpoint in one.

Every run is one self-contained folder::

    logs/<timestamp>/
    ├── model_*.pt              checkpoints
    ├── events.out.tfevents.*   TensorBoard data
    ├── params/                 the exact env and PPO config used
    ├── exported/               policy.pt / policy.onnx, written by play.py
    ├── submission.json         written by the benchmark
    └── videos/                 only if you asked for one

Shared by ``teamcode/train.py``, ``teamcode/play.py`` and the benchmark so that
all three agree on the layout, and so none of them has to reimplement "find the
newest checkpoint".
"""

from __future__ import annotations

import os
from pathlib import Path

#: Hydra otherwise writes a config snapshot and a noisy log next to every run.
#: Both are redundant — train.py already saves the configs into the run's own
#: params/ folder — and they make one run mean two directories.
HYDRA_QUIET = [
    "hydra.run.dir=.",
    "hydra.output_subdir=null",
    "hydra/job_logging=disabled",
]


def project_root() -> Path:
    """The directory that owns ``logs/``.

    The working directory first, since the helper scripts cd into the project;
    that keeps working if the SDK is ever installed into a virtualenv rather
    than sitting in the repository.
    """
    if (Path.cwd() / "logs").is_dir():
        return Path.cwd()
    return Path(__file__).resolve().parent.parent


def logs_dir() -> Path:
    return project_root() / "logs"


def new_run_dir() -> Path:
    """A fresh timestamped folder for a training run."""
    from datetime import datetime

    return logs_dir() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def find_checkpoint(explicit: str | None = None) -> Path:
    """Resolve a checkpoint: the one given, else the newest anywhere in ``logs/``.

    Args:
        explicit: a path, absolute or relative to the project. When omitted,
            the most recently modified ``model_*.pt`` under ``logs/`` wins,
            which is almost always the run you just finished.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            candidate = project_root() / explicit
            if candidate.is_file():
                path = candidate
        if not path.is_file():
            raise SystemExit(f"No such checkpoint: {explicit}")
        return path.resolve()

    root = logs_dir()
    if not root.is_dir():
        raise SystemExit(
            f"No logs directory at {root} — train something first, or pass "
            "--checkpoint."
        )

    checkpoints = sorted(root.glob("*/model_*.pt"), key=os.path.getmtime)
    if not checkpoints:
        raise SystemExit(
            f"No checkpoints under {root} — train something first, or pass "
            "--checkpoint."
        )
    return checkpoints[-1].resolve()


def latest_run_dir() -> Path:
    """The run folder holding the newest checkpoint."""
    return find_checkpoint().parent
