"""The evidence that goes up with a lap: the policy that drove it.

A lap time on its own cannot be checked. The SDK is in your repository, you can
read and edit every line of it, and the fingerprint a run reports is computed by
*your* copy — so a team willing to edit ``_locked.py`` can report whatever
twelve characters it likes. Gating the board on that number gates it on the
honesty of the thing being checked.

What cannot be faked is a policy that drives the lap. So ``benchmark`` publishes
one: a zip holding the checkpoint it scored, your ``teamcode/`` — the
observations and the network shape, without which the checkpoint will not even
load — and the config the run used. The organiser downloads it, runs it against
a clean official SDK on their own machine, and the lap counts if it comes back.
Your fingerprint is now a hint about where to look, not the thing being trusted.

The bundle is small on purpose, and it is *code and configuration only*:

* ``checkpoint/<name>.pt`` — the policy that was scored;
* ``teamcode/`` — every source and data file in it, minus the caches and the
  source assets nothing needs to run (``.blend`` and friends);
* ``params/`` — the env and agent config the run was launched with;
* ``submission.json`` — the report, exactly as it was written to your run;
* ``manifest.json`` — what is in the zip, what it hashes to, and the exact
  command that reproduces the lap.

Nothing else from ``logs/`` goes anywhere: not the TensorBoard events, not the
videos, not the other checkpoints. One run's evidence is a few megabytes.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ._locked import sdk_fingerprint

#: Directories that are never worth sending, wherever they appear.
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".venv", "node_modules"}

#: Source assets: large, and useless to anything that only has to *run* the
#: policy. A Blender file is how a track was made, not what the simulator loads.
#: Everything else in ``teamcode/`` goes, including data files — a weights file
#: your own code loads at import time is part of your solution, not clutter.
SKIP_SUFFIXES = {".blend", ".blend1", ".fbx", ".obj", ".mp4", ".avi"}

#: Nothing in ``teamcode/`` should be near this. A file that is gets left out
#: and named in the manifest, rather than quietly making the upload fail.
MAX_FILE_BYTES = 8 * 1024 * 1024

#: What the board will accept in one bundle. Comfortably more than a checkpoint
#: and a source tree; small enough that nobody is hosting a dataset here.
MAX_BUNDLE_BYTES = 72 * 1024 * 1024


class BundleError(RuntimeError):
    """Raised when a bundle cannot be built. Caught at the edge, never fatal."""


@dataclass(frozen=True)
class Bundle:
    """A built bundle, ready to upload.

    Attributes:
        data: the zip itself.
        sha256: its digest, which is what the organiser checks after download.
        filename: a human name for it, carrying the team and the run.
        included: the paths inside the zip.
        skipped: what was left out, and why, so nothing vanishes silently.
    """

    data: bytes
    sha256: str
    filename: str
    included: list[str]
    skipped: list[str]

    @property
    def bytes(self) -> int:
        return len(self.data)

    @property
    def megabytes(self) -> float:
        return len(self.data) / 1_048_576


# ═══════════════════════════════════════════════════════════════════════════
#  What goes in
# ═══════════════════════════════════════════════════════════════════════════


def _collect(root: Path, prefix: str, skipped: list[str]) -> list[tuple[str, Path]]:
    """Every file under ``root`` worth sending, as ``(name in zip, path)``."""
    if not root.is_dir():
        return []

    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.parts):
            continue

        relative = f"{prefix}/{path.relative_to(root).as_posix()}"
        if path.suffix.lower() in SKIP_SUFFIXES:
            skipped.append(
                f"{relative} ({path.suffix} is not needed to run the policy)"
            )
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped.append(
                f"{relative} ({size / 1_048_576:.1f} MB, over the per-file limit)"
            )
            continue
        found.append((relative, path))
    return found


def _describe_run(report: dict, checkpoint: Path) -> dict:
    """The command that reproduces this lap, and everything it depends on.

    Written down rather than inferred, so that a bundle opened months later
    still says what it was: the benchmark's conditions are defaults today and
    might not be forever.
    """
    return {
        "command": [
            "python",
            "-m",
            "lituanicax_sdk.benchmark",
            "--headless",
            "--checkpoint",
            f"checkpoint/{checkpoint.name}",
            f"--agents={report.get('attempts')}",
            f"--seed={report.get('seed')}",
            f"--spawn-jitter={report.get('spawn_jitter_deg')}",
        ],
        "track": report.get("track"),
        "attempts": report.get("attempts"),
        "seed": report.get("seed"),
        "spawn": report.get("spawn"),
        "spawn_jitter_deg": report.get("spawn_jitter_deg"),
        "claimed_lap_time_s": report.get("best_lap_time_s"),
    }


def _manifest(
    report: dict,
    checkpoint: Path,
    contents: dict[str, str],
    skipped: list[str],
) -> dict:
    return {
        "format": 1,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "sdk_fingerprint": sdk_fingerprint(),
        "sdk_modified": list(report.get("sdk_modified") or []),
        # What the SDK's own critical code hashed to at the end of the scoring
        # run. The organiser compares it against a clean process of their own:
        # a team that patches the SDK in memory rather than on disk moves this
        # and not the source fingerprint. See _locked.runtime_fingerprint.
        "runtime_fingerprint": report.get("runtime_fingerprint"),
        "checkpoint": {
            "name": checkpoint.name,
            "original_path": str(checkpoint),
            "sha256": contents.get(f"checkpoint/{checkpoint.name}"),
        },
        "reproduce": _describe_run(report, checkpoint),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _torch_version(),
        },
        "contents": contents,
        "skipped": skipped,
    }


def _torch_version() -> str | None:
    """Recorded when torch is already imported; never imported for this."""
    module = sys.modules.get("torch")
    return getattr(module, "__version__", None) if module else None


# ═══════════════════════════════════════════════════════════════════════════
#  Building
# ═══════════════════════════════════════════════════════════════════════════


def build_bundle(
    report: dict,
    checkpoint: str | Path,
    *,
    team: str = "team",
    project_root: Path | None = None,
) -> Bundle:
    """Zip up everything needed to re-drive this lap somewhere else.

    Args:
        report: the benchmark's report, as written to ``submission.json``.
        checkpoint: the policy that was scored.
        team: only used to name the file.
        project_root: where ``teamcode/`` lives. Defaults to the project.

    Raises:
        BundleError: if the checkpoint is missing, or the bundle comes out
            larger than the board will take. Both are conditions the caller
            reports and carries on from — the lap has already been driven.
    """
    from .runs import project_root as default_root

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise BundleError(f"the checkpoint is not there any more: {checkpoint}")

    root = Path(project_root) if project_root else default_root()
    skipped: list[str] = []

    files = [(f"checkpoint/{checkpoint.name}", checkpoint)]
    files += _collect(root / "teamcode", "teamcode", skipped)
    # The config the run was actually launched with, which lives beside the
    # checkpoint rather than in the source tree.
    files += _collect(checkpoint.parent / "params", "params", skipped)

    contents: dict[str, str] = {}
    buffer = io.BytesIO()
    # Deterministic: no timestamps from the filesystem, sorted order, so the
    # same run bundled twice produces the same bytes and the same digest.
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in files:
            data = path.read_bytes()
            contents[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(_entry(name), data)

        report_bytes = json.dumps(report, indent=2, sort_keys=True).encode()
        contents["submission.json"] = hashlib.sha256(report_bytes).hexdigest()
        archive.writestr(_entry("submission.json"), report_bytes)

        manifest = _manifest(report, checkpoint, contents, skipped)
        archive.writestr(
            _entry("manifest.json"), json.dumps(manifest, indent=2).encode()
        )

    data = buffer.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        raise BundleError(
            f"the bundle is {len(data) / 1_048_576:.1f} MB, over the "
            f"{MAX_BUNDLE_BYTES / 1_048_576:.0f} MB limit. Something large is in "
            "teamcode/ that the policy does not need to run."
        )

    return Bundle(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        filename=_filename(team, checkpoint),
        included=sorted(contents),
        skipped=skipped,
    )


def _entry(name: str) -> zipfile.ZipInfo:
    """A zip entry with a fixed timestamp, so the archive is reproducible."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _filename(team: str, checkpoint: Path) -> str:
    safe = (
        "".join(c if c.isalnum() or c in "-_" else "-" for c in team).strip("-")
        or "team"
    )
    return f"{safe}-{checkpoint.parent.name}-{checkpoint.stem}.zip"


# ═══════════════════════════════════════════════════════════════════════════
#  Reading one back
# ═══════════════════════════════════════════════════════════════════════════


def extract_bundle(data: bytes, into: str | Path) -> dict:
    """Unpack a downloaded bundle and check it against its own manifest.

    Used by :mod:`lituanicax_sdk.verify` before it runs anything. Every entry is
    hashed and compared, and any path that tries to climb out of the target
    directory stops the extraction — the zip came from a competitor, and a
    ``../`` in a member name is the oldest trick there is.

    Returns:
        The bundle's manifest.

    Raises:
        BundleError: on a bad path, a missing entry or a digest that does not
            match what the manifest claims.
    """
    into = Path(into).resolve()
    into.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if "manifest.json" not in names:
            raise BundleError(
                "this zip has no manifest.json, so it is not a policy bundle."
            )

        for name in names:
            target = (into / name).resolve()
            if not str(target).startswith(str(into) + "/"):
                raise BundleError(
                    f"the bundle tries to write outside its directory: {name}"
                )

        manifest = json.loads(archive.read("manifest.json"))
        archive.extractall(into)

    contents = manifest.get("contents") or {}
    for name, expected in contents.items():
        path = into / name
        if not path.is_file():
            raise BundleError(
                f"the manifest lists {name}, and the bundle does not contain it."
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise BundleError(
                f"{name} does not match the manifest: {actual} vs {expected}."
            )

    return manifest
