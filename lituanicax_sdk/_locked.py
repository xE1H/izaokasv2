"""The lock: how the SDK keeps competition-critical parameters fixed.

Nothing here stops a determined person with a text editor — this is one
repository and teams can read every line of it.  What it does do is make the
boundary impossible to cross *by accident*, and impossible to cross *quietly*:

* :class:`LockedCfg` — a config whose fields raise on assignment, naming the
  parameter and pointing at the documentation.
* :func:`sealed` — marks methods that subclasses may not override, checked when
  the subclass is defined rather than when it misbehaves at runtime.
* :func:`verify_integrity` — hashes the SDK's own source and reports whether it
  matches the manifest, so a lap time can always be traced back to the rules it
  was set under.

The rule of thumb for what belongs behind the lock: if changing it makes the
*car* faster rather than the *policy* better, it is locked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_SDK_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = _SDK_DIR / "manifest.json"

_DOCS_HINT = (
    "This parameter is fixed by the competition rules so that every team's car "
    "behaves identically — see README.md for what you can change instead."
)


class LockedParameterError(RuntimeError):
    """Raised when locked configuration is modified."""


# ═══════════════════════════════════════════════════════════════════════════
#  Locked configuration
# ═══════════════════════════════════════════════════════════════════════════


class LockedCfg:
    """Marker base for configs whose values the competition fixes."""

    def __setattr__(self, name: str, value: object) -> None:
        raise LockedParameterError(
            f"{type(self).__name__}.{name} cannot be changed.\n{_DOCS_HINT}"
        )

    def __delattr__(self, name: str) -> None:
        raise LockedParameterError(
            f"{type(self).__name__}.{name} cannot be deleted.\n{_DOCS_HINT}"
        )


def locked_dataclass(cls):
    """``@dataclass(frozen=True)``, but the error says *why*.

    A frozen dataclass generates its own ``__setattr__`` on the class, which
    shadows anything inherited — so a team editing a locked value would get
    ``FrozenInstanceError: cannot assign to field 'drive_torque_nm'`` and no
    hint about what they should be changing instead. Swapping it back
    afterwards is safe: a frozen dataclass's ``__init__`` assigns through
    ``object.__setattr__`` directly, so construction never goes through here.
    """
    cls = dataclass(frozen=True)(cls)
    cls.__setattr__ = LockedCfg.__setattr__
    cls.__delattr__ = LockedCfg.__delattr__
    return cls


def _comparable(value: object) -> object:
    """Normalise a config value so equality means "the same setting".

    Isaac Lab's config round trip goes through OmegaConf, which turns every
    tuple into a list.  Comparing raw would flag a spawn-point list as tampered
    with purely for having survived being saved and loaded.
    """
    if isinstance(value, (list, tuple)):
        return [_comparable(v) for v in value]
    if isinstance(value, dict):
        return {k: _comparable(v) for k, v in value.items()}
    return value


def assert_unchanged(actual: object, expected: object, what: str) -> None:
    """Raise unless ``actual`` equals ``expected``.

    Used by the environment to re-check locked values that have to be copied
    into the Isaac Lab config (``sim.dt``, ``decimation``, the articulation),
    because those sit in a config tree that Hydra can write to from the command
    line.  Catching it here covers Hydra overrides, loaded YAML and plain
    assignment in one place.
    """
    if _comparable(actual) != _comparable(expected):
        raise LockedParameterError(
            f"{what} is {actual!r}, but the competition fixes it at "
            f"{expected!r}.\n{_DOCS_HINT}\n"
            "If this came from a command-line override such as "
            "`env.sim.dt=...`, remove it."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Sealed methods
# ═══════════════════════════════════════════════════════════════════════════


def sealed(func):
    """Mark a method that subclasses may not override.

    :class:`SealedMeta` checks the marks when a subclass is *defined*, so the
    mistake surfaces at import time with a message naming the method, rather
    than as a silently different simulation hours into a training run.
    """
    func.__lituanicax_sealed__ = True
    return func


class SealedMeta(type):
    """Metaclass that enforces :func:`sealed`."""

    def __new__(mcls, name, bases, namespace, **kwargs):
        # Walk the whole ancestry, not just the direct bases: an intermediate
        # subclass that overrides nothing would otherwise launder the seal.
        for base in bases:
            for ancestor in base.__mro__:
                for attr, value in vars(ancestor).items():
                    if not getattr(value, "__lituanicax_sealed__", False):
                        continue
                    if attr in namespace:
                        raise TypeError(
                            f"{name} overrides {ancestor.__name__}.{attr}(), "
                            f"which the LituanicaX SDK seals.\n"
                            f"{attr}() implements locked behaviour — the vehicle "
                            f"model, the crash rules or lap timing.\n"
                            "Configure the environment instead of subclassing "
                            "it: observations, rewards, terminations, spawning "
                            "and tracks are all fields on RaceEnvCfg.  "
                            "See README.md."
                        )
        return super().__new__(mcls, name, bases, namespace, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  Integrity
# ═══════════════════════════════════════════════════════════════════════════


def _hash_sources() -> dict[str, str]:
    """SHA-256 of every Python file in the SDK, keyed by relative path."""
    digests = {}
    for path in sorted(_SDK_DIR.rglob("*.py")):
        rel = path.relative_to(_SDK_DIR).as_posix()
        digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def write_manifest() -> Path:
    """Record the current sources as the reference. Run this after an SDK release."""
    MANIFEST_PATH.write_text(json.dumps(_hash_sources(), indent=2, sort_keys=True))
    return MANIFEST_PATH


def sdk_fingerprint() -> str:
    """One short hash covering the whole SDK, for stamping into run logs."""
    combined = json.dumps(_hash_sources(), sort_keys=True).encode()
    return hashlib.sha256(combined).hexdigest()[:12]


def verify_integrity(*, quiet: bool = False) -> list[str]:
    """Compare the SDK sources against the manifest.

    Returns the list of files that differ — empty when the SDK is stock.  This
    warns rather than raises: a team may have a good reason to experiment, and
    stopping their training run is not the SDK's call.  But the mismatch is
    printed, and :func:`sdk_fingerprint` goes into the run's logs next to the
    lap times, so nobody has to take a reported time on trust.
    """
    if not MANIFEST_PATH.is_file():
        return []

    expected = json.loads(MANIFEST_PATH.read_text())
    actual = _hash_sources()
    changed = sorted(
        name
        for name in set(expected) | set(actual)
        if expected.get(name) != actual.get(name)
    )

    if changed and not quiet:
        print("\n" + "!" * 74)
        print("!!  The LituanicaX SDK has been modified.")
        print("!!  Lap times from this run are not comparable with other teams'.")
        for name in changed:
            print(f"!!    {name}")
        print("!" * 74 + "\n")
    return changed
