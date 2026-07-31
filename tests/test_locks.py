"""Tests for the lock: frozen config, sealed methods, integrity.

These are the rules that make lap times comparable, so they are worth testing
directly rather than trusting.

    .venv/bin/python -m pytest tests/test_locks.py -q
"""

from __future__ import annotations

import pytest

from lituanicax_sdk._locked import (
    LockedParameterError,
    SealedMeta,
    assert_unchanged,
    sdk_fingerprint,
    sealed,
    verify_integrity,
)
from lituanicax_sdk.track import TrackCfg
from lituanicax_sdk.tracks import OFFICIAL, get, is_official, register
from lituanicax_sdk.vehicle import TIMING, VEHICLE, TimingCfg

# ── Frozen vehicle and timing ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "field",
    [
        "drive_torque_nm",
        "motor_no_load_speed_m_s",
        "mass_kg",
        "max_steer_rad",
        "wheel_radius_m",
        "ground_static_friction",
        "brake_torque_gain_nm",
        "action_dim",
    ],
)
def test_vehicle_parameters_cannot_be_changed(field):
    with pytest.raises(LockedParameterError, match=field):
        setattr(VEHICLE, field, 1.234)


@pytest.mark.parametrize(
    "field", ["policy_hz", "decimation", "physics_dt", "render_interval"]
)
def test_timing_cannot_be_changed(field):
    """Both the authored rates and the values derived from them refuse writes."""
    with pytest.raises(LockedParameterError, match=field):
        setattr(TIMING, field, 99)


def test_the_lock_error_explains_what_to_do_instead():
    with pytest.raises(LockedParameterError) as excinfo:
        VEHICLE.drive_torque_nm = 5.0
    message = str(excinfo.value)
    assert "docs/SDK.md" in message
    assert "drive_torque_nm" in message


def test_locked_values_can_still_be_read():
    """Teams need these to normalise observations."""
    assert VEHICLE.motor_no_load_speed_m_s == 6.7
    assert VEHICLE.wheel_radius_m == 0.037
    assert TIMING.policy_hz == 30.0
    assert TIMING.step_dt == pytest.approx(1 / 30)


def test_the_clock_is_defined_by_two_numbers():
    """policy_hz and decimation are authored; the rest follows from them.

    So the rate can be changed in one place without a stale ``physics_dt``
    silently contradicting it.
    """
    assert TIMING.decimation == 4
    assert TIMING.physics_hz == TIMING.policy_hz * TIMING.decimation
    assert TIMING.physics_dt == pytest.approx(1 / TIMING.physics_hz)
    assert TIMING.step_dt == pytest.approx(TIMING.physics_dt * TIMING.decimation)
    assert TIMING.render_interval == TIMING.decimation


def test_a_different_rate_stays_consistent():
    """The derivation holds for any rate, not just the one shipped."""
    faster = TimingCfg(policy_hz=60.0, decimation=8)
    assert faster.physics_hz == 480.0
    assert faster.physics_dt == pytest.approx(1 / 480)
    assert faster.step_dt == pytest.approx(1 / 60)
    assert faster.render_interval == 8


def test_deleting_a_locked_field_also_raises():
    with pytest.raises(LockedParameterError):
        del VEHICLE.mass_kg


# ── The Hydra escape hatch ────────────────────────────────────────────────


def test_assert_unchanged_accepts_the_locked_value():
    assert_unchanged(1 / (30.0 * 4), TIMING.physics_dt, "sim.dt")


def test_assert_unchanged_rejects_an_override():
    """This is what catches `train.py ... env.sim.dt=0.001` on the command line."""
    with pytest.raises(LockedParameterError) as excinfo:
        assert_unchanged(0.001, TIMING.physics_dt, "sim.dt")
    message = str(excinfo.value)
    assert "sim.dt" in message
    assert "env.sim.dt=" in message, "the error should name the likely cause"


# ── Sealed methods ────────────────────────────────────────────────────────


class _Base(metaclass=SealedMeta):
    @sealed
    def _apply_action(self) -> str:
        return "locked"

    def _get_rewards(self) -> str:
        return "open"


def test_overriding_an_open_method_is_fine():
    class Team(_Base):
        def _get_rewards(self):
            return "mine"

    assert Team()._get_rewards() == "mine"


def test_overriding_a_sealed_method_fails_at_class_definition():
    with pytest.raises(TypeError) as excinfo:

        class Team(_Base):
            def _apply_action(self):
                return "mine"

    message = str(excinfo.value)
    assert "_apply_action" in message
    assert "RaceEnvCfg" in message, "the error should say what to do instead"


def test_the_seal_survives_a_subclass_chain():
    class Middle(_Base):
        pass

    with pytest.raises(TypeError, match="_apply_action"):

        class Team(Middle):
            def _apply_action(self):
                return "mine"


# ── The SDK ships no reward or observation logic ──────────────────────────


def test_the_sdk_has_no_reward_or_observation_library():
    """The whole point of the split: teams write the logic, not pick from a menu."""
    import importlib

    for name in ("rewards", "observations", "terms", "terminations"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"lituanicax_sdk.{name}")


def test_the_env_requires_the_team_to_write_the_logic():
    """RaceEnv refuses to run without compute_observations / compute_reward.

    Read as source rather than imported: env.py needs a running Isaac Sim, and
    these tests deliberately do not.
    """
    import ast
    from pathlib import Path

    import lituanicax_sdk

    source = (Path(lituanicax_sdk.__file__).parent / "env.py").read_text()
    tree = ast.parse(source)
    race_env = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RaceEnv"
    )
    methods = {n.name: n for n in race_env.body if isinstance(n, ast.FunctionDef)}

    for name in ("compute_observations", "compute_reward"):
        assert name in methods, f"RaceEnv should define {name}"
        body = ast.dump(methods[name])
        assert "NotImplementedError" in body, f"{name} must refuse to guess"
        assert "team_solution" in body, f"{name} should say where to write it"

    # Terminations are optional: episodes run to time by default.
    assert "compute_terminations" in methods
    assert "NotImplementedError" not in ast.dump(methods["compute_terminations"])


# ── Official tracks ───────────────────────────────────────────────────────


def test_official_track_is_marked_official():
    assert is_official("official")
    assert get("official") is OFFICIAL


def test_official_tracks_cannot_be_redefined():
    fake = TrackCfg(
        name="official",
        walls_usd="x.usd",
        centerline_csv="x.csv",
        spawn_points=[(0, 0, 0)],
    )
    with pytest.raises(ValueError, match="cannot be redefined"):
        register(fake)


def test_teams_can_register_their_own_tracks():
    mine = TrackCfg(
        name="my_track",
        walls_usd="walls.usd",
        centerline_csv="line.csv",
        spawn_points=[(0.0, 0.0, 0.0)],
    )
    register(mine, overwrite=True)
    assert get("my_track") is mine
    assert not is_official("my_track")


def test_registering_twice_by_accident_is_caught():
    cfg = TrackCfg(
        name="dup", walls_usd="w.usd", centerline_csv="c.csv", spawn_points=[(0, 0, 0)]
    )
    register(cfg, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register(cfg)


def test_an_unknown_track_name_lists_what_is_available():
    with pytest.raises(KeyError, match="official"):
        get("does_not_exist")


# ── Integrity ─────────────────────────────────────────────────────────────


def test_a_stock_sdk_reports_no_changes():
    assert verify_integrity(quiet=True) == []


def test_the_fingerprint_is_stable_and_short():
    assert sdk_fingerprint() == sdk_fingerprint()
    assert len(sdk_fingerprint()) == 12


def test_modifying_the_sdk_is_detected(tmp_path, monkeypatch):
    import json

    from lituanicax_sdk import _locked

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"vehicle.py": "0" * 64}))
    monkeypatch.setattr(_locked, "MANIFEST_PATH", manifest)

    changed = verify_integrity(quiet=True)
    assert "vehicle.py" in changed
