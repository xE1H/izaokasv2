"""The policy bundle: what a lap takes with it, and what comes back out.

None of this needs Isaac Sim or a GPU — a bundle is a zip of files and a
manifest, and the interesting questions are which files, hashed how, and what
happens when somebody hands the organiser a hostile one.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from lituanicax_sdk.bundle import (
    MAX_FILE_BYTES,
    Bundle,
    BundleError,
    build_bundle,
    extract_bundle,
)


@pytest.fixture
def project(tmp_path):
    """A project the shape ``build_bundle`` expects: teamcode, and a run."""
    teamcode = tmp_path / "teamcode"
    teamcode.mkdir()
    (teamcode / "__init__.py").write_text("# registers the task\n")
    (teamcode / "env.py").write_text("OBSERVATIONS = 3\n")
    (teamcode / "tracks").mkdir()
    (teamcode / "tracks" / "__init__.py").write_text("# my tracks\n")

    run = tmp_path / "logs" / "2026-08-10_12-00-00"
    (run / "params").mkdir(parents=True)
    (run / "params" / "env.yaml").write_text("dt: 0.005\n")
    (run / "params" / "agent.yaml").write_text("max_iterations: 500\n")
    (run / "model_500.pt").write_bytes(b"not really a checkpoint, but bytes are bytes")
    (run / "events.out.tfevents.1").write_bytes(b"x" * 4096)
    return tmp_path


@pytest.fixture
def report():
    """What the benchmark writes to submission.json."""
    return {
        "best_lap_time_s": 15.204,
        "attempts": 10,
        "laps_completed": 7,
        "track": "official",
        "seed": 0,
        "spawn_jitter_deg": 5.0,
        "spawn": "(0.00, 0.00) along the track",
        "sdk_fingerprint": "a1b2c3d4e5f6",
        "runtime_fingerprint": "0f0f0f0f0f0f",
        "sdk_modified": [],
    }


def build(project, report, **kwargs) -> Bundle:
    return build_bundle(
        report,
        project / "logs" / "2026-08-10_12-00-00" / "model_500.pt",
        project_root=project,
        **kwargs,
    )


def names_in(bundle: Bundle) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        return sorted(archive.namelist())


# ═══════════════════════════════════════════════════════════════════════════
#  What goes in
# ═══════════════════════════════════════════════════════════════════════════


def test_the_bundle_carries_the_policy_the_team_code_and_the_config(project, report):
    assert names_in(build(project, report)) == [
        "checkpoint/model_500.pt",
        "manifest.json",
        "params/agent.yaml",
        "params/env.yaml",
        "submission.json",
        "teamcode/__init__.py",
        "teamcode/env.py",
        "teamcode/tracks/__init__.py",
    ]


def test_nothing_else_from_the_run_goes_anywhere(project, report):
    """A run folder is gigabytes of TensorBoard and old checkpoints."""
    included = names_in(build(project, report))
    assert not [name for name in included if "tfevents" in name]
    assert [name for name in included if name.startswith("checkpoint/")] == [
        "checkpoint/model_500.pt"
    ]


def test_caches_and_dot_directories_are_left_out(project, report):
    (project / "teamcode" / "__pycache__").mkdir()
    (project / "teamcode" / "__pycache__" / "env.cpython-311.pyc").write_bytes(b"\x00")
    (project / "teamcode" / ".ipynb_checkpoints").mkdir()
    (project / "teamcode" / ".ipynb_checkpoints" / "env.py").write_text("stale")

    assert not [name for name in names_in(build(project, report)) if "cache" in name]
    assert not [
        name for name in names_in(build(project, report)) if "checkpoints" in name
    ]


def test_source_assets_are_named_rather_than_sent(project, report):
    """A 7 MB .blend is how a track was made, not what the simulator loads."""
    (project / "teamcode" / "tracks" / "Track.blend").write_bytes(b"blender" * 100)

    bundle = build(project, report)
    assert "teamcode/tracks/Track.blend" not in names_in(bundle)
    assert any("Track.blend" in line for line in bundle.skipped)


def test_a_data_file_the_team_actually_loads_is_sent(project, report):
    """Anything a policy might need at import time travels with it."""
    (project / "teamcode" / "tracks" / "figure_eight_line.csv").write_text("0,0\n1,1\n")

    assert "teamcode/tracks/figure_eight_line.csv" in names_in(build(project, report))


def test_an_oversized_file_is_left_out_and_reported(project, report):
    big = project / "teamcode" / "huge.npy"
    big.write_bytes(b"\x00" * (MAX_FILE_BYTES + 1))

    bundle = build(project, report)
    assert "teamcode/huge.npy" not in names_in(bundle)
    assert any(
        "huge.npy" in line and "per-file limit" in line for line in bundle.skipped
    )


def test_a_missing_checkpoint_is_reported_not_swallowed(project, report):
    (project / "logs" / "2026-08-10_12-00-00" / "model_500.pt").unlink()

    with pytest.raises(BundleError, match="checkpoint"):
        build(project, report)


def test_a_project_without_teamcode_still_bundles_the_policy(tmp_path, report):
    """Rare, but a bundle missing teamcode is better than no bundle at all."""
    run = tmp_path / "logs" / "run"
    run.mkdir(parents=True)
    (run / "model_1.pt").write_bytes(b"weights")

    bundle = build_bundle(report, run / "model_1.pt", project_root=tmp_path)
    assert "checkpoint/model_1.pt" in names_in(bundle)


# ═══════════════════════════════════════════════════════════════════════════
#  The manifest
# ═══════════════════════════════════════════════════════════════════════════


def manifest_of(bundle: Bundle) -> dict:
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        return json.loads(archive.read("manifest.json"))


def test_every_file_is_hashed_so_a_download_can_be_checked(project, report):
    bundle = build(project, report)
    manifest = manifest_of(bundle)

    assert set(manifest["contents"]) == set(names_in(bundle)) - {"manifest.json"}
    assert all(len(digest) == 64 for digest in manifest["contents"].values())


def test_the_manifest_says_how_to_reproduce_the_lap(project, report):
    reproduce = manifest_of(build(project, report))["reproduce"]

    assert reproduce["claimed_lap_time_s"] == pytest.approx(15.204)
    assert reproduce["seed"] == 0
    assert reproduce["attempts"] == 10
    assert "--agents=10" in reproduce["command"]
    assert "lituanicax_sdk.benchmark" in reproduce["command"]


def test_the_manifest_carries_both_fingerprints(project, report):
    """The source one, and what the SDK's code hashed to as it actually ran."""
    manifest = manifest_of(build(project, report))

    assert manifest["runtime_fingerprint"] == "0f0f0f0f0f0f"
    assert len(manifest["sdk_fingerprint"]) == 12
    assert manifest["sdk_modified"] == []


def test_the_same_run_bundles_to_the_same_bytes(project, report):
    """A zip full of filesystem timestamps would hash differently every time."""
    first, second = build(project, report), build(project, report)

    with (
        zipfile.ZipFile(io.BytesIO(first.data)) as a,
        zipfile.ZipFile(io.BytesIO(second.data)) as b,
    ):
        assert [i.date_time for i in a.infolist()] == [
            i.date_time for i in b.infolist()
        ]
        assert [i.filename for i in a.infolist()] == [i.filename for i in b.infolist()]


def test_the_filename_says_whose_run_it_was(project, report):
    bundle = build(project, report, team="Wingless Wonders")
    assert bundle.filename == "Wingless-Wonders-2026-08-10_12-00-00-model_500.zip"


# ═══════════════════════════════════════════════════════════════════════════
#  Reading one back
# ═══════════════════════════════════════════════════════════════════════════


def test_a_bundle_unpacks_to_something_runnable(project, report, tmp_path):
    bundle = build(project, report)
    into = tmp_path / "workspace"

    manifest = extract_bundle(bundle.data, into)

    assert (into / "teamcode" / "env.py").read_text() == "OBSERVATIONS = 3\n"
    assert (into / "checkpoint" / "model_500.pt").is_file()
    assert manifest["checkpoint"]["name"] == "model_500.pt"


def test_a_tampered_bundle_is_refused(project, report, tmp_path):
    """The manifest is what the organiser checks the download against."""
    bundle = build(project, report)
    swapped = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as source:
        with zipfile.ZipFile(swapped, "w") as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == "teamcode/env.py":
                    data = b"OBSERVATIONS = 999\n"
                target.writestr(info, data)

    with pytest.raises(BundleError, match="does not match the manifest"):
        extract_bundle(swapped.getvalue(), tmp_path / "workspace")


def test_a_bundle_cannot_write_outside_its_own_directory(tmp_path):
    """The zip comes from a competitor, and `../` is the oldest trick there is."""
    hostile = io.BytesIO()
    with zipfile.ZipFile(hostile, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"contents": {}}))
        archive.writestr("../escaped.py", "print('pwned')")

    with pytest.raises(BundleError, match="outside"):
        extract_bundle(hostile.getvalue(), tmp_path / "workspace")
    assert not (tmp_path / "escaped.py").exists()


def test_a_zip_that_is_not_a_bundle_is_refused(tmp_path):
    plain = io.BytesIO()
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr("holiday.jpg", "not a policy")

    with pytest.raises(BundleError, match="manifest"):
        extract_bundle(plain.getvalue(), tmp_path / "workspace")


def test_a_bundle_missing_a_file_it_promises_is_refused(tmp_path):
    incomplete = io.BytesIO()
    with zipfile.ZipFile(incomplete, "w") as archive:
        archive.writestr(
            "manifest.json", json.dumps({"contents": {"checkpoint/model.pt": "0" * 64}})
        )

    with pytest.raises(BundleError, match="does not contain"):
        extract_bundle(incomplete.getvalue(), tmp_path / "workspace")
