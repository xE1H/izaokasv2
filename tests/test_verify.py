"""The organiser's verification client, against a real HTTP server.

The one thing not exercised here is the driving: re-running a policy needs
Isaac Sim and a GPU, so the subprocess is faked and what is checked is
everything around it — the queue, the download and its hash, how the benchmark
is invoked, and what gets posted back for a lap that came out right, a lap that
came out wrong, and a policy that never finished one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lituanicax_sdk import verify as verify_module
from lituanicax_sdk.bundle import build_bundle
from lituanicax_sdk.verify import (
    Organiser,
    Rerun,
    VerificationError,
    _tamper_note,
    download_bundle,
    load_organiser,
    main,
    pending,
    post_verdict,
    rerun_bundle,
)

TOKEN = "the-organisers-token"
SUBMISSION = "0bb1e1ba-0000-4000-8000-000000000001"


# ═══════════════════════════════════════════════════════════════════════════
#  A stand-in board, with the admin half implemented
# ═══════════════════════════════════════════════════════════════════════════


class FakeBoard:
    """Answers the three admin endpoints ``verify`` uses, and records the calls."""

    def __init__(self, bundle_data: bytes = b"", part_size: int = 64):
        self.bundle = bundle_data
        self.part_size = part_size
        self.claimed_sha = hashlib.sha256(bundle_data).hexdigest()
        self.verdicts: list[dict] = []
        self.tokens: list[str] = []
        # The queue, as `id: (team, lap, has a policy)`. One lap unless a test
        # wants more.
        self.queue: dict[str, tuple[str, float, bool]] = {
            SUBMISSION: ("Alpha", 15.204, True)
        }
        # Every call is admin-gated on the real board; the fake one says so too.
        self.expects_token = TOKEN
        self._server = HTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def parts(self) -> int:
        """Zero when there is no bundle, which is how a lap with no policy reads."""
        return -(-len(self.bundle) // self.part_size)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _handler(board):  # noqa: N805 — the closure needs the board, not `self`
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler's spelling
                board.tokens.append(self.headers.get("X-Admin-Token", ""))
                if not self._authorised():
                    return self._respond(401, {"message": "Missing or invalid token."})
                if self.path.startswith("/api/admin/submissions"):
                    return self._respond(200, {"submissions": board._submissions()})
                if "part=" in self.path:
                    part = int(self.path.split("part=")[1])
                    chunk = board.bundle[
                        part * board.part_size : (part + 1) * board.part_size
                    ]
                    return self._respond(
                        200, {"part": part, "data": base64.b64encode(chunk).decode()}
                    )
                if self.path.startswith("/api/admin/artifact"):
                    wanted = self.path.split("id=")[1].split("&")[0]
                    team = board.queue.get(wanted, ("Alpha", 0.0, True))[0]
                    return self._respond(
                        200, {"team": team, "bundle": board._bundle(wanted)}
                    )
                return self._respond(404, {"message": "no such thing"})

            def do_POST(self):  # noqa: N802
                board.tokens.append(self.headers.get("X-Admin-Token", ""))
                if not self._authorised():
                    return self._respond(401, {"message": "Missing or invalid token."})
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                board.verdicts.append(body)
                verified = "lap_time_s" in body
                return self._respond(
                    200,
                    {
                        "ranked": verified,
                        "verification": {
                            "state": "verified" if verified else "rejected"
                        },
                        "message": "verified." if verified else "rejected.",
                    },
                )

            def _authorised(self) -> bool:
                return self.headers.get("X-Admin-Token", "") == board.expects_token

            def _respond(self, status, body):
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        return Handler

    def _bundle(self, submission_id: str = SUBMISSION) -> dict | None:
        if not self.queue.get(submission_id, ("", 0.0, True))[2]:
            return None
        return {
            "sha256": self.claimed_sha,
            "bytes": len(self.bundle),
            "parts": self.parts,
            "filename": "alpha.zip",
        }

    def _submissions(self) -> list[dict]:
        return [
            {
                "id": submission_id,
                "team": team,
                "lap_time_s": lap,
                "status": "pending_verification",
                "verification": {
                    "state": "pending",
                    "bundle": self._bundle(submission_id),
                },
            }
            for submission_id, (team, lap, _) in self.queue.items()
        ]


@pytest.fixture
def policy_bundle(tmp_path):
    """A real bundle, built the way the benchmark builds one."""
    teamcode = tmp_path / "teamcode"
    teamcode.mkdir()
    (teamcode / "__init__.py").write_text("# the team's task registration\n")
    run = tmp_path / "logs" / "run"
    run.mkdir(parents=True)
    (run / "model_9.pt").write_bytes(b"weights" * 50)

    return build_bundle(
        {
            "best_lap_time_s": 15.204,
            "attempts": 10,
            "seed": 0,
            "spawn_jitter_deg": 5.0,
            "sdk_modified": [],
            "runtime_fingerprint": "aaaaaaaaaaaa",
        },
        run / "model_9.pt",
        project_root=tmp_path,
    )


@pytest.fixture
def board(policy_bundle):
    server = FakeBoard(policy_bundle.data)
    yield server
    server.close()


@pytest.fixture
def organiser(board):
    return Organiser(url=board.url, token=TOKEN)


# ═══════════════════════════════════════════════════════════════════════════
#  Who is asking
# ═══════════════════════════════════════════════════════════════════════════


def test_the_token_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)
    assert load_organiser().token == TOKEN


def test_verifying_without_the_admin_token_says_what_to_set(monkeypatch):
    monkeypatch.delenv("LEADERBOARD_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    with pytest.raises(VerificationError, match="LEADERBOARD_ADMIN_TOKEN"):
        load_organiser()


def test_every_call_carries_the_token(organiser, board):
    pending(organiser)
    assert board.tokens == [TOKEN]


# ═══════════════════════════════════════════════════════════════════════════
#  Collecting the policy
# ═══════════════════════════════════════════════════════════════════════════


def test_the_queue_is_what_is_waiting_to_be_reproduced(organiser):
    waiting = pending(organiser)

    assert [s["team"] for s in waiting] == ["Alpha"]
    assert waiting[0]["lap_time_s"] == pytest.approx(15.204)


def test_a_bundle_is_reassembled_from_its_parts(organiser, board, policy_bundle):
    assert board.parts > 1, "the fixture should need more than one request"
    assert download_bundle(organiser, SUBMISSION) == policy_bundle.data


def test_a_bundle_that_does_not_match_its_manifest_is_refused(organiser, board):
    """The board stored whatever it was given; the hash is checked here."""
    board.claimed_sha = "0" * 64

    with pytest.raises(VerificationError, match="does not match its manifest"):
        download_bundle(organiser, SUBMISSION)


def test_a_lap_published_without_a_policy_cannot_be_verified(organiser, board):
    board.bundle = b""

    with pytest.raises(VerificationError, match="without a policy"):
        download_bundle(organiser, SUBMISSION)


# ═══════════════════════════════════════════════════════════════════════════
#  Re-running it
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def fake_benchmark(monkeypatch, tmp_path):
    """Stand in for Isaac Sim: record the command, write the report it would."""
    calls = {}

    def run(command, *, cwd, env, timeout, capture_output, text):
        calls["command"] = command
        calls["cwd"] = cwd
        calls["env"] = env
        out = command[command.index("--out") + 1]
        report = calls.get(
            "report", {"best_lap_time_s": 15.2, "runtime_fingerprint": "aaaaaaaaaaaa"}
        )
        if report is not None:
            with open(out, "w") as handle:
                json.dump(report, handle)
        return subprocess.CompletedProcess(command, 0, "benchmark output\n", "")

    monkeypatch.setattr(verify_module.subprocess, "run", run)
    return calls


def test_the_teams_own_code_is_what_runs_and_this_sdk_is_what_it_runs_on(
    policy_bundle, fake_benchmark, tmp_path
):
    workspace = tmp_path / "workspace"
    rerun_bundle(policy_bundle.data, workspace)

    # The workspace is the working directory, so `import teamcode` finds the
    # extracted one; the repository is behind it on PYTHONPATH and supplies the
    # SDK. That ordering is the whole of the isolation.
    assert fake_benchmark["cwd"] == workspace
    assert (workspace / "teamcode" / "__init__.py").is_file()
    assert str(verify_module.Path(__file__).resolve().parent.parent) in fake_benchmark[
        "env"
    ]["PYTHONPATH"].split(":")


def test_the_rerun_uses_the_conditions_the_lap_was_set_under(
    policy_bundle, fake_benchmark, tmp_path
):
    rerun_bundle(policy_bundle.data, tmp_path / "workspace")
    command = fake_benchmark["command"]

    assert "--agents=10" in command
    assert "--seed=0" in command
    assert "--spawn-jitter=5.0" in command
    assert "--no-submit" in command, "a verification run publishes nothing of its own"
    assert "--headless" in command


def test_the_rerun_reports_the_lap_this_machine_got(
    policy_bundle, fake_benchmark, tmp_path
):
    rerun = rerun_bundle(policy_bundle.data, tmp_path / "workspace")

    assert rerun.drove_a_lap
    assert rerun.lap_time_s == pytest.approx(15.2)
    assert rerun.runtime_fingerprint == "aaaaaaaaaaaa"


def test_a_policy_that_completes_no_lap_is_a_result_not_an_error(
    policy_bundle, fake_benchmark, tmp_path
):
    fake_benchmark["report"] = {"best_lap_time_s": None}
    rerun = rerun_bundle(policy_bundle.data, tmp_path / "workspace")

    assert not rerun.drove_a_lap


def test_a_benchmark_that_writes_nothing_is_an_error(
    policy_bundle, fake_benchmark, tmp_path
):
    fake_benchmark["report"] = None

    with pytest.raises(VerificationError, match="no report"):
        rerun_bundle(policy_bundle.data, tmp_path / "workspace")


def test_the_workspace_is_rebuilt_rather_than_merged(
    policy_bundle, fake_benchmark, tmp_path
):
    """Yesterday's teamcode must not be half of today's verification."""
    workspace = tmp_path / "workspace"
    (workspace / "teamcode").mkdir(parents=True)
    (workspace / "teamcode" / "stale.py").write_text("from another team\n")

    rerun_bundle(policy_bundle.data, workspace)
    assert not (workspace / "teamcode" / "stale.py").exists()


# ═══════════════════════════════════════════════════════════════════════════
#  Saying what happened
# ═══════════════════════════════════════════════════════════════════════════


def rerun_of(lap, fingerprint="aaaaaaaaaaaa", tmp_path=None):
    return Rerun(
        lap_time_s=lap, runtime_fingerprint=fingerprint, report={}, workspace=tmp_path
    )


def test_a_lap_posts_the_time_and_lets_the_board_judge(organiser, board):
    """The client measures; the tolerance and the verdict are the board's."""
    post_verdict(organiser, SUBMISSION, rerun_of(15.21))

    assert board.verdicts[-1]["lap_time_s"] == pytest.approx(15.21)
    assert "state" not in board.verdicts[-1], "the client does not decide"


def test_no_lap_posts_a_rejection(organiser, board):
    post_verdict(organiser, SUBMISSION, rerun_of(None), note="the policy crashed")

    assert board.verdicts[-1]["state"] == "rejected"
    assert board.verdicts[-1]["note"] == "the policy crashed"


def test_an_sdk_that_behaved_differently_is_flagged():
    note = _tamper_note(rerun_of(15.2, "deadbeefdead"), baseline="aaaaaaaaaaaa")

    assert "WARNING" in note
    assert "deadbeefdead" in note and "aaaaaaaaaaaa" in note


def test_an_untouched_sdk_says_nothing():
    assert _tamper_note(rerun_of(15.2, "aaaaaaaaaaaa"), baseline="aaaaaaaaaaaa") == ""


def test_the_check_is_skipped_rather_than_failed_when_there_is_no_baseline():
    assert _tamper_note(rerun_of(15.2, "deadbeefdead"), baseline=None) == ""


# ═══════════════════════════════════════════════════════════════════════════
#  The command
# ═══════════════════════════════════════════════════════════════════════════


def test_listing_the_queue_runs_nothing(board, monkeypatch, capsys):
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)

    assert main(["--list", "--url", board.url]) == 0
    printed = capsys.readouterr().out
    assert "Alpha" in printed
    assert SUBMISSION in printed
    assert board.verdicts == []


def test_verifying_an_unknown_submission_says_so(board, monkeypatch, capsys):
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)

    assert main(["not-a-submission", "--url", board.url]) == 1
    assert "not waiting" in capsys.readouterr().out


def test_a_dry_run_re_runs_the_lap_and_posts_nothing(
    board, monkeypatch, fake_benchmark, tmp_path, capsys
):
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(
        verify_module, "clean_runtime_fingerprint", lambda: "aaaaaaaaaaaa"
    )

    code = main(
        [
            SUBMISSION,
            "--url",
            board.url,
            "--dry-run",
            "--workspace",
            str(tmp_path / "ws"),
        ]
    )

    assert code == 0
    assert board.verdicts == []
    assert "nothing posted" in capsys.readouterr().out


def test_verifying_one_lap_posts_what_this_machine_measured(
    board, monkeypatch, fake_benchmark, tmp_path, capsys
):
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(
        verify_module, "clean_runtime_fingerprint", lambda: "aaaaaaaaaaaa"
    )

    code = main([SUBMISSION, "--url", board.url, "--workspace", str(tmp_path / "ws")])

    assert code == 0
    assert board.verdicts[-1]["id"] == SUBMISSION
    assert board.verdicts[-1]["lap_time_s"] == pytest.approx(15.2)
    assert "1 checked — 1 now on the board" in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════════
#  The token is the whole gate
# ═══════════════════════════════════════════════════════════════════════════


def test_the_whole_queue_is_verified_by_default(
    board, monkeypatch, fake_benchmark, tmp_path, capsys
):
    """No arguments is the normal way to run it: everything waiting, one by one."""
    board.queue = {
        "id-alpha": ("Alpha", 15.2, True),
        "id-bravo": ("Bravo", 16.4, True),
        "id-charlie": ("Charlie", 17.9, True),
    }
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(
        verify_module, "clean_runtime_fingerprint", lambda: "aaaaaaaaaaaa"
    )

    code = main(["--url", board.url, "--workspace", str(tmp_path / "ws")])

    assert code == 0
    assert [v["id"] for v in board.verdicts] == ["id-alpha", "id-bravo", "id-charlie"]
    assert "3 checked" in capsys.readouterr().out


def test_a_lap_with_no_policy_is_left_alone_and_counted(
    board, monkeypatch, fake_benchmark, tmp_path, capsys
):
    """It can never be reproduced, so it is reported rather than judged."""
    board.queue = {
        "id-alpha": ("Alpha", 15.2, True),
        "id-bravo": ("Bravo", 16.4, False),
    }
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(
        verify_module, "clean_runtime_fingerprint", lambda: "aaaaaaaaaaaa"
    )

    main(["--url", board.url, "--workspace", str(tmp_path / "ws")])
    printed = capsys.readouterr().out

    assert [v["id"] for v in board.verdicts] == ["id-alpha"]
    assert "1 lap(s) came with no policy" in printed


def test_one_broken_bundle_does_not_stop_the_queue(
    board, monkeypatch, fake_benchmark, tmp_path, capsys
):
    board.queue = {
        "id-alpha": ("Alpha", 15.2, True),
        "id-bravo": ("Bravo", 16.4, True),
    }
    board.claimed_sha = "0" * 64  # every download will fail its hash check
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(verify_module, "clean_runtime_fingerprint", lambda: None)

    code = main(["--url", board.url, "--workspace", str(tmp_path / "ws")])
    printed = capsys.readouterr().out

    assert code == 1
    assert board.verdicts == [], "nothing was judged on a bundle that did not arrive"
    assert printed.count("could not verify") == 2, "it tried both, not just the first"


def test_a_wrong_token_stops_before_anything_is_touched(
    board, monkeypatch, fake_benchmark, tmp_path, capsys
):
    """The script is public; without the organiser's token it does nothing."""
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", "a team's guess")

    code = main(["--url", board.url, "--workspace", str(tmp_path / "ws")])
    printed = capsys.readouterr().out

    assert code == 2
    assert board.verdicts == []
    assert "401" in printed
    assert "ADMIN_TOKEN" in printed
    assert "nothing here can run" in printed


def test_a_missing_token_never_reaches_the_board(board, monkeypatch, capsys):
    monkeypatch.delenv("LEADERBOARD_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    assert main(["--url", board.url]) == 2
    assert board.tokens == [], "no request was made at all"
    assert "LEADERBOARD_ADMIN_TOKEN" in capsys.readouterr().out


def test_the_connection_says_which_board_accepted_the_token(board, monkeypatch, capsys):
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)

    main(["--list", "--url", board.url])
    printed = capsys.readouterr().out

    assert "token accepted" in printed
    assert board.url in printed


def test_the_token_is_never_sent_in_clear(monkeypatch):
    """A mistyped board URL should not cost the organiser the token."""
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)

    with pytest.raises(VerificationError, match="in clear"):
        load_organiser("http://isaacleaderboard.netlify.app")

    assert load_organiser("https://isaacleaderboard.netlify.app").token == TOKEN


def test_a_local_board_over_http_is_fine(monkeypatch):
    """`netlify dev` and the tests serve plain http on loopback, and always will."""
    monkeypatch.setenv("LEADERBOARD_ADMIN_TOKEN", TOKEN)

    assert load_organiser("http://127.0.0.1:8888").token == TOKEN
    assert load_organiser("http://localhost:8888").token == TOKEN
