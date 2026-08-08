"""The leaderboard client, exercised against a real HTTP server.

Everything here runs on a loopback socket in milliseconds — no Isaac Sim, no
GPU and no network. The server is a stand-in for the Netlify functions, and it
records what it was sent, so the tests can assert on the wire format that the
website is written against rather than on the client's internals.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lituanicax_sdk import submit as submit_module
from lituanicax_sdk._locked import sdk_fingerprint
from lituanicax_sdk.submit import (
    SubmissionError,
    Submitter,
    build_payload,
    load_submitter,
    print_outcome,
    submit,
)

FINGERPRINT = sdk_fingerprint()


# ═══════════════════════════════════════════════════════════════════════════
#  A stand-in leaderboard
# ═══════════════════════════════════════════════════════════════════════════


class FakeBoard:
    """An HTTP server that answers /api/submissions and remembers the request."""

    def __init__(self, status=201, body=None, official=FINGERPRINT):
        self.status = status
        self.body = body
        self.official = official
        self.requests: list[dict] = []
        self._server = HTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _handler(board):  # noqa: N805 — the closure needs the board, not `self`
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler's spelling
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                board.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "payload": payload,
                    }
                )
                body = board.body
                if body is None:
                    body = board._default_response(payload)
                self._respond(board.status, body)

            def _respond(self, status, body):
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass  # keep pytest's output clean

        return Handler

    def _default_response(self, payload: dict) -> dict:
        """The contract's own answer: rank it only if the SDK is the official one."""
        ranked = payload.get("sdk_fingerprint") == self.official and not payload.get(
            "sdk_modified"
        )
        return {
            "accepted": True,
            "status": "ranked" if ranked else "sdk_mismatch",
            "ranked": ranked,
            "team": payload.get("team"),
            "lap_time_s": payload.get("lap_time_s"),
            "rank": 1 if ranked else None,
            "personal_best": ranked,
            "official_sdk_fingerprint": self.official,
            "leaderboard_url": self.url + "/",
            "message": "P1 — a new best." if ranked else "not the official SDK.",
        }


@pytest.fixture
def board():
    server = FakeBoard()
    yield server
    server.close()


@pytest.fixture
def report():
    """What lituanicax_sdk.benchmark writes to submission.json."""
    return {
        "best_lap_time_s": 42.137,
        "attempts": 10,
        "laps_completed": 7,
        "median_lap_time_s": 45.0,
        "slowest_lap_time_s": 51.2,
        "track": "official",
        "track_length_m": 61.4,
        "seed": 0,
        "checkpoint": "logs/2026-08-04_10-00-00/model_1500.pt",
        "sdk_fingerprint": FINGERPRINT,
        "sdk_modified": [],
    }


@pytest.fixture
def configured(monkeypatch, board, tmp_path):
    """A named team, and no stray .lituanicax.json in the way."""
    monkeypatch.setenv("LITUANICAX_TEAM", "Wingless Wonders")
    monkeypatch.setenv("LITUANICAX_LEADERBOARD_URL", board.url)
    monkeypatch.setattr(submit_module, "_config_file", lambda: tmp_path / "absent.json")
    return board


# ═══════════════════════════════════════════════════════════════════════════
#  Who is submitting, and where to
# ═══════════════════════════════════════════════════════════════════════════


def test_the_team_name_comes_from_the_environment(configured, board):
    submitter = load_submitter()
    assert submitter.team == "Wingless Wonders"
    assert submitter.endpoint == board.url + "/api/submissions"


def test_the_team_name_can_come_from_the_config_file(monkeypatch, tmp_path):
    monkeypatch.delenv("LITUANICAX_TEAM", raising=False)
    monkeypatch.delenv("LITUANICAX_LEADERBOARD_URL", raising=False)
    config = tmp_path / ".lituanicax.json"
    config.write_text(json.dumps({"team": "Filed", "url": "https://board.example"}))
    monkeypatch.setattr(submit_module, "_config_file", lambda: config)

    submitter = load_submitter()
    assert submitter.team == "Filed"
    assert submitter.endpoint == "https://board.example/api/submissions"


def test_the_environment_wins_over_the_config_file(monkeypatch, tmp_path):
    config = tmp_path / ".lituanicax.json"
    config.write_text(json.dumps({"team": "Filed"}))
    monkeypatch.setattr(submit_module, "_config_file", lambda: config)
    monkeypatch.setenv("LITUANICAX_TEAM", "Env")

    assert load_submitter().team == "Env"


def test_an_explicit_team_wins_over_both(configured):
    assert load_submitter("Overridden").team == "Overridden"


def test_the_default_url_is_the_official_board(monkeypatch, tmp_path):
    monkeypatch.delenv("LITUANICAX_LEADERBOARD_URL", raising=False)
    monkeypatch.setenv("LITUANICAX_TEAM", "T")
    monkeypatch.setattr(submit_module, "_config_file", lambda: tmp_path / "absent.json")

    assert load_submitter().url == submit_module.DEFAULT_LEADERBOARD_URL


def test_a_missing_team_name_explains_what_to_set(monkeypatch, tmp_path):
    monkeypatch.delenv("LITUANICAX_TEAM", raising=False)
    monkeypatch.setattr(submit_module, "_config_file", lambda: tmp_path / "absent.json")

    with pytest.raises(SubmissionError, match="LITUANICAX_TEAM"):
        load_submitter()


def test_a_blank_team_name_is_no_team_name(monkeypatch, tmp_path):
    monkeypatch.setenv("LITUANICAX_TEAM", "   ")
    monkeypatch.setattr(submit_module, "_config_file", lambda: tmp_path / "absent.json")

    with pytest.raises(SubmissionError, match="LITUANICAX_TEAM"):
        load_submitter()


def test_a_padded_team_name_is_trimmed(monkeypatch, tmp_path):
    """So " Slipstream " and "Slipstream" are one team on the board, not two."""
    monkeypatch.setenv("LITUANICAX_TEAM", "  Slipstream  ")
    monkeypatch.setattr(submit_module, "_config_file", lambda: tmp_path / "absent.json")

    assert load_submitter().team == "Slipstream"


def test_an_unreadable_config_file_is_reported_not_ignored(monkeypatch, tmp_path):
    config = tmp_path / ".lituanicax.json"
    config.write_text("{ not json")
    monkeypatch.setattr(submit_module, "_config_file", lambda: config)

    with pytest.raises(SubmissionError):
        load_submitter()


# ═══════════════════════════════════════════════════════════════════════════
#  The payload
# ═══════════════════════════════════════════════════════════════════════════


def test_the_payload_carries_the_lap_the_team_and_the_fingerprint(report):
    payload = build_payload(report, "Wingless Wonders")
    assert payload["team"] == "Wingless Wonders"
    assert payload["lap_time_s"] == pytest.approx(42.137)
    assert payload["sdk_fingerprint"] == FINGERPRINT
    assert payload["client_version"] == submit_module.CLIENT_VERSION


def test_the_fingerprint_is_recomputed_not_copied_from_the_report(report):
    """A report claiming a different SDK cannot launder a modified one."""
    report["sdk_fingerprint"] = "0" * 12
    assert build_payload(report, "T")["sdk_fingerprint"] == FINGERPRINT


def test_a_modified_sdk_is_declared(report):
    report["sdk_modified"] = ["env.py"]
    assert build_payload(report, "T")["sdk_modified"] == ["env.py"]


def test_a_run_with_no_lap_has_nothing_to_submit(report):
    report["best_lap_time_s"] = None
    with pytest.raises(SubmissionError, match="no lap"):
        build_payload(report, "T")


# ═══════════════════════════════════════════════════════════════════════════
#  Submitting
# ═══════════════════════════════════════════════════════════════════════════


def test_a_valid_lap_is_sent_and_ranked(configured, report, board):
    outcome = submit(report)

    assert outcome.sent and outcome.ranked
    assert board.requests[0]["path"] == "/api/submissions"
    sent = board.requests[0]["payload"]
    assert sent["lap_time_s"] == pytest.approx(42.137)
    assert sent["sdk_fingerprint"] == FINGERPRINT


def test_the_team_name_is_the_whole_of_the_identity(configured, report, board):
    """There is no password; the name in the body is what the board goes on."""
    submit(report)
    request = board.requests[0]

    assert request["payload"]["team"] == "Wingless Wonders"
    assert not [name for name in request["headers"] if "token" in name.lower()]
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["headers"]["User-Agent"].startswith("lituanicax-sdk/")


def test_the_context_the_board_shows_is_included(configured, report, board):
    submit(report)
    sent = board.requests[0]["payload"]

    assert sent["attempts"] == 10
    assert sent["laps_completed"] == 7
    assert sent["track"] == "official"
    assert sent["seed"] == 0


def test_a_modified_sdk_is_sent_but_not_ranked(configured, report, board):
    report["sdk_modified"] = ["vehicle.py"]
    outcome = submit(report)

    assert outcome.sent and not outcome.ranked
    assert outcome.response["status"] == "sdk_mismatch"
    assert outcome.response["rank"] is None


def test_a_stale_sdk_is_not_ranked(monkeypatch, configured, report, board):
    """The board is racing a newer SDK than this run's."""
    board.official = "f" * 12
    outcome = submit(report)

    assert outcome.sent and not outcome.ranked
    assert outcome.response["official_sdk_fingerprint"] == "f" * 12


def test_a_refusal_does_not_raise(monkeypatch, tmp_path, report):
    board = FakeBoard(status=400, body={"accepted": False, "error": "team is required"})
    try:
        monkeypatch.setenv("LITUANICAX_TEAM", "T")
        monkeypatch.setenv("LITUANICAX_LEADERBOARD_URL", board.url)
        monkeypatch.setattr(
            submit_module, "_config_file", lambda: tmp_path / "absent.json"
        )

        outcome = submit(report)
    finally:
        board.close()

    assert not outcome.sent and not outcome.ranked
    assert "team is required" in outcome.message


def test_an_unreachable_board_does_not_raise(monkeypatch, tmp_path, report):
    monkeypatch.setenv("LITUANICAX_TEAM", "T")
    # Port 1 is reserved and nothing listens on it.
    monkeypatch.setenv("LITUANICAX_LEADERBOARD_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(submit_module, "_config_file", lambda: tmp_path / "absent.json")

    outcome = submit(report)

    assert not outcome.sent
    assert "127.0.0.1:1" in outcome.message


def test_missing_credentials_do_not_raise_either(monkeypatch, tmp_path, report):
    monkeypatch.delenv("LITUANICAX_TEAM", raising=False)
    monkeypatch.setattr(submit_module, "_config_file", lambda: tmp_path / "absent.json")

    outcome = submit(report)

    assert not outcome.sent
    assert "LITUANICAX_TEAM" in outcome.message


def test_a_run_with_no_lap_reports_rather_than_raises(configured, report, board):
    report["best_lap_time_s"] = None
    outcome = submit(report)

    assert not outcome.sent
    assert board.requests == []


def test_nonsense_from_the_board_is_survivable(monkeypatch, tmp_path, report):
    board = FakeBoard()
    try:
        monkeypatch.setenv("LITUANICAX_TEAM", "T")
        monkeypatch.setenv("LITUANICAX_LEADERBOARD_URL", board.url)
        monkeypatch.setattr(
            submit_module, "_config_file", lambda: tmp_path / "absent.json"
        )
        monkeypatch.setattr(
            submit_module,
            "_post",
            lambda *a, **k: (_ for _ in ()).throw(SubmissionError("not JSON")),
        )

        outcome = submit(report)
    finally:
        board.close()

    assert not outcome.sent


# ═══════════════════════════════════════════════════════════════════════════
#  What the team sees
# ═══════════════════════════════════════════════════════════════════════════


def test_a_ranked_lap_prints_its_place(configured, report, capsys):
    print_outcome(submit(report))
    printed = capsys.readouterr().out

    assert "P1" in printed
    assert "42.137 s" in printed


def test_an_unranked_lap_prints_both_fingerprints(configured, report, capsys):
    report["sdk_modified"] = ["env.py"]
    print_outcome(submit(report))
    printed = capsys.readouterr().out

    assert FINGERPRINT in printed
    assert "official" in printed


def test_a_failure_says_it_was_not_published(monkeypatch, tmp_path, report, capsys):
    monkeypatch.delenv("LITUANICAX_TEAM", raising=False)
    monkeypatch.setattr(submit_module, "_config_file", lambda: tmp_path / "absent.json")

    print_outcome(submit(report))

    assert "not published" in capsys.readouterr().out


def test_the_endpoint_is_built_without_a_double_slash():
    assert (
        Submitter(team="T", url="https://board.example/").endpoint
        == "https://board.example/api/submissions"
    )


def test_a_lap_that_did_not_improve_says_what_the_board_holds(
    configured, report, board, capsys
):
    """`lap_time_s` is the lap just sent, not the one the board is showing."""
    board.body = {
        "accepted": True,
        "status": "ranked",
        "ranked": True,
        "rank": 1,
        "lap_time_s": 15.367,
        "personal_best": False,
        "message": "Lap 15.367s recorded. Your best stays 15.200s (P1).",
    }
    print_outcome(submit(report))
    printed = capsys.readouterr().out

    assert "Your best stays 15.200s" in printed
    assert "15.367 s" not in printed, "printing the slower lap as the placing misleads"
