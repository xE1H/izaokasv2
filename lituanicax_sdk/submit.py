"""Publishing a benchmark result to the official leaderboard.

``python -m lituanicax_sdk.benchmark`` calls :func:`submit` with the report it
just built. The leaderboard stores three things that matter — the team, the lap
time and the SDK fingerprint — and ranks the submission only if that
fingerprint is the official one. A team running a modified SDK still gets a
row in the audit log; it just does not get a place on the board, which is the
same rule :func:`~lituanicax_sdk._locked.verify_integrity` states locally.

All the board needs is your team name, from the environment or from a
``.lituanicax.json`` next to ``logs/``::

    export LITUANICAX_TEAM="Wingless Wonders"

    # or, once, in .lituanicax.json
    {"team": "Wingless Wonders"}

There is no password, and the name is taken at face value. Nothing is gained by
using someone else's: a lap slower than theirs does not move them, and a lap
faster than theirs is one you want your own name on. What the board does check
is the SDK, which is the only claim worth checking.

Publishing is best-effort and deliberately impossible to fail a run with: a
laptop with no network still scores a lap, it just prints that it could not
send it. Nothing here raises, and nothing here retries — a benchmark you can
rerun is cheaper than a client that hangs.

The URL only needs setting if you are running your own board::

    export LITUANICAX_LEADERBOARD_URL="https://example.netlify.app"
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ._locked import sdk_fingerprint

#: Where results go unless ``LITUANICAX_LEADERBOARD_URL`` says otherwise.
DEFAULT_LEADERBOARD_URL = "https://isaacleaderboard.netlify.app"

#: The file, next to ``logs/``, that can hold the team name instead of the
#: environment. Gitignored, so a fork does not carry one team's name to the
#: next.
CONFIG_FILENAME = ".lituanicax.json"

#: Long enough for a cold serverless function to wake up, short enough that a
#: dead network costs you ten seconds and not a session.
TIMEOUT_S = 10.0

#: Sent as the User-Agent, so the board can tell old clients apart.
CLIENT_VERSION = "1.0.0"


class SubmissionError(RuntimeError):
    """Raised inside this module and caught at its edge; never escapes.

    :func:`submit` reports failures as a :class:`SubmissionOutcome` rather than
    by raising, because a lap that has already been driven should not be lost
    to a DNS hiccup.
    """


@dataclass(frozen=True)
class Submitter:
    """Who is submitting, and where to."""

    team: str
    url: str

    @property
    def endpoint(self) -> str:
        return self.url.rstrip("/") + "/api/submissions"


@dataclass(frozen=True)
class SubmissionOutcome:
    """What happened, in a form :func:`print_outcome` can render.

    Attributes:
        sent: the request reached the board and it answered 2xx.
        ranked: the board put this lap on the leaderboard. False when the SDK
            fingerprint is not the official one.
        message: one line, already written for a human.
        response: the board's decoded JSON, when there was one.
    """

    sent: bool
    ranked: bool
    message: str
    response: dict | None = None


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════


def _config_file() -> Path:
    from .runs import project_root

    return project_root() / CONFIG_FILENAME


def _read_config_file() -> dict:
    path = _config_file()
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"{CONFIG_FILENAME} could not be read: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SubmissionError(f"{CONFIG_FILENAME} should hold a JSON object.")
    return loaded


def load_submitter(team: str | None = None) -> Submitter:
    """Resolve the team name and the board's URL.

    Args:
        team: overrides both sources, for ``benchmark --team``.

    Raises:
        SubmissionError: if there is no team name. The caller turns that into a
            printed hint, not a crash.
    """
    config = _read_config_file()

    resolved = team or os.environ.get("LITUANICAX_TEAM") or config.get("team")
    url = (
        os.environ.get("LITUANICAX_LEADERBOARD_URL")
        or config.get("url")
        or DEFAULT_LEADERBOARD_URL
    )

    if not resolved or not str(resolved).strip():
        raise SubmissionError(
            "no team name. Set LITUANICAX_TEAM, pass --team, or write "
            f"{CONFIG_FILENAME} in the project root:\n"
            '    {"team": "Your Team"}'
        )

    return Submitter(team=str(resolved).strip(), url=str(url))


# ═══════════════════════════════════════════════════════════════════════════
#  Submitting
# ═══════════════════════════════════════════════════════════════════════════


def build_payload(report: dict, team: str) -> dict:
    """The submission itself, taken from the benchmark's own report.

    The fingerprint is recomputed here rather than copied from the report, so
    that what is published is what this SDK hashes to at the moment of
    publishing — one fewer place for the two to disagree.
    """
    best = report.get("best_lap_time_s")
    if best is None:
        raise SubmissionError("no lap was completed, so there is nothing to submit.")

    return {
        "team": team,
        "lap_time_s": float(best),
        "sdk_fingerprint": sdk_fingerprint(),
        "sdk_modified": list(report.get("sdk_modified") or []),
        "attempts": report.get("attempts"),
        "laps_completed": report.get("laps_completed"),
        "track": report.get("track"),
        "track_length_m": report.get("track_length_m"),
        "seed": report.get("seed"),
        "checkpoint": report.get("checkpoint"),
        "client_version": CLIENT_VERSION,
    }


def _post(submitter: Submitter, payload: dict) -> dict:
    """POST the payload and return the decoded response.

    Raises:
        SubmissionError: for anything that is not a 2xx with a JSON body.
    """
    request = urllib.request.Request(
        submitter.endpoint,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"lituanicax-sdk/{CLIENT_VERSION}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as exc:
        # The board explains itself in the body even when it refuses, and that
        # explanation is the useful half of the error.
        raise SubmissionError(_http_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise SubmissionError(f"could not reach {submitter.url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise SubmissionError(
            f"{submitter.url} did not answer within {TIMEOUT_S:g}s"
        ) from exc
    except OSError as exc:
        raise SubmissionError(f"could not reach {submitter.url}: {exc}") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SubmissionError(
            f"the board sent something that is not JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise SubmissionError("the board sent JSON that is not an object.")
    return decoded


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    """Prefer the board's own wording; fall back to the status line."""
    try:
        detail = json.loads(exc.read().decode())
        described = detail.get("message") or detail.get("error")
    except Exception:  # noqa: BLE001 — any unreadable body falls back below
        described = None

    return f"HTTP {exc.code}: {described or exc.reason}"


def submit(report: dict, *, team: str | None = None) -> SubmissionOutcome:
    """Publish a benchmark report. Never raises.

    Args:
        report: the dict :mod:`lituanicax_sdk.benchmark` builds and writes to
            ``submission.json``.
        team: overrides the configured team name.

    Returns:
        What happened, for :func:`print_outcome`.
    """
    try:
        submitter = load_submitter(team)
        payload = build_payload(report, submitter.team)
        response = _post(submitter, payload)
    except SubmissionError as exc:
        return SubmissionOutcome(sent=False, ranked=False, message=str(exc))

    ranked = bool(response.get("ranked"))
    message = response.get("message") or (
        "on the leaderboard." if ranked else "recorded, but not ranked."
    )
    return SubmissionOutcome(
        sent=True, ranked=ranked, message=str(message), response=response
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════


def print_outcome(outcome: SubmissionOutcome, *, url: str | None = None) -> None:
    """Say what the board did with the lap, in the benchmark's own voice.

    A ranked lap is described from the board's structured fields rather than
    its prose, so the line is the same shape every time and the board is free
    to change its wording. The prose is the fallback, and the whole of what a
    refusal says.
    """
    if not outcome.sent:
        print(f"[submit] not published — {outcome.message}")
        return

    response = outcome.response or {}

    if outcome.ranked:
        print(f"[submit] {_placing(response) or outcome.message}")
    else:
        print(f"[submit] {outcome.message}")
        official = response.get("official_sdk_fingerprint")
        this_run = f"this run's sdk is {sdk_fingerprint()}"
        print(
            f"[submit] {this_run}, the official one is {official}"
            if official
            else f"[submit] {this_run}"
        )

    board = response.get("leaderboard_url") or url
    if board:
        print(f"[submit] {board}")


def _placing(response: dict) -> str:
    """``P3   15.200 s   a personal best``, from whichever fields came back."""
    rank, best = response.get("rank"), response.get("lap_time_s")
    parts = []
    if rank is not None:
        parts.append(f"P{rank}")
    if best is not None:
        parts.append(f"{float(best):.3f} s")
    if response.get("personal_best"):
        parts.append("a personal best")
    return "   ".join(parts)
