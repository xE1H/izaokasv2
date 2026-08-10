"""Publishing a benchmark result to the official leaderboard.

``python -m lituanicax_sdk.benchmark`` calls :func:`submit` with the report it
just built and the policy that set the lap. Two things go up, in this order:

1. **the lap** — the team, the time and the SDK fingerprint. The board stores
   it, shows it greyed out, and counts it for nothing;
2. **the policy** — :mod:`lituanicax_sdk.bundle`'s zip of the checkpoint, your
   ``teamcode/`` and the run's config, uploaded in parts against the submission
   id the board just handed back.

The lap turns green and takes a position when the organisers download that
bundle, re-run it against a clean official SDK on their own hardware, and get
the same time back. Until then the row says *awaiting verification*, which is
exactly what it is.

**Why the fingerprint is not enough on its own.** It is computed by your copy of
the SDK, and your copy of the SDK is a directory you can edit. A team that
changes ``_locked.py`` can report any twelve characters it likes, so a board
gated on that number is gated on the honesty of the thing it is checking. It
still travels with every lap and still gates whether a lap is worth checking at
all — it is just no longer the last word. Reproduction is.

All the board needs is your team name, from the environment or from a
``.lituanicax.json`` next to ``logs/``::

    export LITUANICAX_TEAM="Wingless Wonders"

    # or, once, in .lituanicax.json
    {"team": "Wingless Wonders"}

There is no password, and the name is taken at face value. Nothing is gained by
using someone else's: a lap slower than theirs does not move them, and a lap
faster than theirs is one you want your own name on — and a lap posted under
their name comes with *your* policy attached, which is what the organisers will
run. The name is a label; the bundle is the claim.

Publishing is best-effort and deliberately impossible to fail a run with: a
laptop with no network still scores a lap, it just prints that it could not
send it. The upload is best-effort in the same way and for the same reason — a
bundle that does not go costs the team verification, not the run. Nothing here
raises, and nothing here retries: a benchmark you can rerun is cheaper than a
client that hangs.

The URL only needs setting if you are running your own board::

    export LITUANICAX_LEADERBOARD_URL="https://example.netlify.app"
"""

from __future__ import annotations

import base64
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

#: Uploading a policy is megabytes rather than bytes, and a team on a hotel
#: connection should not lose verification to the timeout that suits a JSON
#: request. Still bounded: a stalled upload gives up rather than hanging.
UPLOAD_TIMEOUT_S = 120.0

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

    @property
    def artifact_endpoint(self) -> str:
        return self.url.rstrip("/") + "/api/artifacts"


@dataclass(frozen=True)
class SubmissionOutcome:
    """What happened, in a form :func:`print_outcome` can render.

    Attributes:
        sent: the request reached the board and it answered 2xx.
        ranked: the board is counting this lap. Never true for a lap that has
            just been published — it turns true when the organisers reproduce
            it, which is a later event this run cannot see.
        message: one line, already written for a human.
        response: the board's decoded JSON, when there was one.
        verification: the board's word for where the lap stands, normally
            ``"pending"``. None when nothing was stored.
        uploaded: the policy bundle reached the board.
        upload_message: one line about the bundle, whether or not it went.
    """

    sent: bool
    ranked: bool
    message: str
    response: dict | None = None
    verification: str | None = None
    uploaded: bool = False
    upload_message: str = ""


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


def _post(
    submitter: Submitter,
    payload: dict,
    *,
    endpoint: str | None = None,
    timeout: float = TIMEOUT_S,
) -> dict:
    """POST the payload and return the decoded response.

    Raises:
        SubmissionError: for anything that is not a 2xx with a JSON body.
    """
    request = urllib.request.Request(
        endpoint or submitter.endpoint,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"lituanicax-sdk/{CLIENT_VERSION}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as exc:
        # The board explains itself in the body even when it refuses, and that
        # explanation is the useful half of the error.
        raise SubmissionError(_http_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise SubmissionError(f"could not reach {submitter.url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise SubmissionError(
            f"{submitter.url} did not answer within {timeout:g}s"
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


def upload_bundle(
    submitter: Submitter, submission_id: str, bundle, limits: dict
) -> str:
    """Send a policy bundle up, in parts, and close the upload.

    A part is base64 in a JSON body, so the part size is chosen against what the
    board says it will take rather than against the file — the board knows its
    own request limit and says so in the submission response.

    Raises:
        SubmissionError: on the first part that does not go, so a half-uploaded
            bundle is never completed into a lap that looks checkable.
    """
    part_bytes = int(limits.get("max_part_bytes") or 3 * 1024 * 1024)
    max_bytes = int(limits.get("max_bytes") or 0)
    if max_bytes and bundle.bytes > max_bytes:
        raise SubmissionError(
            f"the policy bundle is {bundle.megabytes:.1f} MB and the board takes "
            f"{max_bytes / 1_048_576:.0f} MB."
        )

    data = bundle.data
    parts = max(1, -(-len(data) // part_bytes))
    endpoint = submitter.artifact_endpoint

    for index in range(parts):
        chunk = data[index * part_bytes : (index + 1) * part_bytes]
        _post(
            submitter,
            {
                "submission_id": submission_id,
                "part": index,
                "parts": parts,
                "data": base64.b64encode(chunk).decode(),
            },
            endpoint=endpoint,
            timeout=UPLOAD_TIMEOUT_S,
        )

    done = _post(
        submitter,
        {
            "complete": True,
            "submission_id": submission_id,
            "sha256": bundle.sha256,
            "bytes": bundle.bytes,
            "parts": parts,
            "filename": bundle.filename,
        },
        endpoint=endpoint,
        timeout=UPLOAD_TIMEOUT_S,
    )
    return str(
        done.get("message")
        or f"policy uploaded ({bundle.megabytes:.1f} MB in {parts} parts)."
    )


def submit(report: dict, *, team: str | None = None, bundle=None) -> SubmissionOutcome:
    """Publish a benchmark report, and the policy that produced it. Never raises.

    Args:
        report: the dict :mod:`lituanicax_sdk.benchmark` builds and writes to
            ``submission.json``.
        team: overrides the configured team name.
        bundle: a :class:`~lituanicax_sdk.bundle.Bundle`. Without one the lap is
            published as a claim nobody will ever be able to check, and the
            board says so on the row.

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
    verification = response.get("verification")
    message = response.get("message") or (
        "on the leaderboard." if ranked else "recorded, but not ranked."
    )

    uploaded, upload_message = _send_bundle(submitter, response, bundle)
    return SubmissionOutcome(
        sent=True,
        ranked=ranked,
        message=str(message),
        response=response,
        verification=str(verification) if verification else None,
        uploaded=uploaded,
        upload_message=upload_message,
    )


def _send_bundle(submitter: Submitter, response: dict, bundle) -> tuple[bool, str]:
    """Upload the policy, if there is one to upload and the board wants it.

    Separate from :func:`submit` because its failures are separate: the lap is
    already stored by the time this runs, and losing the upload costs the team
    verification, not the run. Like everything else here, it reports rather than
    raises.
    """
    limits = response.get("upload")
    submission_id = response.get("submission_id")

    if bundle is None:
        return False, "no policy was sent, so this lap cannot be verified."
    if not submission_id or not isinstance(limits, dict):
        # An ineligible lap gets no upload slot: the board will not verify a run
        # from an SDK that is not the official one, so it does not want the zip.
        return False, "the board is not accepting a policy for this lap."

    try:
        return True, upload_bundle(submitter, str(submission_id), bundle, limits)
    except SubmissionError as exc:
        return False, f"the policy did not upload — {exc}"


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════


def print_outcome(outcome: SubmissionOutcome, *, url: str | None = None) -> None:
    """Say what the board did with the lap, in the benchmark's own voice.

    A lap that was accepted is described from the board's structured fields
    rather than its prose, so the line is the same shape every time and the
    board is free to change its wording. The prose is the fallback, and the
    whole of what a refusal says.
    """
    if not outcome.sent:
        print(f"[submit] not published — {outcome.message}")
        return

    response = outcome.response or {}

    if outcome.ranked:
        # Only reachable from a board that ranks on arrival, which the official
        # one no longer does. The board's own wording is used rather than the
        # lap that was just sent: the two are the same thing only when this lap
        # is the one the board is showing, and "15.367 s" against a board
        # holding 15.200 is a lie in the client's voice.
        rank = response.get("rank")
        print(f"[submit] {f'P{rank}   ' if rank is not None else ''}{outcome.message}")
    elif outcome.verification == "pending":
        # The lap is stored and on the board, greyed out. What it is playing for
        # is the provisional place; what stands between the two is the
        # organisers' re-run, which is the next line.
        print(f"[submit] {_claim(response)}")
        print(f"[submit] {outcome.upload_message}")
        print(
            "[submit] it counts once the organisers reproduce it on their own machine."
        )
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


def _claim(response: dict) -> str:
    """``15.200 s   awaiting verification   would be P3``."""
    parts = []
    lap = response.get("lap_time_s")
    if lap is not None:
        parts.append(f"{float(lap):.3f} s")
    parts.append("awaiting verification")
    rank = response.get("provisional_rank")
    if rank is not None:
        parts.append(f"would be P{rank}")
    return "   ".join(parts)
