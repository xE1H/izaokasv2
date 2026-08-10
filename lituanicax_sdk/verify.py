"""The organiser's side of the board: re-run a team's lap, and say what it did.

Every lap on the leaderboard is published with the policy that set it (see
:mod:`lituanicax_sdk.bundle`). This is what turns one of those claims into a
result: it downloads the bundle, unpacks it into a workspace, drives the team's
own checkpoint through **this** checkout of the SDK, and posts the lap it got.
The board compares that against what the team claimed and turns the row green if
they agree.

    export LEADERBOARD_ADMIN_TOKEN="…"

    python -m lituanicax_sdk.verify                # the whole queue, one by one
    python -m lituanicax_sdk.verify --list         # what is waiting, run nothing
    python -m lituanicax_sdk.verify <id>           # just this one
    python -m lituanicax_sdk.verify <id> --dry-run # re-run it, post nothing

With no arguments it does the whole job: every lap waiting, downloaded, re-run
and marked, in one sitting. That is the normal way to use it — the queue is the
unit of work, not the submission.

**Only the organiser can run this, and this file being public changes nothing.**
Every call it makes — listing the queue, downloading a bundle, posting a verdict
— is an ``/api/admin/`` endpoint behind ``ADMIN_TOKEN``, compared in constant
time on the server. Without the token the board answers 401 to all three, so a
team reading this module can read exactly how verification works and still not
touch a single row: the first request fails and there is no second one. There is
no local verdict to forge either, because the verdict is not reached here — this
posts *the lap this machine measured*, and the server decides what that means.

The other half of the gate is the SDK itself: the team's fingerprint claim is
not consulted anywhere below, and the code that drives the lap is the checkout
this module is running from. The organiser's disk is the reference.

**This runs code a competitor wrote.** ``teamcode/`` is imported into the
process that scores the lap — it has to be, since it defines the observations
the checkpoint expects — and importing it runs whatever is in it. Two things
follow. The benchmark is run in a *subprocess* with the workspace as its working
directory, so the extracted ``teamcode/`` shadows this repository's rather than
replacing it, and this repository is never written to. And the run reports its
:func:`~lituanicax_sdk._locked.runtime_fingerprint`, which is compared against a
clean process here: a team that patches the SDK's crash rules in memory at
import time leaves the source fingerprint alone and moves that one. Neither is a
sandbox. Verify on a machine you are willing to hand to a stranger, and read
``teamcode/`` before you run it if the lap is remarkable.

What comes back is not a verdict, it is a measurement — the lap this machine
produced. The board applies the tolerance and decides, in one place, by the same
rule for everyone.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .bundle import BundleError, extract_bundle
from .submit import DEFAULT_LEADERBOARD_URL

#: Where a downloaded bundle is unpacked and run, unless ``--workspace`` says
#: otherwise. Under the project, gitignored, so a failed verification leaves
#: something to look at rather than a deleted temp directory.
WORKSPACE_DIRNAME = ".verify"

#: Long enough to download a few megabytes over a hotel connection.
TIMEOUT_S = 120.0

#: A benchmark run is ten cars and a lap each; on a cold GPU it is minutes, not
#: seconds. Past this something has gone wrong and waiting longer will not fix
#: it.
RUN_TIMEOUT_S = 1800.0

#: How long a finished run's output is given to drain before whatever is still
#: holding the pipe is treated as a leftover rather than as the run. Generous
#: for a pipe that only has to flush; nothing to a run that has already ended.
PIPE_DRAIN_S = 10.0

#: How long a run gets to shut down *after* it has written its report. The
#: benchmark gives its own teardown 60s before force-exiting, so this is that
#: plus room for the force-exit to happen — past it, the shutdown is wedged in
#: a way the run itself cannot escape, and the queue stops waiting.
SHUTDOWN_GRACE_S = 90.0


class VerificationError(RuntimeError):
    """Anything that stops one lap being checked. Never stops the others."""


@dataclass(frozen=True)
class Organiser:
    """Where the board is, and the token that proves who is asking."""

    url: str
    token: str

    def endpoint(self, path: str) -> str:
        return self.url.rstrip("/") + path


@dataclass(frozen=True)
class Rerun:
    """What this machine got out of a team's policy.

    ``sdk_before`` and ``sdk_after`` are the SDK's own critical code, hashed in
    the run's process before the team's code was imported and again once the
    driving was done. They are produced by *this* machine's SDK during *this*
    run, so nothing in the bundle can influence them.
    """

    lap_time_s: float | None
    report: dict
    workspace: Path
    sdk_before: str | None = None
    sdk_after: str | None = None

    @property
    def drove_a_lap(self) -> bool:
        return self.lap_time_s is not None

    @property
    def sdk_was_replaced(self) -> bool:
        """True when the team's code changed the SDK's behaviour as it ran."""
        if self.sdk_before is None or self.sdk_after is None:
            return False
        return self.sdk_before != self.sdk_after


# ═══════════════════════════════════════════════════════════════════════════
#  Talking to the board
# ═══════════════════════════════════════════════════════════════════════════


def load_organiser(url: str | None = None, token: str | None = None) -> Organiser:
    """Resolve the board's URL and the admin token.

    Raises:
        VerificationError: if the token is missing, or if sending it to the
            given URL would put it on the wire in clear. Nothing here works
            without the token, and nothing here is worth leaking it for.
    """
    resolved_url = (
        url or os.environ.get("LITUANICAX_LEADERBOARD_URL") or DEFAULT_LEADERBOARD_URL
    )
    resolved_token = (
        token
        or os.environ.get("LEADERBOARD_ADMIN_TOKEN")
        or os.environ.get("ADMIN_TOKEN")
        or ""
    )
    if not resolved_token.strip():
        raise VerificationError(
            "no admin token. Set LEADERBOARD_ADMIN_TOKEN, or pass --token.\n"
            "Verifying is the organiser's job and needs the board's ADMIN_TOKEN."
        )

    organiser = Organiser(url=str(resolved_url), token=resolved_token.strip())
    _refuse_plaintext(organiser.url)
    return organiser


def _refuse_plaintext(url: str) -> None:
    """Do not put the admin token on an unencrypted connection.

    The token is the only thing standing between the public and the board's
    verdicts, and it goes in a header on every call below. Over ``http://`` to
    anywhere but this machine, that is one shared network away from being
    somebody else's token — and a mistyped ``LITUANICAX_LEADERBOARD_URL`` is a
    far likelier way to lose it than anyone attacking the site. Loopback is
    allowed, because that is the local board the tests and `netlify dev` run.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme == "https":
        return
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return
    raise VerificationError(
        f"refusing to send the admin token to {url} in clear.\n"
        "Use https:// — the token is the whole of the board's security, and "
        "this would hand it to anything between here and there."
    )


def _call(
    organiser: Organiser, path: str, *, method: str = "GET", body: dict | None = None
) -> dict:
    request = urllib.request.Request(
        organiser.endpoint(path),
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Admin-Token": organiser.token,
            "User-Agent": "lituanicax-sdk-verify/1.0.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("message") or detail
        except json.JSONDecodeError:
            pass
        raise VerificationError(f"HTTP {exc.code} from {path}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise VerificationError(f"could not reach {organiser.url}: {exc}") from exc


def pending(organiser: Organiser) -> list[dict]:
    """Every lap waiting to be checked, oldest first."""
    answer = _call(organiser, "/api/admin/submissions?status=pending_verification")
    return list(answer.get("submissions") or [])


def download_bundle(organiser: Organiser, submission_id: str) -> bytes:
    """Fetch a policy bundle, part by part, and check it against its manifest.

    The digest is checked here rather than trusted from the board: the board
    stored whatever bytes it was given and comparing them against a hash it was
    given at the same time only proves it agrees with itself.
    """
    import hashlib

    manifest = _call(organiser, f"/api/admin/artifact?id={submission_id}")
    bundle = manifest.get("bundle") or {}
    parts = int(bundle.get("parts") or 0)
    if not parts:
        raise VerificationError(
            "this lap was published without a policy — nothing to run."
        )

    chunks = []
    for part in range(parts):
        answer = _call(organiser, f"/api/admin/artifact?id={submission_id}&part={part}")
        chunks.append(base64.b64decode(answer["data"]))
        print(f"    part {part + 1}/{parts}", end="\r", flush=True)

    data = b"".join(chunks)
    digest = hashlib.sha256(data).hexdigest()
    if digest != bundle.get("sha256") or len(data) != bundle.get("bytes"):
        raise VerificationError(
            f"the bundle does not match its manifest: {digest}, {len(data)} bytes."
        )
    return data


def post_verdict(
    organiser: Organiser, submission_id: str, rerun: Rerun, note: str = ""
) -> dict:
    """Tell the board what this machine measured, and let it judge.

    A run that produced no lap posts a rejection outright: there is no number to
    compare, and "no lap" is a fact about the policy rather than a tolerance
    question.
    """
    body: dict = {"id": submission_id, "note": note}
    if rerun.drove_a_lap:
        body["lap_time_s"] = rerun.lap_time_s
    else:
        body["state"] = "rejected"
    return _call(organiser, "/api/admin/verify", method="POST", body=body)


# ═══════════════════════════════════════════════════════════════════════════
#  Re-running one lap
# ═══════════════════════════════════════════════════════════════════════════


def workspace_for(submission_id: str, root: Path | None = None) -> Path:
    from .runs import project_root

    base = Path(root) if root else project_root() / WORKSPACE_DIRNAME
    return base / submission_id


def rerun_bundle(data: bytes, workspace: Path, *, headless: bool = True) -> Rerun:
    """Unpack a bundle and drive its policy through this SDK.

    The benchmark runs as a subprocess whose working directory is the
    workspace, so ``import teamcode`` finds the *extracted* one — ``sys.path[0]``
    is the working directory, and it wins over this repository, which is on
    ``PYTHONPATH`` behind it and supplies the SDK. That ordering is the whole
    trick: the team's code, this SDK, and neither able to reach the other's
    directory.

    Returns:
        What the run produced, whether or not that was a lap.

    Raises:
        VerificationError: if the bundle is unusable or the benchmark could not
            be started at all. A benchmark that ran and produced no lap is a
            result, not an error.
    """
    from .runs import project_root

    if workspace.exists():
        shutil.rmtree(workspace)
    manifest = extract_bundle(data, workspace)

    checkpoint = _checkpoint_in(workspace, manifest)
    out = workspace / "verified.json"
    repo = project_root()

    command = [
        sys.executable,
        "-u",
        "-m",
        "lituanicax_sdk.benchmark",
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out),
        "--no-submit",
        f"--agents={manifest['reproduce'].get('attempts') or 10}",
        f"--seed={manifest['reproduce'].get('seed') or 0}",
    ]
    jitter = manifest["reproduce"].get("spawn_jitter_deg")
    if jitter is not None:
        command.append(f"--spawn-jitter={jitter}")
    if headless:
        command.append("--headless")

    environment = {
        **os.environ,
        # The SDK comes from here; `teamcode` comes from the workspace, which
        # the working directory puts ahead of this on the path.
        "PYTHONPATH": os.pathsep.join(
            [str(repo), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        # A verification run must not publish anything of its own, whatever the
        # organiser's own shell happens to be configured for.
        "LITUANICAX_TEAM": "",
    }

    print(f"    running {' '.join(command[3:])}")
    status, output = run_benchmark(
        command,
        cwd=workspace,
        env=environment,
        log=workspace / "benchmark.log",
        # The report is the run's real finishing line. What happens after it is
        # the simulator shutting down, which is not verification's problem.
        done_when=out,
    )

    if not out.is_file():
        tail = "\n".join(output.strip().splitlines()[-15:])
        raise VerificationError(
            f"the benchmark wrote no report (exit {status}).\nLast lines:\n{tail}"
        )

    report = json.loads(out.read_text())
    return Rerun(
        lap_time_s=report.get("best_lap_time_s"),
        report=report,
        workspace=workspace,
        sdk_before=report.get("runtime_fingerprint_at_import"),
        sdk_after=report.get("runtime_fingerprint"),
    )


def run_benchmark(
    command: list[str],
    *,
    cwd: Path,
    env: dict,
    log: Path,
    timeout: float = RUN_TIMEOUT_S,
    done_when: Path | None = None,
    shutdown_grace: float = SHUTDOWN_GRACE_S,
) -> tuple[int, str]:
    """Run one benchmark to completion, showing what it is doing.

        Three things here are load-bearing, and all three were learned the hard way.

    **The run is over when the report is written, not when the process exits,
        and certainly not when its output ends.** Isaac Sim's teardown does not
        reliably return, and when it hangs it hangs holding the GIL, so the
        benchmark's own escape hatch cannot always fire either. Meanwhile the lap
        has been driven, scored and written to disk. Waiting on the process — let
        alone on end-of-output, which is what ``subprocess.run`` does and which a
        leftover holding the pipe can defer for ever — turns a finished measurement
        into a hang with a blank terminal and an idle GPU. So ``done_when`` is
        watched, the process is given a grace period after it appears to shut down
        tidily, and then it is stopped.

        **The output is streamed, not captured.** A verification run is minutes of
        simulator start-up; an organiser watching a blank terminal cannot tell that
        from a hang. Every line goes to the screen as it arrives, and to
        ``benchmark.log`` for afterwards.

        **Nothing can ask a question.** stdin is closed, so a prompt fails the run
        loudly instead of blocking it silently and invisibly.

        Returns:
            The exit status and everything the run printed.
    """
    try:
        process = subprocess.Popen(  # noqa: S603 — the command is built above
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # The simulator's output is not reliably UTF-8 — a stray byte from a
            # driver or an extension is enough — and a decode error would kill
            # the reader, taking the rest of the run's output with it.
            errors="replace",
            bufsize=1,
            # Its own process group, so any straggler can be cleaned up without
            # signalling the organiser's own shell.
            start_new_session=True,
        )
    except (OSError, FileNotFoundError) as exc:
        raise VerificationError(f"could not start the benchmark: {exc}") from exc

    lines: list[str] = []

    def pump() -> None:
        for line in process.stdout:  # type: ignore[union-attr]
            lines.append(line)
            print(f"    │ {line.rstrip()}", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    started = time.monotonic()
    status, stopped = _await_end(
        process, timeout=timeout, done_when=done_when, shutdown_grace=shutdown_grace
    )
    if status is None:
        _write_log(log, lines)
        raise VerificationError(
            f"the benchmark did not finish within {timeout / 60:.0f} minutes."
        )

    # The process is gone. Give its output a moment to drain, then stop caring:
    # anything still holding the pipe is a leftover of the simulator's, and it
    # is not a reason to hold up the queue.
    reader.join(timeout=PIPE_DRAIN_S)
    if reader.is_alive():
        print("    │ (the simulator left something behind holding the output)")
        _stop(process)

    _write_log(log, lines)
    how = "stopped by us after it scored the lap" if stopped else f"exit {status}"
    print(f"    finished in {time.monotonic() - started:.0f}s ({how})")
    return status, "".join(lines)


def _await_end(
    process: subprocess.Popen,
    *,
    timeout: float,
    done_when: Path | None,
    shutdown_grace: float,
) -> tuple[int | None, bool]:
    """Wait for the run to be over. Returns ``(status, did we stop it)``.

    Over means one of two things: the process exited, or it wrote its report and
    then spent longer than ``shutdown_grace`` failing to exit. The second is the
    simulator's teardown deadlock — the lap is already scored and on disk, and
    nothing is gained by keeping the queue waiting for a process that is only
    going to hold the GPU.

    ``status`` is None when the whole run ran out of time, which is a different
    thing entirely: nothing was measured.
    """
    started = time.monotonic()
    reported_at: float | None = None

    while True:
        try:
            return process.wait(timeout=1.0), False
        except subprocess.TimeoutExpired:
            pass

        now = time.monotonic()
        if reported_at is None and done_when is not None and done_when.is_file():
            reported_at = now
            print("    │ (the lap is scored and written; letting it shut down)")

        if reported_at is not None and now - reported_at > shutdown_grace:
            print(
                f"    the simulator has not shut down {shutdown_grace:.0f}s after "
                "the lap was scored, so this stops it. The result is unaffected."
            )
            _stop(process)
            return process.poll(), True

        if now - started > timeout:
            _stop(process)
            return None, True


def _stop(process: subprocess.Popen) -> None:
    """End the run's whole process group, politely and then not."""
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _write_log(log: Path, lines: list[str]) -> None:
    try:
        log.write_text("".join(lines))
    except OSError:  # pragma: no cover — a full or read-only disk
        pass


def _checkpoint_in(workspace: Path, manifest: dict) -> Path:
    name = (manifest.get("checkpoint") or {}).get("name")
    candidate = workspace / "checkpoint" / str(name)
    if candidate.is_file():
        return candidate
    found = sorted((workspace / "checkpoint").glob("*.pt"))
    if not found:
        raise BundleError("the bundle contains no checkpoint.")
    return found[0]


# ═══════════════════════════════════════════════════════════════════════════
#  The command
# ═══════════════════════════════════════════════════════════════════════════


def verify_one(
    organiser: Organiser,
    submission: dict,
    *,
    dry_run: bool = False,
    headless: bool = True,
    workspace_root: Path | None = None,
) -> str:
    """Check one lap end to end.

    Returns what became of it — ``"verified"``, ``"rejected"``, or ``"held"``
    for a dry run, which reaches a measurement and deliberately stops there.
    """
    submission_id = submission["id"]
    claimed = submission["lap_time_s"]
    print(f"\n── {submission['team']}   claimed {claimed:.3f} s   {submission_id}")

    data = download_bundle(organiser, submission_id)
    workspace = workspace_for(submission_id, workspace_root)
    rerun = rerun_bundle(data, workspace, headless=headless)

    if not rerun.drove_a_lap:
        print("    the policy completed no lap here")
    else:
        drift = rerun.lap_time_s - claimed
        print(f"    this machine got {rerun.lap_time_s:.3f} s ({drift:+.3f})")

    note = _tamper_note(rerun)
    if note:
        print(f"    {note}")

    if dry_run:
        print("    --dry-run: nothing posted")
        return "held"

    answer = post_verdict(organiser, submission_id, rerun, note=note)
    print(f"    {answer.get('message', '')}")
    return "verified" if answer.get("ranked") else "rejected"


def _tamper_note(rerun: Rerun) -> str:
    """One line about whether the team's code left the SDK alone.

    The run hashes the SDK's critical code before importing any of the team's,
    and again once the lap is driven. A difference means something replaced the
    crash rules, the clocks or a locked constant while the lap was being set —
    which no file hash can see, since no file changed.

    It does not by itself reject the lap; the board decides that on the time.
    But it goes in the note, because a lap set under a rewritten SDK is one to
    look at whatever number came out of it.
    """
    if rerun.sdk_before is None or rerun.sdk_after is None:
        return (
            "note: this run did not report what the SDK hashed to; it is an old bundle."
        )
    if not rerun.sdk_was_replaced:
        return ""
    return (
        f"WARNING: this team's code changed the SDK's behaviour as it ran "
        f"({rerun.sdk_before} before importing teamcode, {rerun.sdk_after} after "
        "the lap) — read teamcode/ before accepting this lap."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lituanicax_sdk.verify",
        description=(
            "Re-run every published lap that is waiting, and tell the leaderboard "
            "what each one did. Needs the board's admin token."
        ),
    )
    parser.add_argument(
        "submission",
        nargs="?",
        help="Verify only this submission id (default: everything waiting).",
    )
    parser.add_argument(
        "--list", action="store_true", help="Show what is waiting, run nothing."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify everything waiting. This is the default; the flag is for "
        "saying so out loud.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Re-run, but post no verdict."
    )
    parser.add_argument("--url", default=None, help="Board to talk to.")
    parser.add_argument(
        "--token", default=None, help="Admin token (or LEADERBOARD_ADMIN_TOKEN)."
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help=f"Where to unpack bundles (default: {WORKSPACE_DIRNAME}/).",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Show the simulator while re-running, for watching a suspicious lap.",
    )
    args = parser.parse_args(argv)

    try:
        organiser = load_organiser(args.url, args.token)
    except VerificationError as exc:
        print(f"[verify] {exc}")
        return 2

    # The first call is the door. Everything below it — the queue, the bundles,
    # the verdicts — is behind the same admin token, so a wrong one stops here
    # rather than halfway through a session.
    try:
        waiting = pending(organiser)
    except VerificationError as exc:
        print(f"[verify] {exc}")
        if "401" in str(exc) or "503" in str(exc):
            print(
                "[verify] the board would not accept that token, so nothing here "
                "can run. Only the organiser's ADMIN_TOKEN opens it."
            )
        return 2

    print(f"[verify] {organiser.url} — token accepted, {len(waiting)} lap(s) waiting")

    if args.list:
        return _print_queue(waiting, listed=True)

    if args.submission:
        chosen = [s for s in waiting if s["id"] == args.submission]
        if not chosen:
            print(f"[verify] {args.submission} is not waiting to be verified.")
            return 1
    else:
        # The default: the whole queue, one at a time. Laps published without a
        # policy are not in it — there is nothing to re-run — but they are
        # counted below rather than passed over in silence.
        chosen = [s for s in waiting if (s.get("verification") or {}).get("bundle")]

    orphans = len(waiting) - len(
        [s for s in waiting if (s.get("verification") or {}).get("bundle")]
    )
    if not chosen:
        print("[verify] nothing to re-run.")
        if orphans:
            print(f"[verify] {orphans} lap(s) came with no policy and never can be.")
        return 0

    root = Path(args.workspace) if args.workspace else None
    outcomes: list[str] = []
    failed = 0
    for index, submission in enumerate(chosen, start=1):
        print(f"\n[verify] {index}/{len(chosen)}", end="")
        try:
            outcomes.append(
                verify_one(
                    organiser,
                    submission,
                    dry_run=args.dry_run,
                    headless=not args.windowed,
                    workspace_root=root,
                )
            )
        except (VerificationError, BundleError) as exc:
            # One team's broken bundle must not stop the rest of the queue.
            failed += 1
            print(f"    could not verify: {exc}")

    ranked = outcomes.count("verified")
    rejected = outcomes.count("rejected")
    print(
        f"\n[verify] {len(chosen)} checked — {ranked} now on the board, "
        f"{rejected} rejected, {failed} could not be run"
    )
    if orphans:
        print(f"[verify] {orphans} lap(s) came with no policy and were left alone.")
    return 0 if not failed else 1


def _print_queue(waiting: list[dict], *, listed: bool) -> int:
    if not waiting:
        print("[verify] nothing is waiting to be verified.")
        return 0

    print(f"[verify] {len(waiting)} lap(s) waiting:\n")
    for submission in waiting:
        bundle = (submission.get("verification") or {}).get("bundle")
        policy = (
            f"{bundle['bytes'] / 1_048_576:.1f} MB"
            if bundle
            else "no policy — cannot be verified"
        )
        print(
            f"  {submission['id']}  {submission['team'][:24]:24}"
            f"  {submission['lap_time_s']:8.3f} s  {policy}"
        )
    if not listed:
        print("\nPass an id to verify one, or --all for everything with a policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
