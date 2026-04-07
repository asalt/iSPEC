"""Command-line entry points for the internal supervisor loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time

from ispec.config.paths import resolve_supervisor_state_file
from ispec.logging import get_logger
from ispec.supervisor.loop import SupervisorConfig, _default_agent_id, run_supervisor
from ispec.supervisor.smoke import (
    enqueue_orchestrator_tick,
    generate_smoke_session_id,
    latest_assistant_message_id,
    seed_support_session_for_review,
    wait_for_support_session_review,
)

def _state_file_path() -> Path:
    resolved = resolve_supervisor_state_file()
    return Path(resolved.path or resolved.value)


def _read_state_payload(*, logger) -> dict[str, object] | None:
    path = _state_file_path()
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.debug("Unable to read supervisor state from %s: %s", path, exc)
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring corrupt supervisor state file %s: %s", path, exc)
        return None

    return parsed if isinstance(parsed, dict) else None


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it; treat as running.
        return True
    return True


def register_subcommands(subparsers) -> None:
    run_parser = subparsers.add_parser("run", help="Run the internal supervisor loop")
    run_parser.add_argument(
        "--agent-id",
        default=None,
        help="Agent identifier for emitted events (default: hostname)",
    )
    run_parser.add_argument(
        "--backend-base-url",
        default=None,
        help="Base URL for iSPEC backend (default: http://127.0.0.1:${ISPEC_PORT or 3001})",
    )
    run_parser.add_argument(
        "--frontend-url",
        default="http://127.0.0.1:3000/",
        help="Frontend URL to probe (default: http://127.0.0.1:3000/)",
    )
    run_parser.add_argument(
        "--interval-seconds",
        type=int,
        default=30,
        help="Seconds between steps (default: 30)",
    )
    run_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=2.0,
        help="HTTP timeout seconds (default: 2.0)",
    )
    run_parser.add_argument("--once", action="store_true", help="Run one step then exit")

    status_parser = subparsers.add_parser("status", help="Show supervisor state (pid/run_id)")
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON state payload instead of a human summary.",
    )

    stop_parser = subparsers.add_parser("stop", help="Stop the supervisor process using its PID")
    stop_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait for shutdown after signaling (default: 5.0)",
    )

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Seed a support session and wait for the supervisor review pipeline",
    )
    smoke_parser.add_argument(
        "--session-id",
        default=None,
        help="Support session id to use (default: auto-generated)",
    )
    smoke_parser.add_argument(
        "--no-seed-session",
        action="store_true",
        help="Do not insert synthetic messages (wait for an existing session)",
    )
    smoke_parser.add_argument(
        "--no-enqueue-tick",
        action="store_true",
        help="Do not enqueue an orchestrator tick (useful if one is already running)",
    )
    smoke_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Seconds to wait for an async review record (default: 60)",
    )
    smoke_parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval when waiting (default: 1)",
    )
    smoke_parser.add_argument(
        "--user-message",
        default="Smoke test: user message",
        help="Message content to seed as the user turn",
    )
    smoke_parser.add_argument(
        "--assistant-message",
        default="Smoke test: assistant reply",
        help="Message content to seed as the assistant turn",
    )
    smoke_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional debugging details",
    )


def dispatch(args) -> None:
    if args.subcommand not in {"run", "smoke", "status", "stop"}:
        raise SystemExit(f"Unknown supervisor subcommand: {args.subcommand}")

    logger = get_logger(__file__)

    if args.subcommand == "status":
        payload = _read_state_payload(logger=logger)
        if payload is None:
            logger.info("Supervisor is not running (no state file at %s).", _state_file_path())
            return
        if getattr(args, "json", False):
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return

        pid = payload.get("pid")
        pid_int: int | None = None
        if isinstance(pid, int) and pid > 0:
            pid_int = int(pid)
        elif isinstance(pid, str) and pid.strip().isdigit():
            pid_int = int(pid.strip())
        run_id = payload.get("run_id")
        agent_id = payload.get("agent_id")
        status = payload.get("status")
        started_at = payload.get("started_at")
        ended_at = payload.get("ended_at")

        running = bool(pid_int is not None and _pid_is_running(pid_int))
        logger.info(
            "Supervisor status=%s running=%s pid=%s run_id=%s agent_id=%s started_at=%s ended_at=%s",
            status,
            running,
            pid_int,
            run_id,
            agent_id,
            started_at,
            ended_at,
        )
        return

    if args.subcommand == "stop":
        payload = _read_state_payload(logger=logger)
        if payload is None:
            logger.info("Supervisor is not running.")
            return
        pid = payload.get("pid")
        pid_int: int | None = None
        if isinstance(pid, int) and pid > 0:
            pid_int = int(pid)
        elif isinstance(pid, str) and pid.strip().isdigit():
            pid_int = int(pid.strip())
        if pid_int is None:
            logger.warning("Supervisor state file did not include a pid; cannot stop.")
            return
        if not _pid_is_running(pid_int):
            logger.info("Supervisor process is already gone (pid=%s).", pid_int)
            return

        logger.info("Stopping supervisor pid=%s ...", pid_int)
        try:
            os.kill(pid_int, signal.SIGTERM)
        except ProcessLookupError:
            logger.info("Supervisor process is already gone (pid=%s).", pid_int)
            return
        except PermissionError as exc:
            logger.error("Permission denied signaling pid=%s: %s", pid_int, exc)
            raise SystemExit(2) from exc

        deadline = time.monotonic() + float(getattr(args, "timeout_seconds", 5.0) or 0.0)
        while time.monotonic() < deadline:
            if not _pid_is_running(pid_int):
                logger.info("Supervisor stopped.")
                return
            time.sleep(0.2)

        logger.warning("Timed out waiting for supervisor shutdown (pid=%s).", pid_int)
        return

    if args.subcommand == "smoke":
        _run_smoke(args)
        return

    agent_id = (args.agent_id or "").strip() or _default_agent_id()

    backend_base_url = (args.backend_base_url or "").strip()
    if not backend_base_url:
        port = (os.getenv("ISPEC_PORT") or "").strip() or "3001"
        backend_base_url = f"http://127.0.0.1:{port}"

    config = SupervisorConfig(
        agent_id=agent_id,
        backend_base_url=backend_base_url,
        frontend_url=str(args.frontend_url),
        interval_seconds=int(args.interval_seconds),
        timeout_seconds=float(args.timeout_seconds),
    )
    run_id = run_supervisor(config, once=bool(args.once))
    logger.info("Supervisor run_id=%s", run_id)


def _run_smoke(args) -> None:
    logger = get_logger(__file__)

    session_id = (args.session_id or "").strip()
    if not session_id:
        session_id = generate_smoke_session_id()

    seeded = None
    if not bool(args.no_seed_session):
        seeded = seed_support_session_for_review(
            session_id=session_id,
            user_message=str(args.user_message),
            assistant_message=str(args.assistant_message),
        )
        target_message_id = int(seeded.assistant_message_id)
        if args.verbose:
            logger.info(
                "Seeded support session session_id=%s session_pk=%s user_message_id=%s assistant_message_id=%s",
                seeded.session_id,
                seeded.session_pk,
                seeded.user_message_id,
                seeded.assistant_message_id,
            )
    else:
        target_message_id = latest_assistant_message_id(session_id=session_id) or 0
        if target_message_id <= 0:
            raise SystemExit(
                f"Support session {session_id!r} does not exist or has no assistant messages."
            )

    tick_id = None
    if not bool(args.no_enqueue_tick):
        tick_id = enqueue_orchestrator_tick(
            payload={"source": "cli_smoke", "session_id": session_id},
            priority=10,
            allow_existing=True,
        )
        if args.verbose:
            logger.info("Enqueued orchestrator tick command_id=%s", tick_id)

    result = wait_for_support_session_review(
        session_id=session_id,
        target_message_id=int(target_message_id),
        timeout_seconds=float(args.timeout_seconds),
        poll_seconds=float(args.poll_seconds),
    )

    payload = {
        "ok": bool(result.ok),
        "session_id": result.session_id,
        "target_message_id": int(result.target_message_id),
        "review_id": result.review_id,
        "elapsed_seconds": result.elapsed_seconds,
        "tick_command_id": tick_id,
        "seeded": bool(seeded is not None),
        "error": result.error,
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not result.ok:
        raise SystemExit(2)
