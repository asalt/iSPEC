"""Windows-friendly agent helpers (v0).

This CLI is intentionally lightweight: it collects simple metrics locally and
POSTs structured events to an iSPEC server over HTTP.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import os
import random
import shutil
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.agent.relay import enqueue_relay_request, relay_config_probe


def register_subcommands(subparsers) -> None:
    disk_parser = subparsers.add_parser(
        "disk-free",
        help="Post disk free metrics on an interval",
    )
    disk_parser.add_argument(
        "--server",
        required=True,
        help="Base URL of the iSPEC server (e.g. http://10.16.1.10:8000)",
    )
    disk_parser.add_argument("--agent-id", required=True, help="Unique id for this machine")
    disk_parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Path to measure (repeatable). Defaults to C:\\\\ when omitted.",
    )
    disk_parser.add_argument(
        "--interval-seconds",
        type=int,
        default=3600,
        help="Seconds between samples (default: 3600)",
    )
    disk_parser.add_argument(
        "--once",
        action="store_true",
        help="Send one sample and exit",
    )
    disk_parser.add_argument(
        "--api-key",
        default=None,
        help="Optional iSPEC API key (sent as X-API-Key)",
    )
    disk_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=10,
        help="HTTP timeout seconds (default: 10)",
    )

    watch_parser = subparsers.add_parser(
        "watch",
        help="Survey project folders and tail tackle.log with backoff",
    )
    watch_parser.add_argument(
        "--server",
        required=True,
        help="Base URL of the iSPEC server (e.g. http://10.16.1.11:3001)",
    )
    watch_parser.add_argument("--agent-id", required=True, help="Unique id for this machine")
    watch_parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root directory to survey (repeatable).",
    )
    watch_parser.add_argument(
        "--project-dir-glob",
        action="append",
        default=[],
        help="Glob(s) for project folders inside each root (repeatable; default: MSPC*).",
    )
    watch_parser.add_argument(
        "--tackle-relpath",
        action="append",
        default=[],
        help="Relative path(s) to look for within each project dir (repeatable; default: tackle.log and config/tackle.log).",
    )
    watch_parser.add_argument(
        "--state-db",
        default=None,
        help="SQLite state DB path (default: ~/.ispec-agent/state.db).",
    )
    watch_parser.add_argument(
        "--policy-file",
        default=None,
        help="Optional JSON policy overrides (backoff coefficients, limits, etc).",
    )
    watch_parser.add_argument(
        "--api-key",
        default=None,
        help="Optional iSPEC API key (sent as X-API-Key).",
    )
    watch_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=10,
        help="HTTP timeout seconds (default: 10).",
    )
    watch_parser.add_argument(
        "--once",
        action="store_true",
        help="Run one watch tick and exit.",
    )

    list_parser = subparsers.add_parser(
        "queue-list",
        help="List local supervisor/agent command queue rows",
    )
    list_parser.add_argument("--database", default=None, help="Agent DB path/URI (default: resolved env/default)")
    list_parser.add_argument("--status", action="append", default=[], help="Status to include (repeatable)")
    list_parser.add_argument("--command-type", action="append", default=[], help="Command type to include (repeatable)")
    list_parser.add_argument("--limit", type=int, default=50, help="Max rows to return (default: 50)")
    list_parser.add_argument("--json", action="store_true", help="Print JSON instead of a plain table")

    cancel_parser = subparsers.add_parser(
        "queue-cancel-stale",
        help="Mark stale queued/running command rows as failed without processing them",
    )
    cancel_parser.add_argument("--database", default=None, help="Agent DB path/URI (default: resolved env/default)")
    cancel_parser.add_argument(
        "--older-than-hours",
        type=float,
        default=24.0,
        help="Only cancel commands whose updated_at is older than this many hours (default: 24)",
    )
    cancel_parser.add_argument("--status", action="append", default=[], help="Status to include (default: queued,running)")
    cancel_parser.add_argument("--command-type", action="append", default=[], help="Command type to include (repeatable)")
    cancel_parser.add_argument("--limit", type=int, default=200, help="Max rows to cancel/list (default: 200)")
    cancel_parser.add_argument(
        "--reason",
        default="manual stale queue cleanup",
        help="Reason recorded in command error/result_json",
    )
    cancel_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually mark matching rows failed; without this, only dry-run candidates are printed.",
    )
    cancel_parser.add_argument("--json", action="store_true", help="Print JSON instead of a plain table")

    task_parser = subparsers.add_parser(
        "watch-task",
        help="Sleep once, inspect durable state for a long-running task, and optionally wake an allowlisted tmux pane",
    )
    task_parser.add_argument("--pid", type=int, default=None, help="PID to probe with kill(pid, 0).")
    task_parser.add_argument("--log-file", default=None, help="Optional log file to stat.")
    task_parser.add_argument("--output", action="append", default=[], help="Expected output file to stat; repeatable.")
    task_parser.add_argument("--sentinel-file", default=None, help="Optional sentinel file whose existence means done.")
    task_parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=180,
        help="Initial sleep before inspection; clamped to 0..300 seconds.",
    )
    task_parser.add_argument("--wake-tmux-target", default=None, help="Optional exact allowlisted tmux target to wake.")
    task_parser.add_argument(
        "--wake-message",
        default="Long-running task appears complete.",
        help="Message to send if the task appears complete.",
    )
    task_parser.add_argument(
        "--press-enter",
        action="store_true",
        help="Send Enter/C-m after the wake message. Only use for panes intended to receive submitted text.",
    )
    task_parser.add_argument("--json", action="store_true", help="Print JSON instead of a plain summary")

    probe_parser = subparsers.add_parser(
        "relay-probe",
        help="Inspect local relay Slack config/provenance without sending anything",
    )
    probe_parser.add_argument("--target", default=None, help="Optional Slack destination alias to resolve.")
    probe_parser.add_argument("--message-type", default=None, help="Optional message type for allowlist checks.")
    probe_parser.add_argument("--json", action="store_true", help="Print JSON instead of a plain summary")

    relay_parser = subparsers.add_parser(
        "relay-enqueue",
        help="Enqueue a local relay request; staged by default and processed by the supervisor",
    )
    relay_parser.add_argument("--database", default=None, help="Agent DB path/URI (default: resolved env/default)")
    relay_parser.add_argument(
        "--kind",
        required=True,
        choices=["slack_message", "tmux_send", "status_record"],
        help="Relay request kind.",
    )
    relay_parser.add_argument("--source", default="codex", help="Source id/kind label for provenance.")
    relay_parser.add_argument("--to", "--target", dest="target", default=None, help="Destination alias or tmux target.")
    relay_parser.add_argument("--body", "--text", dest="body", default=None, help="Message body or status text.")
    relay_parser.add_argument("--attachment", action="append", default=[], help="Attachment path to stage; repeatable.")
    relay_parser.add_argument("--mode", choices=["stage", "send"], default="stage", help="Default: stage only.")
    relay_parser.add_argument("--confirm", action="store_true", help="Required for live send mode.")
    relay_parser.add_argument("--message-type", default=None, help="Optional Slack message type for allowlist checks.")
    relay_parser.add_argument("--thread-ts", default=None, help="Optional Slack thread timestamp.")
    relay_parser.add_argument("--press-enter", action="store_true", help="For tmux_send, include Enter if live send is enabled.")
    relay_parser.add_argument("--metadata-json", default=None, help="Optional JSON object copied into request metadata.")
    relay_parser.add_argument("--provenance-json", default=None, help="Optional JSON object copied into request provenance.")
    relay_parser.add_argument("--priority", type=int, default=0)
    relay_parser.add_argument("--delay-seconds", type=int, default=0)
    relay_parser.add_argument("--max-attempts", type=int, default=1)
    relay_parser.add_argument("--json", action="store_true", help="Print JSON instead of a plain summary")

    pdf_parser = subparsers.add_parser(
        "relay-pdf",
        help="Stage or send a PDF report to a Slack destination with report-ready defaults",
    )
    pdf_parser.add_argument("file", help="PDF file path to attach.")
    pdf_parser.add_argument("message", nargs="*", help="Optional message body; defaults to a short report-ready note.")
    pdf_parser.add_argument("--database", default=None, help="Agent DB path/URI (default: resolved env/default)")
    pdf_parser.add_argument("--to", required=True, help="Slack destination alias, e.g. alex.")
    pdf_parser.add_argument(
        "--send",
        action="store_true",
        help="Queue a live send request. This sets mode=send and confirm=true on the relay request.",
    )
    pdf_parser.add_argument("--body", "--text", dest="body", default=None, help="Message body override.")
    pdf_parser.add_argument("--title", default=None, help="Optional Slack file title.")
    pdf_parser.add_argument(
        "--source",
        default="cli:relay-pdf",
        help="Source kind/id for relay policy, e.g. cli:relay-pdf or codex:936-report.",
    )
    pdf_parser.add_argument("--message-type", default="report_ready", help="Slack destination message type.")
    pdf_parser.add_argument("--thread-ts", default=None, help="Optional Slack thread timestamp.")
    pdf_parser.add_argument("--metadata-json", default=None, help="Optional JSON object copied into request metadata.")
    pdf_parser.add_argument("--provenance-json", default=None, help="Optional JSON object copied into request provenance.")
    pdf_parser.add_argument("--priority", type=int, default=0)
    pdf_parser.add_argument("--delay-seconds", type=int, default=0)
    pdf_parser.add_argument("--max-attempts", type=int, default=1)
    pdf_parser.add_argument("--json", action="store_true", help="Print JSON instead of a plain summary")


def dispatch(args) -> None:
    if args.subcommand == "disk-free":
        _run_disk_free(args)
        return
    if args.subcommand == "watch":
        _run_watch(args)
        return
    if args.subcommand == "queue-list":
        _run_queue_list(args)
        return
    if args.subcommand == "queue-cancel-stale":
        _run_queue_cancel_stale(args)
        return
    if args.subcommand == "watch-task":
        _run_watch_task(args)
        return
    if args.subcommand == "relay-probe":
        _run_relay_probe(args)
        return
    if args.subcommand == "relay-enqueue":
        _run_relay_enqueue(args)
        return
    if args.subcommand == "relay-pdf":
        _run_relay_pdf(args)
        return
    raise SystemExit(f"Unknown agent subcommand: {args.subcommand}")


def _normalize_filter_values(values: list[str] | None, *, default: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        for item in str(value or "").split(","):
            item = item.strip()
            if item and item not in normalized:
                normalized.append(item)
    return normalized or list(default)


def _agent_command_row(row: AgentCommand) -> dict[str, Any]:
    payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    schedule = payload.get("schedule") if isinstance(payload.get("schedule"), dict) else {}
    return {
        "id": int(row.id),
        "command_type": row.command_type,
        "status": row.status,
        "attempts": int(row.attempts or 0),
        "max_attempts": int(row.max_attempts or 0),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "available_at": row.available_at.isoformat() if row.available_at else None,
        "job": str(job.get("name") or "").strip() or None,
        "schedule": str(schedule.get("name") or "").strip() or None,
        "error": row.error,
    }


def _query_agent_commands(
    *,
    database: str | None,
    statuses: list[str],
    command_types: list[str],
    limit: int,
    older_than_hours: float | None = None,
) -> list[dict[str, Any]]:
    limit = max(1, min(5000, int(limit or 50)))
    with get_agent_session(database) as db:
        query = db.query(AgentCommand)
        if statuses:
            query = query.filter(AgentCommand.status.in_(statuses))
        if command_types:
            query = query.filter(AgentCommand.command_type.in_(command_types))
        if older_than_hours is not None:
            cutoff = datetime.now(UTC) - timedelta(hours=max(0.0, float(older_than_hours)))
            query = query.filter(AgentCommand.updated_at <= cutoff)
        rows = query.order_by(AgentCommand.id.asc()).limit(limit).all()
        return [_agent_command_row(row) for row in rows]


def _print_command_rows(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    rows = payload.get("commands")
    if not isinstance(rows, list):
        rows = []
    print(
        f"ok={payload.get('ok')} dry_run={payload.get('dry_run')} "
        f"selected={payload.get('selected')} cancelled={payload.get('cancelled')}"
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        print(
            "\t".join(
                [
                    str(row.get("id") or ""),
                    str(row.get("status") or ""),
                    str(row.get("command_type") or ""),
                    str(row.get("updated_at") or ""),
                    str(row.get("job") or ""),
                    str(row.get("schedule") or ""),
                ]
            )
        )


def _load_cli_json_object(raw: str | None, *, label: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise SystemExit(f"Invalid {label}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"Invalid {label}: expected a JSON object.")
    return parsed


def _print_relay_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"ok={payload.get('ok')} command_id={payload.get('command_id')} status={payload.get('status')}")
    request = payload.get("relay_request")
    if isinstance(request, dict):
        print(
            " ".join(
                [
                    f"request_id={request.get('request_id')}",
                    f"kind={request.get('kind')}",
                    f"mode={request.get('mode')}",
                ]
            )
        )


def _source_from_cli_arg(raw: str | None) -> dict[str, Any]:
    text = str(raw or "").strip() or "cli"
    if ":" in text:
        kind, _, source_id = text.partition(":")
        kind = kind.strip()
        source_id = source_id.strip()
        if kind and source_id:
            source = {"kind": kind, "id": source_id}
        else:
            source = {"kind": "cli", "id": text}
    else:
        source = {"kind": "cli", "id": text}
    try:
        source["cwd"] = str(Path.cwd().resolve())
    except Exception:
        pass
    return source


def _enqueue_relay_cli_payload(
    *,
    database: str | None,
    request: dict[str, Any],
    priority: int,
    delay_seconds: int,
    max_attempts: int,
) -> dict[str, Any]:
    with get_agent_session(database) as db:
        row, event_payload = enqueue_relay_request(
            db,
            request=request,
            priority=int(priority or 0),
            delay_seconds=int(delay_seconds or 0),
            max_attempts=int(max_attempts or 1),
        )
        relay_request = event_payload.get("request") if isinstance(event_payload.get("request"), dict) else {}
        return {
            "ok": True,
            "command_id": int(row.id),
            "status": row.status,
            "relay_request": relay_request,
        }


def _run_relay_probe(args) -> None:
    payload = relay_config_probe(
        target_alias=getattr(args, "target", None),
        message_type=getattr(args, "message_type", None),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    slack = payload.get("slack") if isinstance(payload.get("slack"), dict) else {}
    token = slack.get("bot_token") if isinstance(slack.get("bot_token"), dict) else {}
    target = slack.get("target") if isinstance(slack.get("target"), dict) else None
    print(f"root={payload.get('root')}")
    print(f"env_files_loaded={len(payload.get('env_files_loaded') or [])}")
    print(f"slack_token_present={token.get('present')} source={token.get('source')}")
    print(f"destinations_count={slack.get('destinations_count')}")
    if target is not None:
        print(f"target={target}")
    elif slack.get("target_error"):
        print(f"target_error={slack.get('target_error')}")


def _run_relay_enqueue(args) -> None:
    metadata = _load_cli_json_object(getattr(args, "metadata_json", None), label="--metadata-json")
    provenance = _load_cli_json_object(getattr(args, "provenance_json", None), label="--provenance-json")
    source = _source_from_cli_arg(getattr(args, "source", None) or "codex")

    kind = str(getattr(args, "kind", "") or "").strip()
    target_raw = str(getattr(args, "target", "") or "").strip()
    if kind == "tmux_send":
        target: dict[str, Any] | str | None = {"target": target_raw} if target_raw else None
    else:
        target = {"alias": target_raw} if target_raw else None

    request = {
        "kind": kind,
        "source": source,
        "target": target,
        "body": getattr(args, "body", None),
        "attachments": [{"path": path} for path in (getattr(args, "attachment", None) or [])],
        "mode": getattr(args, "mode", "stage"),
        "confirm": bool(getattr(args, "confirm", False)),
        "message_type": getattr(args, "message_type", None),
        "thread_ts": getattr(args, "thread_ts", None),
        "press_enter": bool(getattr(args, "press_enter", False)),
        "metadata": metadata,
        "provenance": provenance,
    }

    try:
        payload = _enqueue_relay_cli_payload(
            database=getattr(args, "database", None),
            request=request,
            priority=int(getattr(args, "priority", 0) or 0),
            delay_seconds=int(getattr(args, "delay_seconds", 0) or 0),
            max_attempts=int(getattr(args, "max_attempts", 1) or 1),
        )
    except ValueError as exc:
        payload = {"ok": False, "error": str(exc)}
    _print_relay_payload(payload, as_json=bool(getattr(args, "json", False)))


def _run_relay_pdf(args) -> None:
    metadata = _load_cli_json_object(getattr(args, "metadata_json", None), label="--metadata-json")
    provenance = _load_cli_json_object(getattr(args, "provenance_json", None), label="--provenance-json")

    file_path = Path(str(getattr(args, "file", "") or "")).expanduser()
    try:
        file_path = file_path.resolve()
    except Exception:
        pass
    if not file_path.is_file():
        raise SystemExit(f"PDF file does not exist: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise SystemExit(f"relay-pdf only accepts PDF files: {file_path}")

    positional_message = " ".join(str(item) for item in (getattr(args, "message", None) or [])).strip()
    body = str(getattr(args, "body", "") or positional_message).strip()
    if not body:
        body = f"PDF report ready: {file_path.name}"

    metadata.setdefault("relay_shortcut", "relay-pdf")
    metadata.setdefault("attachment_kind", "pdf")
    metadata.setdefault("filename", file_path.name)
    provenance.setdefault("source_command", "ispec agent relay-pdf")

    mode = "send" if bool(getattr(args, "send", False)) else "stage"
    attachment: dict[str, Any] = {"path": str(file_path)}
    title = str(getattr(args, "title", "") or "").strip()
    if title:
        attachment["title"] = title
    request = {
        "kind": "slack_message",
        "source": _source_from_cli_arg(getattr(args, "source", None)),
        "target": {"alias": str(getattr(args, "to", "") or "").strip()},
        "body": body,
        "attachments": [attachment],
        "mode": mode,
        "confirm": mode == "send",
        "message_type": getattr(args, "message_type", None) or "report_ready",
        "thread_ts": getattr(args, "thread_ts", None),
        "metadata": metadata,
        "provenance": provenance,
    }
    try:
        payload = _enqueue_relay_cli_payload(
            database=getattr(args, "database", None),
            request=request,
            priority=int(getattr(args, "priority", 0) or 0),
            delay_seconds=int(getattr(args, "delay_seconds", 0) or 0),
            max_attempts=int(getattr(args, "max_attempts", 1) or 1),
        )
    except ValueError as exc:
        payload = {"ok": False, "error": str(exc)}
    _print_relay_payload(payload, as_json=bool(getattr(args, "json", False)))


def _run_queue_list(args) -> None:
    statuses = _normalize_filter_values(getattr(args, "status", None), default=["queued", "running"])
    command_types = _normalize_filter_values(getattr(args, "command_type", None), default=[])
    commands = _query_agent_commands(
        database=getattr(args, "database", None),
        statuses=statuses,
        command_types=command_types,
        limit=int(getattr(args, "limit", 50) or 50),
    )
    _print_command_rows(
        {"ok": True, "dry_run": True, "selected": len(commands), "cancelled": 0, "commands": commands},
        as_json=bool(getattr(args, "json", False)),
    )


def _run_queue_cancel_stale(args) -> None:
    statuses = _normalize_filter_values(getattr(args, "status", None), default=["queued", "running"])
    command_types = _normalize_filter_values(getattr(args, "command_type", None), default=[])
    older_than_hours = max(0.0, float(getattr(args, "older_than_hours", 24.0) or 24.0))
    limit = max(1, min(5000, int(getattr(args, "limit", 200) or 200)))
    database = getattr(args, "database", None)
    reason = str(getattr(args, "reason", "") or "manual stale queue cleanup").strip()
    confirm = bool(getattr(args, "confirm", False))

    candidates = _query_agent_commands(
        database=database,
        statuses=statuses,
        command_types=command_types,
        limit=limit,
        older_than_hours=older_than_hours,
    )

    cancelled = 0
    if confirm and candidates:
        candidate_ids = [int(row["id"]) for row in candidates if isinstance(row.get("id"), int)]
        now = datetime.now(UTC)
        with get_agent_session(database) as db:
            rows = (
                db.query(AgentCommand)
                .filter(AgentCommand.id.in_(candidate_ids))
                .filter(AgentCommand.status.in_(statuses))
                .all()
            )
            for row in rows:
                result_json = dict(row.result_json or {})
                result_json.update(
                    {
                        "cancelled": True,
                        "cancelled_at": now.isoformat(),
                        "cancel_reason": reason,
                        "previous_status": row.status,
                    }
                )
                row.status = "failed"
                row.error = f"cancelled_stale_queue: {reason}"
                row.ended_at = now
                row.updated_at = now
                row.result_json = result_json
                cancelled += 1

    _print_command_rows(
        {
            "ok": True,
            "dry_run": not confirm,
            "selected": len(candidates),
            "cancelled": cancelled,
            "older_than_hours": older_than_hours,
            "statuses": statuses,
            "command_types": command_types,
            "commands": candidates,
        },
        as_json=bool(getattr(args, "json", False)),
    )


def _run_watch_task(args) -> None:
    sleep_seconds = max(0, min(300, int(getattr(args, "sleep_seconds", 180) or 180)))
    if sleep_seconds:
        time.sleep(float(sleep_seconds))

    from ispec.agent.long_task import inspect_long_task

    state = inspect_long_task(
        pid=getattr(args, "pid", None),
        log_file=getattr(args, "log_file", None),
        output_paths=getattr(args, "output", None) or [],
        sentinel_file=getattr(args, "sentinel_file", None),
    )
    wake_target = str(getattr(args, "wake_tmux_target", "") or "").strip()
    wake_result: dict[str, Any] | None = None
    if state.get("should_wake") is True and wake_target:
        from ispec.assistant.tools import _tmux_send_text

        wake_result = _tmux_send_text(
            target=wake_target,
            text=str(getattr(args, "wake_message", "") or "Long-running task appears complete."),
            press_enter=bool(getattr(args, "press_enter", False)),
        )
    payload = {
        "ok": True,
        "slept_seconds": sleep_seconds,
        "state": state,
        "wake": wake_result,
    }
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    print(
        "ok=True "
        f"slept_seconds={sleep_seconds} "
        f"complete={state.get('complete')} "
        f"pid_alive={state.get('pid_alive')} "
        f"outputs_ready={state.get('outputs_ready')} "
        f"wake_sent={wake_result is not None}"
    )


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _post_events(
    *,
    server: str,
    events: list[dict[str, Any]],
    api_key: str | None,
    timeout_seconds: int,
) -> tuple[int, str]:
    url = server.rstrip("/") + "/api/agents/events"
    body = json.dumps(events).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ispec-agent/0",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def _run_disk_free(args) -> None:
    paths = args.path or ["C:\\"]
    interval_seconds = max(1, int(args.interval_seconds))

    while True:
        ts = _utcnow_iso()
        events: list[dict[str, Any]] = []
        for path in paths:
            usage = shutil.disk_usage(path)
            events.append(
                {
                    "type": "metric",
                    "agent_id": args.agent_id,
                    "ts": ts,
                    "name": "disk_free_bytes",
                    "dimensions": {"path": path},
                    "value": int(usage.free),
                }
            )

        try:
            status, payload = _post_events(
                server=args.server,
                events=events,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
            )
            print(f"[{ts}] POST {len(events)} events -> {status} {payload}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            print(f"[{ts}] HTTP error {exc.code}: {body}")
        except urllib.error.URLError as exc:
            print(f"[{ts}] Connection error: {exc}")
        except Exception as exc:
            print(f"[{ts}] Unexpected error: {exc}")

        if args.once:
            return
        time.sleep(interval_seconds)


@dataclass
class BackoffPolicy:
    """Idle backoff schedule with optional jitter + random probes.

    `levels_seconds` defines the interval at each backoff level. Level advances
    every `idle_rounds_per_level` consecutive "idle" runs.
    """

    levels_seconds: list[int]
    idle_rounds_per_level: int = 3
    jitter_fraction: float = 0.1
    random_probe_probability: float = 0.0
    random_probe_min_seconds: int = 5
    random_probe_max_seconds: int = 15


@dataclass
class WatchPolicy:
    policy_version: str = "watch_v1"

    heartbeat: BackoffPolicy = field(
        default_factory=lambda: BackoffPolicy(levels_seconds=[30, 60, 120, 180])
    )
    survey: BackoffPolicy = field(
        default_factory=lambda: BackoffPolicy(levels_seconds=[60, 120, 300, 900])
    )
    log_tail: BackoffPolicy = field(
        default_factory=lambda: BackoffPolicy(levels_seconds=[30, 60, 120, 180])
    )

    max_log_bytes: int = 32_768
    max_log_lines: int = 200
    max_logs_per_tick: int = 10
    max_tracked_logs: int = 20_000
    max_glob_matches_per_project: int = 25
    resolve_refresh_seconds: int = 3600
    max_tokens_per_resolve: int = 2000
    max_spool_batches: int = 5000
    max_events_per_post: int = 50


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_backoff_policy(raw: Any, default: BackoffPolicy) -> BackoffPolicy:
    if not isinstance(raw, dict):
        return default

    levels = raw.get("levels_seconds", default.levels_seconds)
    if not isinstance(levels, list) or not levels:
        levels = default.levels_seconds
    parsed_levels: list[int] = []
    for item in levels:
        try:
            parsed_levels.append(max(1, int(item)))
        except Exception:
            continue
    if not parsed_levels:
        parsed_levels = default.levels_seconds

    try:
        idle_rounds_per_level = max(1, int(raw.get("idle_rounds_per_level", default.idle_rounds_per_level)))
    except Exception:
        idle_rounds_per_level = default.idle_rounds_per_level

    try:
        jitter_fraction = float(raw.get("jitter_fraction", default.jitter_fraction))
    except Exception:
        jitter_fraction = default.jitter_fraction
    jitter_fraction = _clamp(jitter_fraction, lo=0.0, hi=0.5)

    try:
        random_probe_probability = float(
            raw.get("random_probe_probability", default.random_probe_probability)
        )
    except Exception:
        random_probe_probability = default.random_probe_probability
    random_probe_probability = _clamp(random_probe_probability, lo=0.0, hi=1.0)

    try:
        random_probe_min_seconds = max(
            1, int(raw.get("random_probe_min_seconds", default.random_probe_min_seconds))
        )
    except Exception:
        random_probe_min_seconds = default.random_probe_min_seconds

    try:
        random_probe_max_seconds = max(
            random_probe_min_seconds,
            int(raw.get("random_probe_max_seconds", default.random_probe_max_seconds)),
        )
    except Exception:
        random_probe_max_seconds = default.random_probe_max_seconds

    return BackoffPolicy(
        levels_seconds=parsed_levels,
        idle_rounds_per_level=idle_rounds_per_level,
        jitter_fraction=jitter_fraction,
        random_probe_probability=random_probe_probability,
        random_probe_min_seconds=random_probe_min_seconds,
        random_probe_max_seconds=random_probe_max_seconds,
    )


def _parse_watch_policy(raw: Any) -> WatchPolicy:
    default = WatchPolicy()
    if not isinstance(raw, dict):
        return default

    merged = asdict(default)
    _deep_update(merged, raw)

    policy_version = str(merged.get("policy_version") or default.policy_version)
    heartbeat = _parse_backoff_policy(merged.get("heartbeat"), default.heartbeat)
    survey = _parse_backoff_policy(merged.get("survey"), default.survey)
    log_tail = _parse_backoff_policy(merged.get("log_tail"), default.log_tail)

    def _int(name: str, fallback: int) -> int:
        try:
            return max(1, int(merged.get(name, fallback)))
        except Exception:
            return fallback

    max_log_bytes = _int("max_log_bytes", default.max_log_bytes)
    max_log_lines = _int("max_log_lines", default.max_log_lines)
    max_logs_per_tick = _int("max_logs_per_tick", default.max_logs_per_tick)
    max_tracked_logs = _int("max_tracked_logs", default.max_tracked_logs)
    max_glob_matches_per_project = _int(
        "max_glob_matches_per_project", default.max_glob_matches_per_project
    )
    resolve_refresh_seconds = _int("resolve_refresh_seconds", default.resolve_refresh_seconds)
    max_tokens_per_resolve = _int("max_tokens_per_resolve", default.max_tokens_per_resolve)
    max_spool_batches = _int("max_spool_batches", default.max_spool_batches)
    max_events_per_post = _int("max_events_per_post", default.max_events_per_post)

    return WatchPolicy(
        policy_version=policy_version,
        heartbeat=heartbeat,
        survey=survey,
        log_tail=log_tail,
        max_log_bytes=max_log_bytes,
        max_log_lines=max_log_lines,
        max_logs_per_tick=max_logs_per_tick,
        max_tracked_logs=max_tracked_logs,
        max_glob_matches_per_project=max_glob_matches_per_project,
        resolve_refresh_seconds=resolve_refresh_seconds,
        max_tokens_per_resolve=max_tokens_per_resolve,
        max_spool_batches=max_spool_batches,
        max_events_per_post=max_events_per_post,
    )


def _policy_hash(policy: WatchPolicy) -> str:
    payload = json.dumps(asdict(policy), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _next_interval_seconds(
    *,
    policy: BackoffPolicy,
    idle_rounds: int,
    activity: bool,
    rng: random.Random,
) -> tuple[int, int, str]:
    if activity:
        return max(1, int(policy.levels_seconds[0])), 0, "activity"

    next_idle_rounds = max(0, int(idle_rounds)) + 1
    rounds_per_level = max(1, int(policy.idle_rounds_per_level))
    # Keep level 0 for the first `rounds_per_level` idle runs.
    level = min((next_idle_rounds - 1) // rounds_per_level, len(policy.levels_seconds) - 1)
    base = max(1, int(policy.levels_seconds[level]))

    if level > 0 and policy.random_probe_probability > 0 and rng.random() < policy.random_probe_probability:
        probe = rng.randint(
            max(1, int(policy.random_probe_min_seconds)),
            max(1, int(policy.random_probe_max_seconds)),
        )
        return max(1, int(probe)), next_idle_rounds, f"idle_level_{level}:probe"

    jitter = policy.jitter_fraction
    if jitter > 0:
        factor = rng.uniform(1.0 - jitter, 1.0 + jitter)
        base = max(1, int(round(base * factor)))

    return base, next_idle_rounds, f"idle_level_{level}"


class _StateDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
              key TEXT PRIMARY KEY,
              value_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS log_cursor (
              path TEXT PRIMARY KEY,
              offset INTEGER NOT NULL DEFAULT 0,
              last_size INTEGER NOT NULL DEFAULT 0,
              last_mtime REAL NOT NULL DEFAULT 0,
              idle_rounds INTEGER NOT NULL DEFAULT 0,
              next_due REAL NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spool (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at REAL NOT NULL,
              batch_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get_json(self, key: str) -> Any | None:
        row = self._conn.execute("SELECT value_json FROM kv WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def set_json(self, key: str, value: Any) -> None:
        payload = json.dumps(value, separators=(",", ":"), sort_keys=True)
        self._conn.execute(
            "INSERT INTO kv (key,value_json) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, payload),
        )
        self._conn.commit()

    def ensure_log(self, path: str) -> None:
        self.ensure_logs([path])

    def ensure_logs(self, paths: list[str]) -> None:
        payload = [(p,) for p in paths if isinstance(p, str) and p]
        if not payload:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO log_cursor (path) VALUES (?)",
            payload,
        )
        self._conn.commit()

    def due_logs(self, *, now: float, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT path, offset, last_size, last_mtime, idle_rounds, next_due FROM log_cursor WHERE next_due <= ? ORDER BY next_due ASC LIMIT ?",
            (now, max(1, int(limit))),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "path": str(row[0]),
                    "offset": int(row[1]),
                    "last_size": int(row[2]),
                    "last_mtime": float(row[3]),
                    "idle_rounds": int(row[4]),
                    "next_due": float(row[5]),
                }
            )
        return results

    def update_log(
        self,
        *,
        path: str,
        offset: int,
        last_size: int,
        last_mtime: float,
        idle_rounds: int,
        next_due: float,
    ) -> None:
        self._conn.execute(
            """
            UPDATE log_cursor
            SET offset=?, last_size=?, last_mtime=?, idle_rounds=?, next_due=?
            WHERE path=?
            """,
            (int(offset), int(last_size), float(last_mtime), int(idle_rounds), float(next_due), path),
        )
        self._conn.commit()

    def spool_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM spool").fetchone()
        return int(row[0] if row else 0)

    def spool_put(self, batch: list[dict[str, Any]], *, now: float, max_batches: int) -> None:
        if self.spool_count() >= max(1, int(max_batches)):
            # Drop oldest to keep bounded.
            self._conn.execute(
                "DELETE FROM spool WHERE id IN (SELECT id FROM spool ORDER BY id ASC LIMIT 1)"
            )
        self._conn.execute(
            "INSERT INTO spool (created_at, batch_json) VALUES (?, ?)",
            (float(now), json.dumps(batch, separators=(",", ":"), sort_keys=True)),
        )
        self._conn.commit()

    def spool_peek(self, *, limit: int) -> list[tuple[int, list[dict[str, Any]]]]:
        rows = self._conn.execute(
            "SELECT id, batch_json FROM spool ORDER BY id ASC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        batches: list[tuple[int, list[dict[str, Any]]]] = []
        for row in rows:
            try:
                batches.append((int(row[0]), list(json.loads(row[1]))))
            except Exception:
                continue
        return batches

    def spool_delete(self, batch_id: int) -> None:
        self._conn.execute("DELETE FROM spool WHERE id=?", (int(batch_id),))
        self._conn.commit()


def _scan_project_dirs(*, roots: list[str], globs: list[str]) -> dict[str, list[str]]:
    patterns = globs or ["MSPC*"]
    found: dict[str, list[str]] = {}
    for root in roots:
        try:
            with os.scandir(root) as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    name = entry.name
                    if patterns and not any(fnmatch.fnmatch(name, pat) for pat in patterns):
                        continue
                    found.setdefault(name, []).append(entry.path)
        except FileNotFoundError:
            continue
        except PermissionError:
            continue
    return found


def _chunks(items: list[str], n: int) -> list[list[str]]:
    if n <= 0:
        return [items]
    return [items[i : i + n] for i in range(0, len(items), n)]


def _resolve_projects(
    *,
    server: str,
    tokens: list[str],
    api_key: str | None,
    timeout_seconds: int,
    max_tokens_per_request: int,
) -> dict[str, dict[str, Any]]:
    if not tokens:
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    url = server.rstrip("/") + "/api/agents/projects/resolve"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ispec-agent/0",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    for batch in _chunks(tokens, max_tokens_per_request):
        body = json.dumps({"tokens": batch}, separators=(",", ":"), sort_keys=True).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        for row in payload.get("projects", []):
            token = row.get("token")
            if isinstance(token, str) and token:
                mapping[token] = row

    return mapping


def _tail_file(
    *,
    path: str,
    offset: int,
    max_bytes: int,
    max_lines: int,
) -> tuple[str, int, dict[str, Any]]:
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return "", offset, {"exists": False}
    except PermissionError:
        return "", offset, {"exists": False, "error": "permission_denied"}

    size = int(stat.st_size)
    mtime = float(stat.st_mtime)
    rotated = False
    if size < offset:
        offset = 0
        rotated = True

    read_bytes = max(0, int(max_bytes))
    if read_bytes <= 0:
        return "", offset, {"exists": True, "size": size, "mtime": mtime, "rotated": rotated}

    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(read_bytes)
            new_offset = offset + len(data)
    except Exception as exc:
        return "", offset, {"exists": True, "size": size, "mtime": mtime, "error": str(exc)}

    if not data:
        return "", new_offset, {"exists": True, "size": size, "mtime": mtime, "rotated": rotated, "bytes_read": 0}

    text = data.decode("utf-8", errors="replace")
    if max_lines > 0:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(lines[-max_lines:])

    meta: dict[str, Any] = {
        "exists": True,
        "size": size,
        "mtime": mtime,
        "rotated": rotated,
        "bytes_read": len(data),
    }
    if new_offset < size:
        meta["truncated"] = True
    return text, new_offset, meta


def _run_watch(args) -> None:
    roots = args.root or []
    if not roots:
        raise SystemExit("--root is required (repeatable)")

    globs = args.project_dir_glob or ["MSPC*"]
    tackle_relpaths = args.tackle_relpath or [
        "tackle.log",
        os.path.join("config", "tackle.log"),
        os.path.join("results", "*", "tackle.log"),
    ]

    policy_raw: Any = {}
    if args.policy_file:
        with open(args.policy_file, "r", encoding="utf-8") as f:
            policy_raw = json.load(f)
    policy = _parse_watch_policy(policy_raw)
    policy_id = _policy_hash(policy)

    state_path = Path(args.state_db).expanduser() if args.state_db else Path.home() / ".ispec-agent" / "state.db"
    db = _StateDB(state_path)
    rng = random.Random()

    ts = _utcnow_iso()
    policy_event = {
        "type": "agent_policy_v1",
        "agent_id": args.agent_id,
        "ts": ts,
        "name": "watch_policy",
        "dimensions": {"policy_id": policy_id, "policy_version": policy.policy_version},
        "value": asdict(policy),
    }

    try:
        _post_events(
            server=args.server,
            events=[policy_event],
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception:
        db.spool_put([policy_event], now=time.time(), max_batches=policy.max_spool_batches)

    survey_state = db.get_json("survey_state") or {}
    survey_idle_rounds = int(survey_state.get("idle_rounds", 0) or 0)
    survey_next_due = float(survey_state.get("next_due", 0) or 0)
    last_tokens_hash = str(survey_state.get("tokens_hash") or "")
    # Always re-check immediately on startup to avoid stale scheduling after restarts.
    survey_next_due = min(survey_next_due, time.time())

    resolve_state = db.get_json("resolve_state") or {}
    last_resolve_at = float(resolve_state.get("resolved_at", 0) or 0)
    project_index: dict[str, dict[str, Any]] = {}

    token_paths: dict[str, list[str]] = {}
    log_ctx_by_path: dict[str, dict[str, Any]] = {}
    tracked_log_paths: list[str] = []

    while True:
        now = time.time()
        events: list[dict[str, Any]] = []
        activity_any = False
        errors: list[str] = []
        survey_changed = False

        # Best-effort flush of spooled event batches.
        for batch_id, batch in db.spool_peek(limit=5):
            try:
                _post_events(
                    server=args.server,
                    events=batch,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout_seconds,
                )
                db.spool_delete(batch_id)
            except Exception:
                break

        # Survey for MSPC* folders (or configured patterns).
        survey_did_run = False
        if now >= survey_next_due:
            survey_did_run = True
            token_paths = _scan_project_dirs(roots=roots, globs=globs)
            tokens = sorted(token_paths.keys())
            tokens_hash = hashlib.sha256(
                ("\n".join(tokens)).encode("utf-8", errors="replace")
            ).hexdigest()
            survey_changed = tokens_hash != last_tokens_hash
            activity_any = activity_any or survey_changed

            interval, survey_idle_rounds, survey_reason = _next_interval_seconds(
                policy=policy.survey,
                idle_rounds=survey_idle_rounds,
                activity=survey_changed,
                rng=rng,
            )
            survey_next_due = now + interval
            last_tokens_hash = tokens_hash
            db.set_json(
                "survey_state",
                {
                    "idle_rounds": survey_idle_rounds,
                    "next_due": survey_next_due,
                    "tokens_hash": tokens_hash,
                },
            )
            events.append(
                {
                    "type": "project_dir_survey_v1",
                    "agent_id": args.agent_id,
                    "ts": _utcnow_iso(),
                    "name": "project_dir_survey",
                    "dimensions": {
                        "policy_id": policy_id,
                        "roots": roots,
                        "project_dir_globs": globs,
                        "changed": bool(survey_changed),
                        "reason": survey_reason,
                    },
                    "value": {"project_dir_count": len(tokens)},
                }
            )

        # Resolve folder tokens -> project ids (avoid MSPC parsing in the agent).
        tokens = sorted(token_paths.keys())
        should_resolve = bool(
            tokens
            and (
                not project_index
                or survey_changed
                or (now - last_resolve_at) >= float(policy.resolve_refresh_seconds)
            )
        )
        resolved_this_tick = False
        if should_resolve:
            try:
                project_index = _resolve_projects(
                    server=args.server,
                    tokens=tokens,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout_seconds,
                    max_tokens_per_request=policy.max_tokens_per_resolve,
                )
                resolved_this_tick = True
                last_resolve_at = now
                db.set_json("resolve_state", {"resolved_at": last_resolve_at, "count": len(project_index)})
                events.append(
                    {
                        "type": "project_resolve_v1",
                        "agent_id": args.agent_id,
                        "ts": _utcnow_iso(),
                        "name": "project_resolve",
                        "dimensions": {"policy_id": policy_id},
                        "value": {"resolved": len(project_index), "tokens": len(tokens)},
                    }
                )
            except Exception as exc:
                errors.append(f"resolve_failed:{exc}")

        # Track tackle.log candidates for (currently present) project directories.
        # Rebuild only when survey or resolution changed, to keep the steady-state loop cheap.
        if survey_changed or resolved_this_tick or not tracked_log_paths:
            log_ctx_by_path = {}
            tracked_log_paths = []
            for token, paths in token_paths.items():
                proj = project_index.get(token)
                for base in paths:
                    root_dir = str(Path(base).parent)
                    for rel in tackle_relpaths:
                        if len(tracked_log_paths) >= policy.max_tracked_logs:
                            break
                        has_glob = any(ch in rel for ch in ("*", "?", "["))
                        if has_glob:
                            pattern = str(Path(base) / rel)
                            matches = glob.glob(pattern, recursive="**" in rel)
                            for match in matches[: policy.max_glob_matches_per_project]:
                                if len(tracked_log_paths) >= policy.max_tracked_logs:
                                    break
                                if not os.path.isfile(match):
                                    continue
                                full = str(Path(match))
                                tracked_log_paths.append(full)
                                log_ctx_by_path[full] = {
                                    "token": token,
                                    "project": proj,
                                    "root": root_dir,
                                    "pattern": rel,
                                }
                        else:
                            full = str(Path(base) / rel)
                            tracked_log_paths.append(full)
                            log_ctx_by_path[full] = {
                                "token": token,
                                "project": proj,
                                "root": root_dir,
                                "pattern": rel,
                            }
            db.ensure_logs(tracked_log_paths)

        # Tail due logs (per-log backoff).
        due = db.due_logs(now=now, limit=policy.max_logs_per_tick)
        tailed = 0
        new_bytes_total = 0
        log_decisions: list[dict[str, Any]] = []
        for row in due:
            path = row["path"]
            offset = row["offset"]
            idle_rounds = row["idle_rounds"]
            text, new_offset, meta = _tail_file(
                path=path,
                offset=offset,
                max_bytes=policy.max_log_bytes,
                max_lines=policy.max_log_lines,
            )
            had_new = bool(text)
            if had_new:
                activity_any = True
                new_bytes_total += int(meta.get("bytes_read") or 0)

                # Attach best-effort project context.
                ctx = log_ctx_by_path.get(path) or {}
                proj = ctx.get("project") if isinstance(ctx, dict) else None
                dims: dict[str, Any] = {"policy_id": policy_id, "path": path}
                if isinstance(ctx, dict):
                    dims.update({k: v for k, v in ctx.items() if k != "project"})
                if isinstance(proj, dict):
                    dims["project_id"] = proj.get("project_id")
                    dims["display_id"] = proj.get("display_id")

                events.append(
                    {
                        "type": "tackle_log_tail_v1",
                        "agent_id": args.agent_id,
                        "ts": _utcnow_iso(),
                        "name": "tackle_log_tail",
                        "dimensions": dims,
                        "value": {
                            "from_offset": offset,
                            "to_offset": new_offset,
                            "meta": meta,
                            "text": text,
                        },
                    }
                )

            interval, next_idle_rounds, reason = _next_interval_seconds(
                policy=policy.log_tail,
                idle_rounds=idle_rounds,
                activity=had_new,
                rng=rng,
            )
            next_due = now + interval
            db.update_log(
                path=path,
                offset=new_offset,
                last_size=int(meta.get("size") or row["last_size"]),
                last_mtime=float(meta.get("mtime") or row["last_mtime"]),
                idle_rounds=next_idle_rounds,
                next_due=next_due,
            )
            tailed += 1
            if len(log_decisions) < 25:
                log_decisions.append(
                    {
                        "path": path,
                        "had_new": bool(had_new),
                        "reason": reason,
                        "next_due_seconds": int(interval),
                        "rotated": bool(meta.get("rotated")),
                        "truncated": bool(meta.get("truncated")),
                    }
                )

        # Heartbeat / decision summary (learning-friendly).
        hb_interval, hb_idle_rounds, hb_reason = _next_interval_seconds(
            policy=policy.heartbeat,
            idle_rounds=int(db.get_json("heartbeat_idle_rounds") or 0),
            activity=activity_any,
            rng=rng,
        )
        db.set_json("heartbeat_idle_rounds", hb_idle_rounds)

        events.append(
            {
                "type": "heartbeat_v1",
                "agent_id": args.agent_id,
                "ts": _utcnow_iso(),
                "name": "watch_heartbeat",
                "dimensions": {
                    "policy_id": policy_id,
                    "policy_version": policy.policy_version,
                    "reason": hb_reason,
                    "errors": errors,
                },
                "value": {
                    "roots": roots,
                    "project_dir_count": len(token_paths),
                    "resolved_projects": len(project_index),
                    "tracked_logs": len(tracked_log_paths),
                    "tailed_logs": tailed,
                    "new_bytes_total": new_bytes_total,
                    "spool_batches": db.spool_count(),
                    "log_decisions_sample": log_decisions,
                    "next_heartbeat_seconds": hb_interval,
                },
            }
        )

        # Send events (or spool) in small batches.
        max_events = max(1, int(policy.max_events_per_post))
        for batch in [events[i : i + max_events] for i in range(0, len(events), max_events)]:
            try:
                _post_events(
                    server=args.server,
                    events=batch,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout_seconds,
                )
            except Exception:
                db.spool_put(batch, now=now, max_batches=policy.max_spool_batches)

        if args.once:
            db.close()
            return

        # Sleep until the next due work (survey/log) or the heartbeat interval, whichever is sooner.
        next_due_times = [survey_next_due, now + hb_interval]
        # Also consider all logs (not just due subset) to avoid stalling.
        row = db._conn.execute("SELECT MIN(next_due) FROM log_cursor").fetchone()
        if row and row[0] is not None:
            next_due_times.append(float(row[0]))
        sleep_until = min(next_due_times) if next_due_times else (now + hb_interval)
        sleep_seconds = max(1.0, sleep_until - time.time())
        time.sleep(sleep_seconds)
