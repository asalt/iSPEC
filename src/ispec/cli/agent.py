"""Windows-friendly agent helpers (v0).

This CLI is intentionally lightweight: it collects simple metrics locally and
POSTs structured events to an iSPEC server over HTTP.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any


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


def dispatch(args) -> None:
    if args.subcommand == "disk-free":
        _run_disk_free(args)
        return
    raise SystemExit(f"Unknown agent subcommand: {args.subcommand}")


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
