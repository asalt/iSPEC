"""CLI helpers for interacting with the support assistant API.

This is intentionally a minimal, one-shot dev interface that talks to the same
HTTP endpoints used by the UI and Slack bridge:
  - POST /api/support/chat
  - POST /api/support/choose
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from ispec.logging import get_logger

logger = get_logger(__name__)


def register_subcommands(subparsers) -> None:
    chat_parser = subparsers.add_parser("chat", help="Send a one-shot message to the support assistant")
    chat_parser.add_argument(
        "--server",
        default=os.getenv("ISPEC_API_URL") or "",
        help="iSPEC API base URL (default: $ISPEC_API_URL or http://127.0.0.1:${ISPEC_PORT or 3001})",
    )
    chat_parser.add_argument(
        "--api-key",
        default=os.getenv("ISPEC_API_KEY") or "",
        help="Optional iSPEC API key (default: $ISPEC_API_KEY)",
    )
    chat_parser.add_argument(
        "--session-id",
        default=None,
        help="Support session id (repeat to continue a conversation; default: auto-generated)",
    )
    chat_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout seconds (default: 60)",
    )
    chat_parser.add_argument(
        "--choose",
        type=int,
        choices=[0, 1],
        default=None,
        help="If compare mode returns two choices, automatically select 0 or 1.",
    )
    chat_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON response (instead of just the message text).",
    )
    chat_parser.add_argument(
        "message",
        nargs="*",
        help="Message text. Use '-' to read from stdin, or pipe stdin with no args.",
    )

    choose_parser = subparsers.add_parser("choose", help="Choose a compare-mode response")
    choose_parser.add_argument(
        "--server",
        default=os.getenv("ISPEC_API_URL") or "",
        help="iSPEC API base URL (default: $ISPEC_API_URL or http://127.0.0.1:${ISPEC_PORT or 3001})",
    )
    choose_parser.add_argument(
        "--api-key",
        default=os.getenv("ISPEC_API_KEY") or "",
        help="Optional iSPEC API key (default: $ISPEC_API_KEY)",
    )
    choose_parser.add_argument("--session-id", required=True, help="Support session id.")
    choose_parser.add_argument("--user-message-id", type=int, required=True, help="User message id to choose for.")
    choose_parser.add_argument("--choice-index", type=int, choices=[0, 1], required=True, help="Choice index (0 or 1).")
    choose_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout seconds (default: 60)",
    )
    choose_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON response (instead of just the message text).",
    )


def dispatch(args) -> None:
    if args.subcommand == "chat":
        _cmd_chat(args)
        return
    if args.subcommand == "choose":
        _cmd_choose(args)
        return
    raise SystemExit(f"Unknown support subcommand: {args.subcommand}")


def _probe_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _default_server_url(value: str | None) -> str:
    raw = (value or "").strip().rstrip("/")
    if raw:
        return raw

    host = (os.getenv("ISPEC_HOST") or "").strip() or "127.0.0.1"
    port = (os.getenv("ISPEC_PORT") or "").strip() or "3001"
    return f"http://{_probe_host(host)}:{port}"


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    return headers


def _read_message(tokens: list[str]) -> str:
    if tokens:
        if len(tokens) == 1 and tokens[0] == "-":
            return (sys.stdin.read() or "").strip()
        return " ".join(tokens).strip()

    if not sys.stdin.isatty():
        return (sys.stdin.read() or "").strip()

    raise SystemExit("Message is required. Pass it as arguments, use '-', or pipe stdin.")


def _default_session_id() -> str:
    suffix = uuid.uuid4().hex[:12]
    return f"cli:{suffix}"


@dataclass(frozen=True)
class _ChatResult:
    session_id: str
    message_id: int | None
    message: str | None
    compare: dict[str, Any] | None
    raw: dict[str, Any]


def _post_chat(
    *,
    server: str,
    api_key: str,
    session_id: str,
    message: str,
    timeout_seconds: float,
) -> _ChatResult:
    url = server.rstrip("/") + "/api/support/chat"
    payload: dict[str, Any] = {
        "sessionId": session_id,
        "message": message,
        "history": [],
        "meta": {"source": "cli", "sent_at_unix": time.time()},
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers=_headers(api_key),
            timeout=max(1.0, float(timeout_seconds)),
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        body = ""
        try:
            body = str(getattr(getattr(exc, "response", None), "text", "") or "")
        except Exception:
            body = ""
        body = body.strip()
        suffix = f": {body}" if body else ""
        raise SystemExit(f"Support chat failed ({status or 'http_error'}){suffix}") from exc
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response payload (expected JSON object).")

    session = str(data.get("sessionId") or session_id)
    msg_id = data.get("messageId")
    message_id = int(msg_id) if isinstance(msg_id, int) and msg_id > 0 else None
    message_text = data.get("message")
    message = str(message_text) if isinstance(message_text, str) else None
    compare = data.get("compare") if isinstance(data.get("compare"), dict) else None
    return _ChatResult(
        session_id=session,
        message_id=message_id,
        message=message,
        compare=compare,
        raw=data,
    )


def _post_choose(
    *,
    server: str,
    api_key: str,
    session_id: str,
    user_message_id: int,
    choice_index: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = server.rstrip("/") + "/api/support/choose"
    payload = {"sessionId": session_id, "userMessageId": int(user_message_id), "choiceIndex": int(choice_index)}
    try:
        resp = requests.post(
            url,
            json=payload,
            headers=_headers(api_key),
            timeout=max(1.0, float(timeout_seconds)),
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        body = ""
        try:
            body = str(getattr(getattr(exc, "response", None), "text", "") or "")
        except Exception:
            body = ""
        body = body.strip()
        suffix = f": {body}" if body else ""
        raise SystemExit(f"Support choose failed ({status or 'http_error'}){suffix}") from exc
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response payload (expected JSON object).")
    return data


def _print_compare(compare: dict[str, Any]) -> None:
    user_message_id = compare.get("userMessageId")
    choices = compare.get("choices")
    if not isinstance(choices, list):
        print(json.dumps(compare, ensure_ascii=False, indent=2))
        return

    if isinstance(user_message_id, int):
        print(f"compare userMessageId={user_message_id}")
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        idx = choice.get("index")
        msg = str(choice.get("message") or "").strip()
        if msg:
            print(f"\n--- choice {idx} ---\n{msg}\n")


def _cmd_chat(args) -> None:
    server = _default_server_url(args.server)
    api_key = str(args.api_key or "")
    requested_session_id = str(args.session_id or "").strip()
    session_id = requested_session_id or _default_session_id()
    session_id_auto = not bool(requested_session_id)
    message = _read_message(list(getattr(args, "message", []) or []))

    result = _post_chat(
        server=server,
        api_key=api_key,
        session_id=session_id,
        message=message,
        timeout_seconds=float(args.timeout_seconds),
    )

    if args.json:
        print(json.dumps(result.raw, ensure_ascii=False, indent=2))
        return

    if session_id_auto:
        print(f"session_id={result.session_id}", file=sys.stderr)

    if result.message is not None:
        print(result.message)
        return

    if result.compare is None:
        print(json.dumps(result.raw, ensure_ascii=False, indent=2))
        return

    if args.choose is not None:
        user_message_id = result.compare.get("userMessageId")
        if isinstance(user_message_id, int) and user_message_id > 0:
            chosen = _post_choose(
                server=server,
                api_key=api_key,
                session_id=result.session_id,
                user_message_id=int(user_message_id),
                choice_index=int(args.choose),
                timeout_seconds=float(args.timeout_seconds),
            )
            msg = chosen.get("message")
            if isinstance(msg, str) and msg.strip():
                print(msg.strip())
                return
            print(json.dumps(chosen, ensure_ascii=False, indent=2))
            return

    _print_compare(result.compare)


def _cmd_choose(args) -> None:
    server = _default_server_url(args.server)
    api_key = str(args.api_key or "")
    session_id = str(args.session_id or "").strip()
    if not session_id:
        raise SystemExit("--session-id is required")

    data = _post_choose(
        server=server,
        api_key=api_key,
        session_id=session_id,
        user_message_id=int(args.user_message_id),
        choice_index=int(args.choice_index),
        timeout_seconds=float(args.timeout_seconds),
    )
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    msg = data.get("message")
    if isinstance(msg, str) and msg.strip():
        print(msg.strip())
        return
    print(json.dumps(data, ensure_ascii=False, indent=2))
