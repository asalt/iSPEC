"""Small explicit-target Slack reader/replier using iSPEC's configured bot.

Run with ``python -m ispec.cli.slack_history --help``. No assistant is invoked.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from ispec.cli import slack
from ispec.slack_delivery import log_delivery
from ispec.cli.env import extract_env_files, load_env_files
from ispec.config.paths import resolve_config_dir


class SlackReadError(Exception):
    """An error safe to show without raw provider responses or credentials."""


def _call(token: str, endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        result = slack._slack_direct_api_call(
            token=token, endpoint=endpoint, payload=payload,
            timeout_seconds=timeout, as_form=True,
        )
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        hint = " Rate limited; wait before retrying." if status == 429 else ""
        if endpoint == "chat.postMessage":
            hint += " Delivery may be uncertain; read the thread before retrying."
        raise SlackReadError(f"{endpoint}: request failed (HTTP {status or 'unavailable'}).{hint}") from None
    except (ValueError, TypeError):
        raise SlackReadError(f"{endpoint}: invalid provider response.") from None
    if not result.get("ok"):
        error = str(result.get("error") or "unknown_error")
        if not re.fullmatch(r"[a-z_]+", error):
            error = "provider_error"
        hints = {
            "missing_scope": "The configured bot lacks this read permission; no access changes were made.",
            "not_in_channel": "The bot is not a member; this helper never joins channels.",
            "channel_not_found": "Check the channel ID and the bot's membership.",
            "not_allowed_token_type": "Slack does not support this method for the configured bot token.",
            "thread_not_found": "Use the parent message's exact timestamp for --thread-ts.",
            "invalid_auth": "The configured bot credential was rejected.",
            "ratelimited": "Wait before retrying; this helper does not retry automatically.",
        }
        raise SlackReadError(f"{endpoint}: {error}. {hints.get(error, '')}".strip())
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", action="append", help="Existing iSPEC KEY=value credentials file; repeatable.")
    p.add_argument("--channel", required=True, help="Exact Slack channel ID (no aliases or auto-join).")
    p.add_argument("--thread-ts", help="Exact parent timestamp; selects a thread read or reply target.")
    p.add_argument("--date", help="Read channel history for one YYYY-MM-DD; never sends or imports receipts.")
    p.add_argument("--timezone", help="IANA timezone for --date; remember for later reads (initial default America/Chicago).")
    p.add_argument("--limit", type=int, default=10, help="Messages per page, 1–100 (default 10).")
    p.add_argument("--max-pages", type=int, default=1, help="Bounded cursor pagination, 1–100 (default 1).")
    p.add_argument("--cursor", default="", help="Continue a previous read from next_cursor.")
    p.add_argument("--record-outbound", action="store_true", help="Import only the exact bot-authored --thread-ts message as a delivery receipt; never send.")
    p.add_argument("--timeout-seconds", type=float, default=15)
    text = p.add_mutually_exclusive_group()
    text.add_argument("--text", help="Exact reply text; requires --dry-run or --send.")
    text.add_argument("--stdin", action="store_true", help="Read exact reply text from stdin.")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Preview reply payload without network access.")
    action.add_argument("--send", action="store_true", help="Explicitly post one reply; requires --thread-ts and text.")
    return p


def timezone_settings_path() -> Path:
    return Path(resolve_config_dir().value) / "slack-history.json"


def saved_timezone() -> str:
    try:
        value = json.loads(timezone_settings_path().read_text(encoding="utf-8"))["timezone"]
        if not isinstance(value, str):
            raise ValueError("Invalid timezone setting")
        ZoneInfo(value)
        return value
    except FileNotFoundError:
        return "America/Chicago"
    except (OSError, ValueError, KeyError, TypeError, ZoneInfoNotFoundError):
        raise SlackReadError("Cannot read saved Slack history timezone; specify --timezone to replace it.") from None


def save_timezone(value: str) -> None:
    path = timezone_settings_path()
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # One local preference, atomically replaced; no credentials or messages.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".slack-history-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({"timezone": value}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError:
        raise SlackReadError("Cannot save Slack history timezone; no Slack request was made.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def date_window(day: str, timezone: str) -> tuple[int, int]:
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("Invalid date")
        start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=ZoneInfo(timezone))
        return int(start.timestamp()), int((start + timedelta(days=1)).timestamp())
    except (ValueError, ZoneInfoNotFoundError):
        raise SlackReadError("Use a valid --date YYYY-MM-DD and IANA --timezone.") from None


def in_date_window(row: dict, bounds: tuple[int, int]) -> bool:
    try:
        timestamp = Decimal(str(row.get("ts", "")))
        return timestamp.is_finite() and bounds[0] <= timestamp < bounds[1]
    except InvalidOperation:
        return False


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[CGD][A-Z0-9]+", args.channel):
        raise SlackReadError("--channel must be an explicit Slack channel ID.")
    if args.thread_ts and not re.fullmatch(r"[0-9]+\.[0-9]+", args.thread_ts):
        raise SlackReadError("--thread-ts must be the exact Slack parent timestamp.")
    if not 1 <= args.limit <= 100 or not 1 <= args.max_pages <= 100:
        raise SlackReadError("--limit and --max-pages must each be between 1 and 100.")
    if not 0 < args.timeout_seconds <= 120:
        raise SlackReadError("--timeout-seconds must be greater than 0 and at most 120.")
    text = sys.stdin.read() if args.stdin else args.text
    posting = args.send or args.dry_run
    timezone = args.timezone
    if timezone is not None:
        try:
            ZoneInfo(timezone)
        except (ValueError, ZoneInfoNotFoundError):
            raise SlackReadError("Use a valid IANA --timezone.") from None
    if args.date and timezone is None:
        timezone = saved_timezone()
    bounds = date_window(args.date, timezone) if args.date else None
    if bounds and (posting or args.record_outbound or args.thread_ts or text is not None):
        raise SlackReadError("--date is a read-only channel scan; fetch chosen threads separately.")
    if args.record_outbound and (posting or text is not None or not args.thread_ts or args.cursor):
        raise SlackReadError("--record-outbound requires one exact --thread-ts and cannot send or paginate.")
    if text is not None and not posting:
        raise SlackReadError("Reply text requires explicit --dry-run or --send; nothing was sent.")
    if posting and (not args.thread_ts or not text or not text.strip()):
        raise SlackReadError("A reply requires --thread-ts and nonempty --text or --stdin.")
    if posting and args.cursor:
        raise SlackReadError("--cursor is only valid for reads.")
    payload = {"channel": args.channel, "thread_ts": args.thread_ts, "text": text,
               "unfurl_links": "false", "unfurl_media": "false"}
    if args.dry_run:
        return {"ok": True, "dry_run": True, "identity_verified": False, "payload": payload}
    if args.timezone is not None and not posting and not args.record_outbound:
        save_timezone(timezone)
    token = slack._cli_slack_bot_token()
    if not token:
        raise SlackReadError("No bot token configured. Use existing iSPEC --env-file settings or inherited bot environment.")
    auth = _call(token, "auth.test", {}, args.timeout_seconds)
    info = _call(token, "conversations.info", {"channel": args.channel}, args.timeout_seconds)
    channel = info.get("channel") or {}
    if channel.get("id") != args.channel:
        raise SlackReadError("Slack returned an unexpected channel identity; stopped.")
    base = {"ok": True, "source": "live_slack", "identity": {
        k: auth.get(k) for k in ("team", "team_id", "user_id", "bot_id")},
        "channel": {k: channel.get(k) for k in ("id", "name", "is_member", "is_private")}}
    if args.record_outbound:
        # This explicit repair is not a history backfill and never invokes the assistant.
        recipient = channel.get("user")
        if channel.get("is_im"):
            if not recipient:
                raise SlackReadError("Cannot verify the DM peer; nothing imported.")
            peer = _call(token, "users.info", {"user": recipient}, args.timeout_seconds).get("user") or {}
            if peer.get("id") != recipient:
                raise SlackReadError("DM peer verification failed; nothing imported.")
            base["recipient"] = {k: peer.get(k) for k in ("id", "name", "real_name")}
        found = _call(token, "conversations.replies", {
            "channel": args.channel, "ts": args.thread_ts, "limit": 1,
        }, args.timeout_seconds)
        row = next((m for m in found.get("messages", []) if m.get("ts") == args.thread_ts), None)
        if row is None or not auth.get("user_id") or row.get("user") != auth["user_id"]:
            raise SlackReadError("Exact message is missing or is not authored by this bot; nothing imported.")
        refs = [{"file_id": f.get("id"), "filename": f.get("name"), "title": f.get("title"),
                 "mime_type": f.get("mimetype"), "size_bytes": f.get("size")}
                for f in row.get("files", []) if f.get("id")]
        parent = row.get("thread_ts")
        retention = log_delivery(
            response={"ok": True, "ts": row["ts"], "message": {"user": row["user"]}},
            channel=args.channel, text=row.get("text", ""), source="slack_history.exact_import",
            thread_ts=parent if parent != row["ts"] else None, recipient_id=recipient,
            identity=auth, attachments=refs, secrets=(token,),
        )
        return {**base, "sent": False, "imported": retention.get("retained") is True,
                "ts": row["ts"], "retention": retention, "files": refs}
    if args.send:
        result = _call(token, "chat.postMessage", payload, args.timeout_seconds)
        retention = log_delivery(
            response=result, channel=args.channel, text=text, thread_ts=args.thread_ts,
            recipient_id=channel.get("user"), source="slack_history.send", identity=auth, secrets=(token,),
        )
        return {**base, "sent": True, "ts": result.get("ts"), "thread_ts": args.thread_ts, "retention": retention}
    endpoint = "conversations.replies" if args.thread_ts else "conversations.history"
    cursor = args.cursor
    seen = {cursor} if cursor else set()
    messages: list[dict[str, Any]] = []
    for page in range(args.max_pages):
        query: dict[str, Any] = {"channel": args.channel, "limit": args.limit}
        if bounds:
            query.update(oldest=str(bounds[0]), latest=str(bounds[1]), inclusive="true")
        if args.thread_ts:
            query["ts"] = args.thread_ts
        if cursor:
            query["cursor"] = cursor
        result = _call(token, endpoint, query, args.timeout_seconds)
        messages.extend({k: row.get(k) for k in ("ts", "thread_ts", "user", "text", "reply_count")}
                        for row in result.get("messages", []) if not bounds or in_date_window(row, bounds))
        cursor = str((result.get("response_metadata") or {}).get("next_cursor") or "").strip()
        has_more = bool(cursor or result.get("has_more"))
        if not cursor:
            break
        if cursor in seen:
            raise SlackReadError("Slack repeated a pagination cursor; stopped to avoid a loop.")
        seen.add(cursor)
    output = {**base, "thread_ts": args.thread_ts, "messages": messages,
              "pages": page + 1, "has_more": has_more, "next_cursor": cursor}
    if bounds:
        output["date_window"] = {"date": args.date, "timezone": timezone,
                                 "oldest_inclusive": bounds[0], "latest_exclusive": bounds[1]}
        output["coverage"] = "Channel history only; not a complete scan of replies to older threads. Read selected threads separately."
    if has_more and not cursor:
        output["warning"] = "Slack reports more results but supplied no cursor; result is incomplete."
    return output


def main(argv: list[str] | None = None) -> int:
    env_files, argv = extract_env_files(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if env_files:
        load_env_files(env_files, override=True)
    try:
        output = execute(args)
        code = 0
    except SlackReadError as exc:
        output, code = {"ok": False, "error": str(exc)}, 1
    serialized = json.dumps(output, ensure_ascii=False)
    token = slack._cli_slack_bot_token()
    if token:
        serialized = serialized.replace(token, "[REDACTED]")
    print(re.sub(r"xox[a-z]-[A-Za-z0-9-]+", "[REDACTED]", serialized))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
