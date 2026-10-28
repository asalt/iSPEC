"""Slack bot helpers (Socket Mode).

This module is intentionally lightweight so it can run on instrument-adjacent
machines without additional infrastructure. The bot forwards messages to the
iSPEC assistant API (/api/support/chat) and posts replies back to Slack.
"""

from __future__ import annotations

import os
import re
import hashlib
import json
import sys
import mimetypes
import uuid
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from ispec.slack_delivery import log_delivery
from ispec.assistant.slack_inbox import SlackInbox, load_capture_channels
from ispec.assistant.slack_participants import MAX_PARTICIPANTS, USER_MENTION, SlackParticipant

from ispec.assistant.reply_delivery import (
    SlackEventClaims,
    chat_reply_text,
)

from ispec.assistant.slack_tmux_bridge import (
    BRIDGE_AGENT_ID,
    EVENT_SLACK_ARTIFACT_SENT,
    build_artifact_sent_payload,
    stable_json,
)
from ispec.logging import get_logger

logger = get_logger(__name__)

_CHOOSE_RE = re.compile(r"^\s*choose\s+([01])\s*$", re.IGNORECASE)
_ASSISTANT_SLACK_DESTINATIONS_PATH_ENV = "ISPEC_ASSISTANT_SLACK_DESTINATIONS_PATH"
_ASSISTANT_SLACK_DESTINATIONS_JSON_ENVS = (
    "ISPEC_ASSISTANT_SLACK_ALLOWED_DESTINATIONS_JSON",
    "ISPEC_ASSISTANT_SLACK_DESTINATIONS_JSON",
)
_ASSISTANT_SLACK_DESTINATIONS_DEFAULT_FILENAME = "assistant-slack-destinations.local.json"


def register_subcommands(subparsers) -> None:
    from ispec.cli.slack_inbox import register as register_inbox

    register_inbox(subparsers)
    run_parser = subparsers.add_parser("run", help="Run the Slack bot (Socket Mode)")
    run_parser.add_argument(
        "--ispec-server",
        default=os.getenv("ISPEC_SLACK_ISPEC_SERVER") or os.getenv("ISPEC_API_URL") or "",
        help="iSPEC API base URL (default: $ISPEC_SLACK_ISPEC_SERVER or $ISPEC_API_URL)",
    )
    run_parser.add_argument(
        "--api-key",
        default=os.getenv("ISPEC_SLACK_API_KEY") or os.getenv("ISPEC_API_KEY") or "",
        help="Optional iSPEC API key (default: $ISPEC_SLACK_API_KEY or $ISPEC_API_KEY)",
    )
    run_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout for iSPEC calls (default: 60)",
    )

    send_parser = subparsers.add_parser(
        "send",
        help="Send one explicit Slack text message using the configured bot token",
    )
    send_parser.add_argument(
        "--to",
        default=os.getenv("ISPEC_SLACK_DEFAULT_RECIPIENT") or "",
        help=(
            "Recipient alias, channel id, user id, or email. Alias resolution reads "
            "ISPEC_SLACK_RECIPIENTS_JSON / ISPEC_SLACK_DM_ALIASES_JSON and "
            "ISPEC_SLACK_DM_<ALIAS>_{CHANNEL,USER_ID,EMAIL}."
        ),
    )
    send_parser.add_argument("--channel", help="Slack channel/DM id to post to.")
    send_parser.add_argument("--user-id", help="Slack user id; opens a DM before posting.")
    send_parser.add_argument("--email", help="Slack user email; resolves to a DM before posting.")
    send_parser.add_argument("--thread-ts", help="Optional Slack thread timestamp.")
    send_parser.add_argument("--text", help="Message text. If omitted, positional text or --stdin is used.")
    send_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read message text from stdin.",
    )
    send_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the recipient and print the payload without calling Slack.",
    )
    send_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("ISPEC_SLACK_TIMEOUT_SECONDS") or 10),
        help="Slack HTTP timeout seconds (default: $ISPEC_SLACK_TIMEOUT_SECONDS or 10).",
    )
    send_parser.add_argument("message", nargs="*", help="Message text when --text is omitted.")

    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload one explicit local file to Slack using the configured bot token",
    )
    upload_parser.add_argument(
        "--to",
        default=os.getenv("ISPEC_SLACK_DEFAULT_RECIPIENT") or "",
        help="Recipient alias, channel id, user id, or email.",
    )
    upload_parser.add_argument("--channel", help="Slack channel/DM id to upload into.")
    upload_parser.add_argument("--user-id", help="Slack user id; opens a DM before upload.")
    upload_parser.add_argument("--email", help="Slack user email; resolves to a DM before upload.")
    upload_parser.add_argument("--thread-ts", help="Optional Slack thread timestamp.")
    upload_parser.add_argument("--file", required=True, help="Exact local file path to upload.")
    upload_parser.add_argument("--title", help="Optional Slack file title.")
    upload_parser.add_argument("--text", help="Optional initial Slack message/comment.")
    upload_parser.add_argument("--alt-txt", help="Optional accessibility text for images.")
    upload_parser.add_argument(
        "--record-artifact-receipt",
        action="store_true",
        help="After a successful upload, record a Slack artifact receipt in the iSPEC agent event log.",
    )
    upload_parser.add_argument("--artifact-id", help="Optional stable artifact id for receipt logging.")
    upload_parser.add_argument("--origin-tmux-target", help="Human-friendly origin tmux target/alias.")
    upload_parser.add_argument("--origin-tmux-pane-id", help="Exact origin tmux pane id, such as %1.")
    upload_parser.add_argument("--origin-tmux-capture-target", help="Exact tmux capture/send target for receipt provenance.")
    upload_parser.add_argument("--origin-tmux-allowlist-match", help="Allowlist entry that made the origin pane eligible.")
    upload_parser.add_argument("--receipt-note", help="Optional receipt note/context.")
    upload_parser.add_argument(
        "--submit-allowed",
        action="store_true",
        help="Mark this artifact's origin pane as allowing Enter/C-m when a reply is explicitly relayed.",
    )
    upload_parser.add_argument(
        "--ispec-server",
        default=os.getenv("ISPEC_SLACK_ISPEC_SERVER") or os.getenv("ISPEC_API_URL") or "",
        help="iSPEC API base URL for receipt logging.",
    )
    upload_parser.add_argument(
        "--api-key",
        default=os.getenv("ISPEC_SLACK_API_KEY") or os.getenv("ISPEC_API_KEY") or "",
        help="Optional iSPEC API key for receipt logging.",
    )
    upload_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve recipient and file metadata without uploading bytes.",
    )
    upload_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("ISPEC_SLACK_TIMEOUT_SECONDS") or 30),
        help="Slack HTTP timeout seconds (default: $ISPEC_SLACK_TIMEOUT_SECONDS or 30).",
    )


def dispatch(args) -> None:
    if args.subcommand == "inbox":
        from ispec.cli.slack_inbox import dispatch as dispatch_inbox

        dispatch_inbox(args)
        return
    if args.subcommand == "run":
        _run_socket_mode(args)
        return
    if args.subcommand == "send":
        _run_send(args)
        return
    if args.subcommand == "upload":
        _run_upload(args)
        return
    raise SystemExit(f"Unknown slack subcommand: {args.subcommand}")


@dataclass
class _SlackConfig:
    bot_token: str
    app_token: str
    ispec_server: str
    api_key: str
    timeout_seconds: int


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _clean_slack_text(text: str, *, bot_user_id: str | None) -> str:
    raw = _unescape_slack_text(text)
    if not raw:
        return ""
    if bot_user_id:
        raw = raw.replace(f"<@{bot_user_id}>", "")
    return raw.strip()


def _unescape_slack_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    return raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


def _session_id(*, team_id: str | None, channel: str, thread_ts: str) -> str:
    safe_team = team_id or "unknown"
    return f"slack:{safe_team}:{channel}:{thread_ts}"


def _dm_day_bucket(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    else:
        current = current.astimezone(UTC)
    return current.strftime("%Y%m%d")


def _session_id_for_dm(*, team_id: str | None, channel: str, now: datetime | None = None) -> str:
    safe_team = team_id or "unknown"
    # Do not reuse older daily sessions that mixed root messages with threads.
    return f"slack:{safe_team}:{channel}:dm24:{_dm_day_bucket(now)}:root"


def _pending_key(*, channel: str, thread_ts: str | None) -> str:
    return f"{channel}:{thread_ts or 'root'}"


def _reply_thread_ts(*, channel_type: str | None, thread_ts: str | None) -> str | None:
    if thread_ts:
        return thread_ts
    if channel_type in {"im", "mpim"}:
        return None
    return None


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _normalize_slack_speaker(value: str | None) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return ""
    return text[:64]


def _slack_user_summary(
    *,
    client: Any,
    user_cache: dict[str, dict[str, str]],
    user_id: str,
) -> dict[str, str]:
    user_id = (user_id or "").strip()
    if not user_id:
        return {}

    cached = user_cache.get(user_id)
    if isinstance(cached, dict) and cached:
        return cached

    summary: dict[str, str] = {"user_id": user_id}
    try:
        info = client.users_info(user=user_id)
        user_obj = info.get("user")  # SlackResponse supports .get(), but is not a dict.
        if isinstance(user_obj, dict):
            name = str(user_obj.get("name") or "").strip()
            if name:
                summary["user_name"] = name

            profile = user_obj.get("profile") if isinstance(user_obj.get("profile"), dict) else {}
            display = (
                str(profile.get("display_name") or "").strip()
                or str(profile.get("display_name_normalized") or "").strip()
            )
            real_name = (
                str(profile.get("real_name") or "").strip()
                or str(profile.get("real_name_normalized") or "").strip()
                or str(user_obj.get("real_name") or "").strip()
            )
            if display:
                summary["user_display_name"] = display
            if real_name:
                summary["user_real_name"] = real_name
    except Exception:
        pass

    user_cache[user_id] = summary
    return summary


def _slack_speaker_label(summary: dict[str, str] | None) -> str:
    if not isinstance(summary, dict):
        return ""
    return _normalize_slack_speaker(
        str(summary.get("user_display_name") or "").strip()
        or str(summary.get("user_real_name") or "").strip()
        or str(summary.get("user_name") or "").strip()
    )


def _format_message_for_ispec(*, text: str, slack_user: dict[str, str] | None) -> str:
    speaker = _slack_speaker_label(slack_user)
    return f"[{speaker}] {text}" if speaker else text


def _slack_participant_map(*, client, user_cache, meta, text) -> list[dict[str, str]]:
    """Resolve current/thread authors and actual mentions; no workspace-wide fetch."""
    ids = [str(meta.get("slack", {}).get("user_id") or "")]
    ids.extend(USER_MENTION.findall(text))
    thread = meta.get("slack_thread_context") or {}
    for row in thread.get("messages", []):
        ids.append(str(row.get("user_id") or ""))
        ids.extend(USER_MENTION.findall(str(row.get("text") or "")))
    # Plain names can reuse profiles already encountered in this receiver process.
    for user_id, person in list(user_cache.items()):
        names = (person.get(key) for key in ("user_name", "user_display_name", "user_real_name"))
        if any(name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I) for name in names):
            ids.append(user_id)
    participants = []
    valid_ids = (uid for uid in dict.fromkeys(ids) if re.fullmatch(r"[UW][A-Z0-9]{1,79}", uid))
    for user_id in list(valid_ids)[:MAX_PARTICIPANTS]:
        summary = _slack_user_summary(client=client, user_cache=user_cache, user_id=user_id)
        participant = SlackParticipant(**{
            key: str(summary.get(key) or "")[:100 if key != "user_id" else 80]
            for key in SlackParticipant.model_fields
        })
        participants.append(participant.model_dump())
    return participants


def _post_ispec_chat(
    *,
    server: str,
    api_key: str,
    session_id: str,
    message: str,
    meta: dict[str, Any] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    meta = dict(meta or {})
    meta.pop("slack_outbound_context", None)
    scope = meta.get("slack")
    if isinstance(scope, dict) and api_key:
        try:
            context_response = requests.post(
                server.rstrip("/") + "/api/agents/slack/outbound-context",
                json=scope, headers=_headers(api_key), timeout=min(5, max(1, timeout_seconds)),
            )
            context_response.raise_for_status()
            meta["slack_outbound_context"] = context_response.json().get("messages", [])
            meta["slack_outbound_context_status"] = "loaded"
        except Exception as exc:
            meta["slack_outbound_context_status"] = "unavailable"
            logger.warning("Slack outbound context unavailable error_type=%s", type(exc).__name__)
    url = server.rstrip("/") + "/api/support/chat"
    resp = requests.post(
        url,
        json={"sessionId": session_id, "message": message, "history": [], "meta": meta},
        headers=_headers(api_key),
        timeout=max(1, int(timeout_seconds)),
    )
    resp.raise_for_status()
    return resp.json()


def _post_ispec_choose(
    *,
    server: str,
    api_key: str,
    session_id: str,
    user_message_id: int,
    choice_index: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = server.rstrip("/") + "/api/support/choose"
    resp = requests.post(
        url,
        json={
            "sessionId": session_id,
            "userMessageId": int(user_message_id),
            "choiceIndex": int(choice_index),
        },
        headers=_headers(api_key),
        timeout=max(1, int(timeout_seconds)),
    )
    resp.raise_for_status()
    return resp.json()


def _post_ispec_events(
    *,
    server: str,
    api_key: str,
    events: list[dict[str, Any]],
    timeout_seconds: int,
) -> dict[str, Any]:
    url = server.rstrip("/") + "/api/agents/events"
    resp = requests.post(
        url,
        json=events,
        headers=_headers(api_key),
        timeout=max(1, int(timeout_seconds)),
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


def _post_ispec_slack_artifact_reply(
    *,
    server: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    url = server.rstrip("/") + "/api/agents/slack/artifact-replies"
    resp = requests.post(
        url,
        json=payload,
        headers=_headers(api_key),
        timeout=max(1, int(timeout_seconds)),
    )
    resp.raise_for_status()
    parsed = resp.json()
    return parsed if isinstance(parsed, dict) else {}


def _format_compare_choices(compare: dict[str, Any]) -> str:
    choices = compare.get("choices") if isinstance(compare.get("choices"), list) else []
    lines = [
        "I have two draft replies. Reply in this thread with `choose 0` or `choose 1`.\n"
    ]
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        idx = choice.get("index")
        msg = str(choice.get("message") or "").strip()
        if msg:
            lines.append(f"*Option {idx}*\n{msg}\n")
    return "\n".join(lines).strip()


def _slack_response_field(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    try:
        getter = getattr(response, "get", None)
        if callable(getter):
            return getter(key)
    except Exception:
        pass
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return data.get(key)
    return None


def _slack_api_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("error") or "").strip()
    try:
        getter = getattr(response, "get", None)
        if callable(getter):
            return str(getter("error") or "").strip()
    except Exception:
        pass
    return ""


def _cli_slack_bot_token() -> str:
    return (os.getenv("ISPEC_SLACK_BOT_TOKEN") or os.getenv("SLACK_BOT_TOKEN") or "").strip()


def _slack_direct_api_call(
    *,
    token: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    as_form: bool = False,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    request_kwargs: dict[str, Any]
    if as_form:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        request_kwargs = {"data": payload}
    else:
        headers["Content-Type"] = "application/json; charset=utf-8"
        request_kwargs = {"json": payload}
    resp = requests.post(
        f"https://slack.com/api/{endpoint.lstrip('/')}",
        headers=headers,
        **request_kwargs,
        timeout=max(1.0, float(timeout_seconds)),
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _slack_alias_key(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def _load_slack_alias_maps() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for env_name in ("ISPEC_SLACK_RECIPIENTS_JSON", "ISPEC_SLACK_DM_ALIASES_JSON"):
        raw = (os.getenv(env_name) or "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            logger.warning("Ignoring invalid %s", env_name)
            continue
        if not isinstance(parsed, dict):
            logger.warning("Ignoring non-object %s", env_name)
            continue
        for key, value in parsed.items():
            alias = _slack_alias_key(str(key))
            if alias:
                merged[alias] = value
    return merged


def _recipient_from_string(value: str | None) -> dict[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return {}
    if raw[0] in {"C", "G", "D"} and len(raw) >= 6:
        return {"channel": raw}
    if raw[0] in {"U", "W"} and len(raw) >= 6:
        return {"user_id": raw}
    if "@" in raw and "." in raw:
        return {"email": raw}
    return {}


def _recipient_from_alias_env(alias: str) -> dict[str, str]:
    alias_key = _slack_alias_key(alias)
    if not alias_key:
        return {}

    aliases = _load_slack_alias_maps()
    mapped = aliases.get(alias_key)
    if isinstance(mapped, str):
        direct = _recipient_from_string(mapped)
        if direct:
            return direct
    elif isinstance(mapped, dict):
        for field in ("channel", "user_id", "email"):
            value = str(mapped.get(field) or "").strip()
            if value:
                return {field: value}

    env_stems = [f"ISPEC_SLACK_DM_{alias_key.upper()}"]
    if alias_key == "me":
        env_stems.append("ISPEC_SLACK_ME")
    for stem in env_stems:
        for suffix, field in (
            ("CHANNEL", "channel"),
            ("USER_ID", "user_id"),
            ("EMAIL", "email"),
        ):
            value = (os.getenv(f"{stem}_{suffix}") or "").strip()
            if value:
                return {field: value}

    assistant_destination = _recipient_from_assistant_destination(alias_key)
    if assistant_destination:
        return assistant_destination

    return _recipient_from_string(alias)


def _recipient_from_assistant_destination(alias_key: str) -> dict[str, str]:
    if not alias_key:
        return {}

    for parsed in _load_assistant_slack_destination_mappings():
        if isinstance(parsed, dict) and isinstance(parsed.get("destinations"), dict):
            parsed = parsed["destinations"]
        if not isinstance(parsed, dict):
            continue
        entry = parsed.get(alias_key)
        if entry is None:
            for raw_key, raw_value in parsed.items():
                if _slack_alias_key(str(raw_key)) == alias_key:
                    entry = raw_value
                    break
        if entry is None:
            continue
        if isinstance(entry, str):
            direct = _recipient_from_string(entry)
            if direct:
                return direct
        if isinstance(entry, dict):
            for field in ("channel", "user_id", "user", "email"):
                value = str(entry.get(field) or "").strip()
                if value:
                    return {"user_id" if field == "user" else field: value}

    return {}


def _load_assistant_slack_destination_mappings() -> list[Any]:
    mappings: list[Any] = []
    for env_name in _ASSISTANT_SLACK_DESTINATIONS_JSON_ENVS:
        raw = (os.getenv(env_name) or "").strip()
        if not raw:
            continue
        try:
            mappings.append(json.loads(raw))
        except Exception:
            logger.warning("Ignoring invalid %s", env_name)

    path = _assistant_slack_destinations_path()
    if path is not None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            raw = ""
        if raw:
            try:
                mappings.append(json.loads(raw))
            except Exception:
                logger.warning("Ignoring invalid %s=%s", _ASSISTANT_SLACK_DESTINATIONS_PATH_ENV, path)

    return mappings


def _assistant_slack_destinations_path() -> Path | None:
    raw = (os.getenv(_ASSISTANT_SLACK_DESTINATIONS_PATH_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()

    seen: set[Path] = set()
    bases: list[Path] = []
    try:
        cwd = Path.cwd().resolve()
        bases.extend([cwd, *cwd.parents])
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        bases.extend([here.parent, *here.parents])
    except Exception:
        pass

    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        candidate = base / "configs" / _ASSISTANT_SLACK_DESTINATIONS_DEFAULT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _resolve_slack_send_channel(
    *,
    token: str,
    channel: str | None,
    user_id: str | None,
    email: str | None,
    recipient: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    target: dict[str, str] = {}
    if channel:
        target = {"channel": str(channel).strip()}
    elif user_id:
        target = {"user_id": str(user_id).strip()}
    elif email:
        target = {"email": str(email).strip()}
    else:
        target = _recipient_from_alias_env(str(recipient or "").strip())

    if target.get("channel"):
        return {"channel": target["channel"], "resolution": "channel"}

    resolved_user_id = str(target.get("user_id") or "").strip()
    if not resolved_user_id and target.get("email"):
        lookup = _slack_direct_api_call(
            token=token,
            endpoint="users.lookupByEmail",
            payload={"email": target["email"]},
            timeout_seconds=timeout_seconds,
        )
        if lookup.get("ok") is not True:
            error = str(lookup.get("error") or "unknown_error")
            raise SystemExit(f"Slack users.lookupByEmail failed: {error}")
        user = lookup.get("user")
        if isinstance(user, dict):
            resolved_user_id = str(user.get("id") or "").strip()

    if resolved_user_id:
        opened = _slack_direct_api_call(
            token=token,
            endpoint="conversations.open",
            payload={"users": resolved_user_id},
            timeout_seconds=timeout_seconds,
        )
        if opened.get("ok") is not True:
            error = str(opened.get("error") or "unknown_error")
            raise SystemExit(f"Slack conversations.open failed: {error}")
        channel_obj = opened.get("channel")
        dm_channel = ""
        if isinstance(channel_obj, dict):
            dm_channel = str(channel_obj.get("id") or "").strip()
        if not dm_channel:
            raise SystemExit("Slack conversations.open did not return a channel id.")
        return {
            "channel": dm_channel,
            "resolution": "dm",
            "user_id": resolved_user_id,
        }

    raise SystemExit(
        "Could not resolve Slack recipient. Pass --channel, --user-id, --email, "
        "or configure ISPEC_SLACK_RECIPIENTS_JSON / ISPEC_SLACK_DM_<ALIAS>_*."
    )


def send_slack_text(
    *,
    text: str,
    channel: str | None = None,
    user_id: str | None = None,
    email: str | None = None,
    recipient: str | None = None,
    thread_ts: str | None = None,
    timeout_seconds: float = 10.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    token = _cli_slack_bot_token()
    if not token:
        raise SystemExit("Missing ISPEC_SLACK_BOT_TOKEN/SLACK_BOT_TOKEN.")

    message_text = str(text or "").strip()
    if not message_text:
        raise SystemExit("Missing Slack message text.")

    resolved = _resolve_slack_send_channel(
        token=token,
        channel=channel,
        user_id=user_id,
        email=email,
        recipient=recipient,
        timeout_seconds=timeout_seconds,
    )
    post_payload: dict[str, Any] = {
        "channel": resolved["channel"],
        "text": message_text,
    }
    if thread_ts:
        post_payload["thread_ts"] = str(thread_ts).strip()

    if dry_run:
        return {"ok": True, "dry_run": True, "resolved": resolved, "payload": post_payload}

    response = _slack_direct_api_call(
        token=token,
        endpoint="chat.postMessage",
        payload=post_payload,
        timeout_seconds=timeout_seconds,
    )
    if response.get("ok") is not True:
        error = str(response.get("error") or "").strip() or "unknown_error"
        if error == "not_in_channel" and str(post_payload["channel"])[0] in {"C", "G"}:
            join_response = _slack_direct_api_call(
                token=token,
                endpoint="conversations.join",
                payload={"channel": post_payload["channel"]},
                timeout_seconds=timeout_seconds,
            )
            if join_response.get("ok") is True:
                response = _slack_direct_api_call(
                    token=token,
                    endpoint="chat.postMessage",
                    payload=post_payload,
                    timeout_seconds=timeout_seconds,
                )
                error = str(response.get("error") or "").strip() or error
        if response.get("ok") is not True:
            return {"ok": False, "error": error, "resolved": resolved, "slack": response}

    retention = log_delivery(
        response=response, channel=resolved["channel"], text=message_text,
        thread_ts=thread_ts, recipient_id=resolved.get("user_id"), source="slack_cli.send",
        lookup_identity=lambda: _slack_direct_api_call(token=token, endpoint="auth.test", payload={}, timeout_seconds=5),
        secrets=(token,),
    )
    return {"ok": True, "resolved": resolved, "slack": response, "retention": retention}


def _slack_upload_max_bytes() -> int:
    raw = (os.getenv("ISPEC_SLACK_UPLOAD_MAX_BYTES") or "").strip()
    if not raw:
        return 50 * 1024 * 1024
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 50 * 1024 * 1024


def _resolve_upload_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise SystemExit(f"Slack upload file does not exist: {file_path}")
    if not file_path.is_file():
        raise SystemExit(f"Slack upload path is not a file: {file_path}")
    if file_path.name.startswith(".env"):
        raise SystemExit("Refusing to upload env files.")
    size_bytes = file_path.stat().st_size
    max_bytes = _slack_upload_max_bytes()
    if size_bytes > max_bytes:
        raise SystemExit(
            f"Refusing to upload {size_bytes} bytes; limit is {max_bytes} bytes "
            "(set ISPEC_SLACK_UPLOAD_MAX_BYTES to override)."
        )
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return {
        "path": str(file_path),
        "filename": file_path.name,
        "size_bytes": int(size_bytes),
        "mime_type": mime_type,
    }


def upload_slack_file(
    *,
    file_path: str | Path,
    channel: str | None = None,
    user_id: str | None = None,
    email: str | None = None,
    recipient: str | None = None,
    thread_ts: str | None = None,
    text: str | None = None,
    title: str | None = None,
    alt_txt: str | None = None,
    timeout_seconds: float = 30.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    token = _cli_slack_bot_token()
    if not token:
        raise SystemExit("Missing ISPEC_SLACK_BOT_TOKEN/SLACK_BOT_TOKEN.")

    file_info = _resolve_upload_file(file_path)
    resolved = _resolve_slack_send_channel(
        token=token,
        channel=channel,
        user_id=user_id,
        email=email,
        recipient=recipient,
        timeout_seconds=timeout_seconds,
    )
    upload_request: dict[str, Any] = {
        "filename": file_info["filename"],
        "length": file_info["size_bytes"],
    }
    if alt_txt:
        upload_request["alt_txt"] = str(alt_txt).strip()

    complete_payload: dict[str, Any] = {
        "files": [
            {
                "id": "<file_id>",
                "title": str(title or file_info["filename"]).strip() or file_info["filename"],
            }
        ],
        "channel_id": resolved["channel"],
    }
    if thread_ts:
        complete_payload["thread_ts"] = str(thread_ts).strip()
    if text:
        complete_payload["initial_comment"] = str(text).strip()

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "resolved": resolved,
            "file": file_info,
            "upload_request": upload_request,
            "complete_payload": complete_payload,
        }

    upload_url_response = _slack_direct_api_call(
        token=token,
        endpoint="files.getUploadURLExternal",
        payload=upload_request,
        timeout_seconds=timeout_seconds,
        as_form=True,
    )
    if upload_url_response.get("ok") is not True:
        error = str(upload_url_response.get("error") or "").strip() or "unknown_error"
        return {
            "ok": False,
            "error": error,
            "stage": "get_upload_url",
            "resolved": resolved,
            "file": file_info,
            "slack": upload_url_response,
        }

    upload_url = str(upload_url_response.get("upload_url") or "").strip()
    file_id = str(upload_url_response.get("file_id") or "").strip()
    if not upload_url or not file_id:
        return {
            "ok": False,
            "error": "missing_upload_url_or_file_id",
            "stage": "get_upload_url",
            "resolved": resolved,
            "file": file_info,
            "slack": upload_url_response,
        }

    with Path(file_info["path"]).open("rb") as handle:
        upload_response = requests.post(
            upload_url,
            files={"file": (file_info["filename"], handle, file_info["mime_type"])},
            timeout=max(1.0, float(timeout_seconds)),
        )
    if upload_response.status_code < 200 or upload_response.status_code >= 300:
        return {
            "ok": False,
            "error": f"upload_http_{upload_response.status_code}",
            "stage": "upload_bytes",
            "resolved": resolved,
            "file": file_info,
            "body": upload_response.text[:500],
        }

    complete_payload["files"][0]["id"] = file_id
    complete_response = _slack_direct_api_call(
        token=token,
        endpoint="files.completeUploadExternal",
        payload=complete_payload,
        timeout_seconds=timeout_seconds,
    )
    if complete_response.get("ok") is not True:
        error = str(complete_response.get("error") or "").strip() or "unknown_error"
        return {
            "ok": False,
            "error": error,
            "stage": "complete_upload",
            "resolved": resolved,
            "file": file_info,
            "slack": complete_response,
        }

    retention = log_delivery(
        response=complete_response, channel=resolved["channel"], text=text,
        thread_ts=thread_ts, recipient_id=resolved.get("user_id"), source="slack_cli.upload",
        attachments=[{**file_info, "file_id": file_id, "title": title}],
        lookup_identity=lambda: _slack_direct_api_call(token=token, endpoint="auth.test", payload={}, timeout_seconds=5),
        secrets=(token,),
    )
    return {"ok": True, "resolved": resolved, "file": file_info,
            "file_id": file_id, "slack": complete_response, "retention": retention}


def _message_text_from_args(args) -> str:
    if getattr(args, "stdin", False):
        return sys.stdin.read().strip()
    text = str(getattr(args, "text", "") or "").strip()
    if text:
        return text
    parts = getattr(args, "message", None) or []
    return " ".join(str(part) for part in parts).strip()


def _run_send(args) -> None:
    result = send_slack_text(
        text=_message_text_from_args(args),
        channel=getattr(args, "channel", None),
        user_id=getattr(args, "user_id", None),
        email=getattr(args, "email", None),
        recipient=getattr(args, "to", None),
        thread_ts=getattr(args, "thread_ts", None),
        timeout_seconds=float(getattr(args, "timeout_seconds", 10.0)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def _run_upload(args) -> None:
    result = upload_slack_file(
        file_path=getattr(args, "file"),
        channel=getattr(args, "channel", None),
        user_id=getattr(args, "user_id", None),
        email=getattr(args, "email", None),
        recipient=getattr(args, "to", None),
        thread_ts=getattr(args, "thread_ts", None),
        text=getattr(args, "text", None),
        title=getattr(args, "title", None),
        alt_txt=getattr(args, "alt_txt", None),
        timeout_seconds=float(getattr(args, "timeout_seconds", 30.0)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    if bool(getattr(args, "record_artifact_receipt", False)):
        artifact_id = str(getattr(args, "artifact_id", "") or "").strip() or str(uuid.uuid4())
        origin_tmux = {
            "target": getattr(args, "origin_tmux_target", None),
            "preferred_alias": getattr(args, "origin_tmux_target", None),
            "pane_id": getattr(args, "origin_tmux_pane_id", None),
            "capture_target": getattr(args, "origin_tmux_capture_target", None),
            "allowlist_match": getattr(args, "origin_tmux_allowlist_match", None),
        }
        origin_tmux = {key: value for key, value in origin_tmux.items() if value}
        receipt_payload = build_artifact_sent_payload(
            upload_result=result,
            artifact_id=artifact_id,
            origin_tmux=origin_tmux,
            thread_ts=getattr(args, "thread_ts", None),
            submit_allowed=bool(getattr(args, "submit_allowed", False)),
            note=getattr(args, "receipt_note", None),
        )
        result["artifact_receipt"] = {
            "artifact_id": artifact_id,
            "payload": receipt_payload,
            "posted": False,
        }
        server = str(getattr(args, "ispec_server", "") or "").strip()
        if not server:
            result["artifact_receipt"]["error"] = "missing_ispec_server"
        elif result.get("ok") is True and not bool(getattr(args, "dry_run", False)):
            event = {
                "type": EVENT_SLACK_ARTIFACT_SENT,
                "agent_id": BRIDGE_AGENT_ID,
                "ts": datetime.now(UTC).isoformat(),
                "name": "slack_artifact_sent",
                "severity": "info",
                "correlation_id": artifact_id,
                "dimensions": {
                    "artifact_id": artifact_id,
                    "channel": receipt_payload.get("slack", {}).get("channel") if isinstance(receipt_payload.get("slack"), dict) else None,
                    "thread_ts": receipt_payload.get("slack", {}).get("thread_ts") if isinstance(receipt_payload.get("slack"), dict) else None,
                },
                "value": receipt_payload,
            }
            try:
                posted = _post_ispec_events(
                    server=server,
                    api_key=str(getattr(args, "api_key", "") or ""),
                    events=[event],
                    timeout_seconds=int(float(getattr(args, "timeout_seconds", 30.0))),
                )
                result["artifact_receipt"]["posted"] = True
                result["artifact_receipt"]["ingest"] = posted
            except Exception as exc:
                result["artifact_receipt"]["error"] = str(exc)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def _safe_slack_post_message(*, client: Any, channel: str, thread_ts: str | None, text: str) -> str | None:
    kwargs: dict[str, Any] = {"channel": channel, "text": text}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    try:
        response = client.chat_postMessage(**kwargs)
        ts = str(_slack_response_field(response, "ts") or "").strip()
        return ts or None
    except Exception as exc:
        error = _slack_api_error_code(exc)
        logger.exception("Slack chat_postMessage failed (%s) channel=%s thread_ts=%s", error, channel, thread_ts)
        if error != "not_in_channel":
            return None
        try:
            client.conversations_join(channel=channel)
        except Exception:
            logger.exception("Slack join failed channel=%s", channel)
            return None
        try:
            response = client.chat_postMessage(**kwargs)
            ts = str(_slack_response_field(response, "ts") or "").strip()
            return ts or None
        except Exception:
            logger.exception("Slack chat_postMessage retry failed channel=%s thread_ts=%s", channel, thread_ts)
            return None


def _safe_slack_update_message(*, client: Any, channel: str, message_ts: str, text: str) -> bool:
    if not message_ts:
        return False
    try:
        client.chat_update(channel=channel, ts=message_ts, text=text)
        return True
    except Exception as exc:
        error = _slack_api_error_code(exc)
        logger.exception("Slack chat_update failed (%s) channel=%s ts=%s", error, channel, message_ts)
        return False


def _ack_ignored_slack_event(*, event_name: str, event: dict[str, Any] | None, ack: Any, logger: Any) -> None:
    try:
        ack()
    except Exception:
        pass
    try:
        payload = event if isinstance(event, dict) else {}
        logger.debug(
            "Slack ignored event event=%s type=%s subtype=%s user=%s channel=%s",
            event_name,
            payload.get("type"),
            payload.get("subtype"),
            payload.get("user"),
            payload.get("channel"),
        )
    except Exception:
        pass


def _reply_user_allowed(event: dict[str, Any]) -> bool:
    """Optional receiver allowlist; empty preserves existing deployment behavior."""
    raw = os.getenv("ISPEC_SLACK_ALLOWED_USER_IDS", "").strip()
    if not raw:
        return True
    allowed = {value.strip() for value in raw.split(",") if value.strip()}
    return str(event.get("user") or "") in allowed


def _run_socket_mode(args) -> None:
    try:
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "slack-bolt is required for `ispec slack run`. Install with: pip install slack-bolt"
        ) from exc

    try:
        from slack_sdk.errors import SlackApiError
    except Exception:  # pragma: no cover
        SlackApiError = Exception  # type: ignore[misc,assignment]

    cfg = _SlackConfig(
        bot_token=_require_env("SLACK_BOT_TOKEN"),
        app_token=_require_env("SLACK_APP_TOKEN"),
        ispec_server=str(args.ispec_server or "").strip(),
        api_key=str(args.api_key or "").strip(),
        timeout_seconds=max(1, int(args.timeout_seconds)),
    )
    if not cfg.ispec_server:
        raise SystemExit("--ispec-server is required (or set ISPEC_SLACK_ISPEC_SERVER)")

    pending_compare: dict[str, dict[str, Any]] = {}
    user_cache: dict[str, dict[str, str]] = {}
    event_claims = SlackEventClaims()
    inbox = SlackInbox(channels=load_capture_channels(os.getenv("ISPEC_SLACK_INBOX_CONFIG")))
    logger.info("Slack durable inbox ready passive_channels=%s", len(inbox.channels))

    app = App(token=cfg.bot_token)
    app.middleware(inbox.middleware)

    api_key_fingerprint = ""
    if cfg.api_key:
        api_key_fingerprint = hashlib.sha256(cfg.api_key.encode("utf-8")).hexdigest()[:10]
    logger.info(
        "Slack bot config ispec_server=%s api_key_present=%s api_key_fingerprint=%s timeout_seconds=%s",
        cfg.ispec_server,
        bool(cfg.api_key),
        api_key_fingerprint or "none",
        cfg.timeout_seconds,
    )

    # Best-effort token sanity check so failures show up immediately in logs.
    try:
        auth = app.client.auth_test()
        logger.info(
            "Slack auth_test ok team=%s team_id=%s bot_user_id=%s",
            auth.get("team"),
            auth.get("team_id"),
            auth.get("user_id"),
        )
    except Exception:
        logger.exception("Slack auth_test failed (check SLACK_BOT_TOKEN)")

    def _safe_say(*, say, client, channel: str, thread_ts: str | None, text: str) -> None:
        try:
            if thread_ts:
                say(text=text, thread_ts=thread_ts)
            else:
                say(text=text)
            return
        except SlackApiError as exc:
            error = ""
            try:
                error = str(exc.response.get("error") or "")
            except Exception:
                error = ""
            logger.exception("Slack say failed (%s) channel=%s thread_ts=%s", error, channel, thread_ts)

            if error == "not_in_channel":
                try:
                    client.conversations_join(channel=channel)
                except Exception:
                    logger.exception("Slack join failed channel=%s", channel)
                    return
                try:
                    if thread_ts:
                        say(text=text, thread_ts=thread_ts)
                    else:
                        say(text=text)
                except Exception:
                    logger.exception("Slack say retry failed channel=%s thread_ts=%s", channel, thread_ts)

    def _send_or_update(*, say, client, channel: str, thread_ts: str | None, text: str, pending_ts: str | None) -> None:
        if pending_ts and _safe_slack_update_message(
            client=client,
            channel=channel,
            message_ts=pending_ts,
            text=text,
        ):
            return
        _safe_say(say=say, client=client, channel=channel, thread_ts=thread_ts, text=text)

    @app.error
    def _on_slack_error(error, body, logger):  # type: ignore[no-untyped-def]
        try:
            logger.error("Slack handler failed error_type=%s", type(error).__name__)
        except Exception:
            logger.error("Slack handler failed")

    @app.event("app_home_opened")
    def _on_app_home_opened(event, body, logger, ack):  # type: ignore[no-untyped-def]
        # Slack fires this when a user opens the bot's App Home. We don't
        # currently publish a Home tab view, but we still register a handler so
        # Bolt doesn't warn about unhandled events.
        try:
            ack()
        except Exception:
            pass
        try:
            logger.info(
                "Slack app_home_opened user=%s tab=%s",
                event.get("user"),
                event.get("tab"),
            )
        except Exception:
            logger.info("Slack app_home_opened")

    @app.event("file_created")
    def _on_file_created(event, logger, ack):  # type: ignore[no-untyped-def]
        _ack_ignored_slack_event(event_name="file_created", event=event, ack=ack, logger=logger)

    @app.event("file_shared")
    def _on_file_shared(event, logger, ack):  # type: ignore[no-untyped-def]
        _ack_ignored_slack_event(event_name="file_shared", event=event, ack=ack, logger=logger)

    @app.event("file_deleted")
    def _on_file_deleted(event, logger, ack):  # type: ignore[no-untyped-def]
        _ack_ignored_slack_event(event_name="file_deleted", event=event, ack=ack, logger=logger)

    @app.event("reaction_added")
    def _on_reaction_added(event, logger, ack):  # type: ignore[no-untyped-def]
        _ack_ignored_slack_event(event_name="reaction_added", event=event, ack=ack, logger=logger)

    @app.event("reaction_removed")
    def _on_reaction_removed(event, logger, ack):  # type: ignore[no-untyped-def]
        _ack_ignored_slack_event(event_name="reaction_removed", event=event, ack=ack, logger=logger)

    @app.event("app_mention")
    @inbox.handler
    def _on_mention(event, say, context, client, ack):  # type: ignore[no-untyped-def]
        try:
            ack()
        except Exception:
            pass
        if not _reply_user_allowed(event):
            return
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        channel = str(event.get("channel") or "")
        if not channel:
            return

        thread_ts = str(event.get("thread_ts") or event.get("ts") or "")
        if not thread_ts:
            return

        team_id = context.get("team_id")
        if event.get("channel_type") in {"im", "mpim"} and not event.get("thread_ts"):
            session_id = _session_id_for_dm(team_id=team_id, channel=channel)
            thread_ts = None
        else:
            session_id = _session_id(team_id=team_id, channel=channel, thread_ts=thread_ts)

        text = _clean_slack_text(str(event.get("text") or ""), bot_user_id=context.get("bot_user_id"))
        if not inbox.claim(context, fallback=event_claims, team_id=team_id, event=event, route="reply"):
            return
        if not text:
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                text="Say something after mentioning me.",
            )
            return

        slack_user_id = str(event.get("user") or "").strip()

        slack_user = _slack_user_summary(client=client, user_cache=user_cache, user_id=slack_user_id)
        message_for_ispec = _format_message_for_ispec(text=text, slack_user=slack_user)
        meta: dict[str, Any] = {
            "source": "slack",
            "slack_inbox_id": context.get("slack_inbox_id"),
            "slack": {
                "event": "app_mention",
                "team_id": team_id,
                "channel": channel,
                "channel_type": event.get("channel_type"),
                "thread_ts": thread_ts,
                "message_ts": event.get("ts"),
                "user_id": slack_user_id or None,
                "user_name": slack_user.get("user_name"),
                "user_display_name": slack_user.get("user_display_name"),
                "user_real_name": slack_user.get("user_real_name"),
            },
        }

        meta["slack_thread_context"] = inbox.thread_context(meta)
        meta["slack_participants"] = _slack_participant_map(
            client=client, user_cache=user_cache, meta=meta, text=text,
        )
        logger.info(
            "Slack app_mention channel=%s thread_ts=%s user=%s text=%r",
            channel,
            thread_ts,
            slack_user_id,
            message_for_ispec,
        )
        pending_reply_ts: str | None = None

        match = _CHOOSE_RE.match(text)
        if match:
            key = _pending_key(channel=channel, thread_ts=thread_ts)
            pending = pending_compare.get(key)
            if isinstance(pending, dict):
                try:
                    choice_index = int(match.group(1))
                except Exception:
                    choice_index = -1
                session_id_pending = str(pending.get("session_id") or "")
                user_message_id = int(pending.get("user_message_id") or 0)
                if session_id_pending and user_message_id > 0 and choice_index in {0, 1}:
                    pending_reply_ts = _safe_slack_post_message(
                        client=client,
                        channel=channel,
                        thread_ts=thread_ts,
                        text="Working on it...",
                    )
                    try:
                        payload = _post_ispec_choose(
                            server=cfg.ispec_server,
                            api_key=cfg.api_key,
                            session_id=session_id_pending,
                            user_message_id=user_message_id,
                            choice_index=choice_index,
                            timeout_seconds=cfg.timeout_seconds,
                        )
                    except Exception as exc:
                        logger.warning("iSPEC reply request failed error_type=%s", type(exc).__name__)
                        _send_or_update(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=thread_ts,
                            pending_ts=pending_reply_ts,
                            text=inbox.failed_reply(context, exc),
                        )
                        return
                    pending_compare.pop(key, None)
                    inbox.backend_result(context, payload)
                    message = chat_reply_text(payload)
                    if message:
                        _send_or_update(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=thread_ts,
                            pending_ts=pending_reply_ts,
                            text=message,
                        )
                    return
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                text="No pending compare choices in this thread.",
            )
            return

        pending_reply_ts = _safe_slack_post_message(
            client=client,
            channel=channel,
            thread_ts=thread_ts,
            text="Working on it...",
        )
        try:
            payload = _post_ispec_chat(
                server=cfg.ispec_server,
                api_key=cfg.api_key,
                session_id=session_id,
                message=message_for_ispec,
                meta=meta,
                timeout_seconds=cfg.timeout_seconds,
            )
        except Exception as exc:
            logger.warning("iSPEC reply request failed error_type=%s", type(exc).__name__)
            _send_or_update(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                pending_ts=pending_reply_ts,
                text=inbox.failed_reply(context, exc),
            )
            return

        inbox.backend_result(context, payload)
        compare = payload.get("compare") if isinstance(payload.get("compare"), dict) else None
        message = chat_reply_text(payload)
        if compare is not None:
            key = _pending_key(channel=channel, thread_ts=thread_ts)
            pending_compare[key] = {
                "session_id": session_id,
                "user_message_id": int(compare.get("userMessageId") or 0),
            }
            _send_or_update(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                pending_ts=pending_reply_ts,
                text=_format_compare_choices(compare),
            )
            return

        if not message:
            _send_or_update(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                pending_ts=pending_reply_ts,
                text="(no response)",
            )
            return
        _send_or_update(
            say=say,
            client=client,
            channel=channel,
            thread_ts=thread_ts,
            pending_ts=pending_reply_ts,
            text=message,
        )

    @app.event("message")
    @inbox.handler
    def _on_message(event, say, client, context, ack):  # type: ignore[no-untyped-def]
        """Handle DMs and thread followups (choose 0/1)."""

        try:
            ack()
        except Exception:
            pass
        if not _reply_user_allowed(event):
            return
        if event.get("bot_id") or event.get("subtype"):
            return

        channel = str(event.get("channel") or "")
        if not channel:
            return

        channel_type = event.get("channel_type")
        team_id = context.get("team_id")

        raw_text = _unescape_slack_text(str(event.get("text") or ""))
        if not raw_text:
            return

        # Channel mentions have their own handler; do not also route them as
        # artifact replies. Bot/subtype messages were already ignored above.
        bot_user_id = str(context.get("bot_user_id") or "")
        if channel_type not in {"im", "mpim"} and bot_user_id and f"<@{bot_user_id}>" in raw_text:
            return
        if not inbox.claim(context, fallback=event_claims, team_id=team_id, event=event, route="reply"):
            return

        thread_ts_raw = str(event.get("thread_ts") or "")
        reply_thread = _reply_thread_ts(channel_type=channel_type, thread_ts=thread_ts_raw or None)
        pending_key = _pending_key(channel=channel, thread_ts=thread_ts_raw or None)

        choose_match = _CHOOSE_RE.match(raw_text)
        if choose_match:
            pending = pending_compare.get(pending_key)
            if isinstance(pending, dict):
                try:
                    choice_index = int(choose_match.group(1))
                except Exception:
                    return

                session_id = str(pending.get("session_id") or "")
                user_message_id = int(pending.get("user_message_id") or 0)
                if session_id and user_message_id > 0:
                    pending_reply_ts = _safe_slack_post_message(
                        client=client,
                        channel=channel,
                        thread_ts=reply_thread,
                        text="Working on it...",
                    )
                    try:
                        payload = _post_ispec_choose(
                            server=cfg.ispec_server,
                            api_key=cfg.api_key,
                            session_id=session_id,
                            user_message_id=user_message_id,
                            choice_index=choice_index,
                            timeout_seconds=cfg.timeout_seconds,
                        )
                    except Exception as exc:
                        logger.warning("iSPEC reply request failed error_type=%s", type(exc).__name__)
                        _send_or_update(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=reply_thread,
                            pending_ts=pending_reply_ts,
                            text=inbox.failed_reply(context, exc),
                        )
                        return

                    pending_compare.pop(pending_key, None)
                    inbox.backend_result(context, payload)
                    message = chat_reply_text(payload)
                    if message:
                        _send_or_update(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=reply_thread,
                            pending_ts=pending_reply_ts,
                            text=message,
                        )
            return

        slack_user_id = str(event.get("user") or "").strip()
        cached: dict[str, str] = {}
        if slack_user_id:
            cached = _slack_user_summary(client=client, user_cache=user_cache, user_id=slack_user_id)

        if thread_ts_raw:
            try:
                route = _post_ispec_slack_artifact_reply(
                    server=cfg.ispec_server,
                    api_key=cfg.api_key,
                    payload={
                        "team_id": team_id,
                        "channel": channel,
                        "channel_type": channel_type,
                        "thread_ts": thread_ts_raw,
                        "message_ts": event.get("ts"),
                        "user_id": slack_user_id or None,
                        "user_name": cached.get("user_name"),
                        "user_display_name": cached.get("user_display_name"),
                        "user_real_name": cached.get("user_real_name"),
                        "text": raw_text,
                    },
                    timeout_seconds=cfg.timeout_seconds,
                )
            except Exception:
                route = {}
                logger.exception("Slack artifact reply route check failed channel=%s thread_ts=%s", channel, thread_ts_raw)
            if route.get("matched") is True:
                if str(os.getenv("ISPEC_SLACK_ARTIFACT_REPLY_ACK") or "1").strip().lower() not in {"0", "false", "no", "off"}:
                    _safe_slack_post_message(
                        client=client,
                        channel=channel,
                        thread_ts=reply_thread,
                        text="Recorded this review for the originating Codex/tmux pane.",
                    )
                return

        # Only respond to free-form messages in DMs; keep channels quiet unless mentioned.
        if channel_type not in {"im", "mpim"}:
            return

        session_id = (
            _session_id(team_id=team_id, channel=channel, thread_ts=thread_ts_raw)
            if thread_ts_raw else _session_id_for_dm(team_id=team_id, channel=channel)
        )
        logger.info(
            "Slack DM message channel=%s user=%s text=%r",
            channel,
            event.get("user"),
            raw_text,
        )

        message_for_ispec = _format_message_for_ispec(text=raw_text, slack_user=cached)
        meta: dict[str, Any] = {
            "source": "slack",
            "slack_inbox_id": context.get("slack_inbox_id"),
            "slack": {
                "event": "dm",
                "team_id": team_id,
                "channel": channel,
                "channel_type": channel_type,
                "thread_ts": thread_ts_raw or None,
                "message_ts": event.get("ts"),
                "user_id": slack_user_id or None,
                "user_name": cached.get("user_name"),
                "user_display_name": cached.get("user_display_name"),
                "user_real_name": cached.get("user_real_name"),
            },
        }
        meta["slack_thread_context"] = inbox.thread_context(meta)
        meta["slack_participants"] = _slack_participant_map(
            client=client, user_cache=user_cache, meta=meta, text=raw_text,
        )
        pending_reply_ts = _safe_slack_post_message(
            client=client,
            channel=channel,
            thread_ts=reply_thread,
            text="Working on it...",
        )

        try:
            payload = _post_ispec_chat(
                server=cfg.ispec_server,
                api_key=cfg.api_key,
                session_id=session_id,
                message=message_for_ispec,
                meta=meta,
                timeout_seconds=cfg.timeout_seconds,
            )
        except Exception as exc:
            logger.warning("iSPEC reply request failed error_type=%s", type(exc).__name__)
            _send_or_update(
                say=say,
                client=client,
                channel=channel,
                thread_ts=reply_thread,
                pending_ts=pending_reply_ts,
                text=inbox.failed_reply(context, exc),
            )
            return

        inbox.backend_result(context, payload)
        compare = payload.get("compare") if isinstance(payload.get("compare"), dict) else None
        message = chat_reply_text(payload)
        if compare is not None:
            pending_compare[pending_key] = {
                "session_id": session_id,
                "user_message_id": int(compare.get("userMessageId") or 0),
            }
            _send_or_update(
                say=say,
                client=client,
                channel=channel,
                thread_ts=reply_thread,
                pending_ts=pending_reply_ts,
                text=_format_compare_choices(compare),
            )
            return

        if not message:
            _send_or_update(
                say=say,
                client=client,
                channel=channel,
                thread_ts=reply_thread,
                pending_ts=pending_reply_ts,
                text="(no response)",
            )
            return
        _send_or_update(
            say=say,
            client=client,
            channel=channel,
            thread_ts=reply_thread,
            pending_ts=pending_reply_ts,
            text=message,
        )

    logger.info("Starting Slack bot (Socket Mode) -> %s", cfg.ispec_server)
    SocketModeHandler(app, cfg.app_token).start()
