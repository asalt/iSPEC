"""Local relay inbox helpers.

The relay is a small awaiting dispatcher: trusted local callers submit a
structured request, iSPEC validates it with canonical config/allowlists, and
the supervisor records a receipt. Live external writes are intentionally
opt-in; the default outcome is a staged request with precise provenance.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import requests
from sqlalchemy.orm import Session

from ispec.agent.commands import COMMAND_LOCAL_RELAY_REQUEST
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentEvent
from ispec.assistant.slack_tmux_bridge import stable_json
from ispec.cli.env import parse_env_file_text

try:  # Python 3.11+
    import tomllib
except Exception:  # pragma: no cover - Python <3.11 fallback
    tomllib = None  # type: ignore[assignment]


RELAY_AGENT_ID = "local-relay"
RELAY_SCHEMA_VERSION = 1

EVENT_RELAY_REQUEST_ENQUEUED = "local_relay_request_enqueued_v1"
EVENT_RELAY_RECEIPT = "local_relay_receipt_v1"

FAILURE_INVALID_REQUEST = "invalid_request"
FAILURE_UNSUPPORTED_KIND = "unsupported_kind"
FAILURE_MISSING_BODY = "missing_body"
FAILURE_MISSING_TARGET = "missing_target"
FAILURE_TARGET_NOT_ALLOWED = "target_not_allowed"
FAILURE_TARGET_BLOCKED = "target_blocked"
FAILURE_CONFIRMATION_REQUIRED = "confirmation_required"
FAILURE_LIVE_SEND_DISABLED = "live_send_disabled"
FAILURE_MISSING_TOKEN = "missing_token"
FAILURE_PROVIDER_ERROR = "provider_error"
FAILURE_TMUX_SEND_FAILED = "tmux_send_failed"
FAILURE_SOURCE_NOT_ALLOWED = "source_not_allowed"
FAILURE_ATTACHMENT_UNSUPPORTED = "attachment_upload_unsupported"
FAILURE_ATTACHMENT_MISSING = "attachment_missing"
FAILURE_ATTACHMENT_TOO_LARGE = "attachment_too_large"
FAILURE_ATTACHMENT_UPLOAD_FAILED = "attachment_upload_failed"

KIND_SLACK_MESSAGE = "slack_message"
KIND_TMUX_SEND = "tmux_send"
KIND_STATUS_RECORD = "status_record"
SUPPORTED_KINDS = {KIND_SLACK_MESSAGE, KIND_TMUX_SEND, KIND_STATUS_RECORD}


def utcnow() -> datetime:
    return datetime.now(UTC)


def _slug(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def _truncate(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _now_plus(delay_seconds: int) -> datetime:
    return utcnow() + timedelta(seconds=max(0, int(delay_seconds)))


def _find_make_root_from_here() -> Path | None:
    override = str(os.getenv("ISPEC_RELAY_CONFIG_ROOT") or os.getenv("ISPEC_FULL_ROOT") or "").strip()
    if override:
        path = Path(override).expanduser()
        try:
            return path.resolve()
        except Exception:
            return path

    try:
        from ispec.cli import dev as dev_cli

        found = dev_cli._find_make_root(start=Path(__file__).resolve().parent)  # type: ignore[attr-defined]
        if found is not None:
            return Path(found).resolve()
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "Makefile").is_file():
            return parent
    return None


def _candidate_env_files(root: Path | None) -> list[Path]:
    if root is None:
        return []
    names = [
        ".env",
        ".env.local",
        ".env.slack",
        ".env.slack.local",
        "iSPEC/.env.local",
        "iSPEC/.env.slack",
        "iSPEC/.env.slack.local",
    ]
    return [root / name for name in names]


@dataclass(frozen=True)
class CanonicalEnv:
    root: Path | None
    values: dict[str, str]
    sources: dict[str, str]
    loaded_files: list[str]
    errors: list[str]


def _read_env_file(path: Path, values: dict[str, str], sources: dict[str, str], errors: list[str]) -> None:
    try:
        parsed = parse_env_file_text(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
        return
    for key, value in parsed.items():
        values[str(key)] = str(value)
        sources[str(key)] = f"file:{path}:{key}"


def _load_toml_slack_tokens(root: Path | None, values: dict[str, str], sources: dict[str, str], errors: list[str]) -> None:
    if tomllib is None or root is None:
        return
    configured = str(values.get("ISPEC_SLACK_CONFIG_TOML") or "").strip()
    candidates: list[Path] = []
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = root / path
        candidates.append(path)
    candidates.append(root / ".env.slack.toml")

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        slack = data.get("slack") if isinstance(data, dict) else None
        if not isinstance(slack, dict):
            continue
        for toml_key, env_key in (
            ("bot_token", "ISPEC_SLACK_BOT_TOKEN"),
            ("slack_bot_token", "ISPEC_SLACK_BOT_TOKEN"),
            ("app_token", "ISPEC_SLACK_APP_TOKEN"),
            ("slack_app_token", "ISPEC_SLACK_APP_TOKEN"),
        ):
            value = str(slack.get(toml_key) or "").strip()
            if value and not values.get(env_key):
                values[env_key] = value
                sources[env_key] = f"toml:{path}:slack.{toml_key}"


def load_canonical_env() -> CanonicalEnv:
    root = _find_make_root_from_here()
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    loaded_files: list[str] = []
    errors: list[str] = []

    for path in _candidate_env_files(root):
        if path.is_file():
            loaded_files.append(str(path))
            _read_env_file(path, values, sources, errors)

    _load_toml_slack_tokens(root, values, sources, errors)

    # The already-running iSPEC process remains the highest-precedence source,
    # but callers from arbitrary cwd can still rely on canonical repo env files.
    for key, value in os.environ.items():
        if key.startswith("ISPEC_") or key.startswith("SLACK_") or key in {"DEV_TMUX_SESSION"}:
            values[key] = str(value)
            sources[key] = f"process_env:{key}"

    return CanonicalEnv(root=root, values=values, sources=sources, loaded_files=loaded_files, errors=errors)


def _token_info(env: CanonicalEnv) -> tuple[str, dict[str, Any]]:
    for key in ("ISPEC_SLACK_BOT_TOKEN", "SLACK_BOT_TOKEN"):
        token = str(env.values.get(key) or "").strip()
        if token:
            return token, {"present": True, "env_var": key, "source": env.sources.get(key)}
    return "", {"present": False, "env_var": None, "source": None}


def _json_object_from_env(env: CanonicalEnv, key: str) -> Any:
    raw = str(env.values.get(key) or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _merge_destinations(destinations: dict[str, dict[str, Any]], value: Any, *, source: str) -> None:
    if isinstance(value, dict) and isinstance(value.get("destinations"), dict):
        value = value["destinations"]
    if not isinstance(value, dict):
        return
    for raw_key, raw_entry in value.items():
        alias = _slug(str(raw_key))
        if not alias:
            continue
        if isinstance(raw_entry, str):
            entry: dict[str, Any] = {"channel": raw_entry} if raw_entry.startswith(("C", "G", "D")) else {"user_id": raw_entry}
        elif isinstance(raw_entry, dict):
            entry = dict(raw_entry)
        else:
            continue
        entry["alias"] = alias
        entry["source"] = source
        destinations[alias] = entry


def _assistant_destination_file(env: CanonicalEnv) -> Path | None:
    raw = str(env.values.get("ISPEC_ASSISTANT_SLACK_DESTINATIONS_PATH") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute() and env.root is not None:
            path = env.root / path
        return path
    if env.root is None:
        return None
    path = env.root / "configs" / "assistant-slack-destinations.local.json"
    return path if path.is_file() else None


def _load_destinations(env: CanonicalEnv) -> dict[str, dict[str, Any]]:
    destinations: dict[str, dict[str, Any]] = {}
    staff_channel = str(env.values.get("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL") or "").strip()
    if staff_channel:
        destinations["staff"] = {
            "alias": "staff",
            "kind": "channel",
            "audience": "staff",
            "channel": staff_channel,
            "source": env.sources.get("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL") or "ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL",
        }
    for key in ("ISPEC_ASSISTANT_SLACK_ALLOWED_DESTINATIONS_JSON", "ISPEC_ASSISTANT_SLACK_DESTINATIONS_JSON"):
        parsed = _json_object_from_env(env, key)
        _merge_destinations(destinations, parsed, source=env.sources.get(key) or key)

    path = _assistant_destination_file(env)
    if path is not None and path.is_file():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            parsed = None
        _merge_destinations(destinations, parsed, source=str(path))
    return destinations


def resolve_slack_destination(env: CanonicalEnv, alias: str | None, *, message_type: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    requested = _slug(alias or "staff")
    if not requested:
        return None, FAILURE_MISSING_TARGET
    destinations = _load_destinations(env)
    entry = destinations.get(requested)
    if not isinstance(entry, dict):
        return None, FAILURE_TARGET_NOT_ALLOWED

    requested_type = _slug(message_type)
    allowed_raw = entry.get("allowed_message_types")
    if requested_type and isinstance(allowed_raw, list) and allowed_raw:
        allowed = {_slug(str(item)) for item in allowed_raw if _slug(str(item))}
        if requested_type not in allowed:
            return None, FAILURE_TARGET_NOT_ALLOWED

    channel = str(entry.get("channel") or "").strip()
    user_id = str(entry.get("user_id") or entry.get("user") or "").strip()
    email = str(entry.get("email") or "").strip()
    if not channel and not user_id and not email:
        return None, FAILURE_TARGET_NOT_ALLOWED

    kind = str(entry.get("kind") or "").strip().lower()
    if not kind:
        kind = "dm" if user_id or email or channel.startswith("D") else "channel"
    resolved: dict[str, Any] = {
        "alias": requested,
        "kind": kind,
        "audience": str(entry.get("audience") or "").strip() or None,
        "source": str(entry.get("source") or "").strip() or None,
    }
    if channel:
        resolved["channel"] = channel
    if user_id:
        resolved["user_id"] = user_id
    if email:
        resolved["email"] = email
    if requested_type:
        resolved["message_type"] = requested_type
    return resolved, None


def relay_config_probe(*, target_alias: str | None = None, message_type: str | None = None) -> dict[str, Any]:
    env = load_canonical_env()
    _token, token_info = _token_info(env)
    destinations = _load_destinations(env)
    allowed_sources = _allowed_source_entries(env)
    destination, destination_error = (None, None)
    if target_alias:
        destination, destination_error = resolve_slack_destination(env, target_alias, message_type=message_type)
    return {
        "schema_version": RELAY_SCHEMA_VERSION,
        "root": str(env.root) if env.root is not None else None,
        "env_files_loaded": list(env.loaded_files),
        "env_errors": list(env.errors),
        "relay": {
            "live_send_enabled": _relay_live_enabled(env),
            "allowed_sources_configured": bool(allowed_sources),
            "allowed_sources_count": len(allowed_sources),
            "pdf_attachment_upload": {
                "supported": True,
                "max_bytes": _slack_upload_max_bytes(env),
                "requires_mode_send": True,
                "requires_confirm": True,
            },
        },
        "slack": {
            "bot_token": token_info,
            "destinations_count": len(destinations),
            "destination_sources": sorted(
                {
                    str(item.get("source") or "")
                    for item in destinations.values()
                    if isinstance(item, dict) and str(item.get("source") or "").strip()
                }
            ),
            "target": destination,
            "target_error": destination_error,
        },
    }


def _normalize_source(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "kind": _truncate(value.get("kind") or value.get("type") or "local", limit=64),
            "id": _truncate(value.get("id") or value.get("agent_id") or value.get("name") or "", limit=256) or None,
            "cwd": _truncate(value.get("cwd") or "", limit=1000) or None,
        }
    text = _truncate(value, limit=256)
    return {"kind": "local", "id": text or None, "cwd": None}


def _normalize_target(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        alias = value.get("alias") or value.get("to") or value.get("name")
        tmux_target = value.get("target") or value.get("tmux_target")
        normalized = {
            "alias": _truncate(alias, limit=120) or None,
            "target": _truncate(tmux_target, limit=240) or None,
        }
        if value.get("channel"):
            normalized["channel"] = _truncate(value.get("channel"), limit=120)
        if value.get("user_id"):
            normalized["user_id"] = _truncate(value.get("user_id"), limit=120)
        return normalized
    text = _truncate(value, limit=240)
    return {"alias": text or None, "target": text or None}


def normalize_relay_request(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, FAILURE_INVALID_REQUEST

    nested = raw.get("relay_request") if isinstance(raw.get("relay_request"), dict) else raw
    kind = _slug(str(nested.get("kind") or nested.get("type") or ""))
    if kind not in SUPPORTED_KINDS:
        return None, FAILURE_UNSUPPORTED_KIND

    body = nested.get("body")
    if body is None:
        body = nested.get("text") or nested.get("message")
    body_text = _truncate(body, limit=20_000)

    target = _normalize_target(nested.get("target") if "target" in nested else nested.get("to"))
    mode = _slug(str(nested.get("mode") or "stage"))
    if mode not in {"stage", "send"}:
        mode = "stage"

    attachments_raw = nested.get("attachments")
    if attachments_raw is None and nested.get("attachment"):
        attachments_raw = [nested.get("attachment")]
    attachments: list[dict[str, Any]] = []
    if isinstance(attachments_raw, list):
        for item in attachments_raw[:20]:
            if isinstance(item, dict):
                path = _truncate(item.get("path") or item.get("file"), limit=2000)
                title = _truncate(item.get("title"), limit=240) or None
            else:
                path = _truncate(item, limit=2000)
                title = None
            if path:
                attachments.append({"path": path, "title": title})

    request_id = _truncate(nested.get("request_id") or nested.get("id"), limit=128)
    if not request_id:
        import uuid

        request_id = f"relay-{uuid.uuid4().hex[:16]}"

    normalized = {
        "schema_version": RELAY_SCHEMA_VERSION,
        "request_id": request_id,
        "kind": kind,
        "mode": mode,
        "confirm": bool(nested.get("confirm") is True),
        "source": _normalize_source(nested.get("source") or nested.get("origin") or "local"),
        "target": target,
        "body": body_text,
        "attachments": attachments,
        "message_type": _slug(nested.get("message_type")) or None,
        "thread_ts": _truncate(nested.get("thread_ts"), limit=80) or None,
        "press_enter": bool(nested.get("press_enter") is True),
        "metadata": nested.get("metadata") if isinstance(nested.get("metadata"), dict) else {},
        "provenance": nested.get("provenance") if isinstance(nested.get("provenance"), dict) else {},
    }
    return normalized, None


def enqueue_relay_request(
    db: Session,
    *,
    request: dict[str, Any],
    priority: int = 0,
    delay_seconds: int = 0,
    max_attempts: int = 1,
) -> tuple[AgentCommand, dict[str, Any]]:
    normalized, error = normalize_relay_request(request)
    if normalized is None:
        raise ValueError(error or FAILURE_INVALID_REQUEST)

    now = utcnow()
    row = AgentCommand(
        command_type=COMMAND_LOCAL_RELAY_REQUEST,
        status="queued",
        priority=max(-50, min(1000, int(priority or 0))),
        created_at=now,
        updated_at=now,
        available_at=_now_plus(int(delay_seconds or 0)),
        attempts=0,
        max_attempts=max(1, min(10, int(max_attempts or 1))),
        payload_json={"relay_request": normalized},
        result_json={},
    )
    db.add(row)
    db.flush()

    event_payload = {
        "schema_version": RELAY_SCHEMA_VERSION,
        "request": normalized,
        "command_id": int(row.id),
        "enqueued_at": now.isoformat(),
    }
    db.add(
        AgentEvent(
            agent_id=RELAY_AGENT_ID,
            event_type=EVENT_RELAY_REQUEST_ENQUEUED,
            ts=now,
            received_at=now,
            name="relay_request_enqueued",
            severity="info",
            correlation_id=str(normalized.get("request_id") or ""),
            payload_json=stable_json(event_payload),
        )
    )
    db.commit()
    db.refresh(row)
    return row, event_payload


def _tmux_entries(kind: str) -> list[str]:
    try:
        from ispec.assistant import tools as assistant_tools

        if kind == "allow":
            return list(assistant_tools._tmux_allowlist_entries())  # type: ignore[attr-defined]
        return list(assistant_tools._tmux_blacklist_entries())  # type: ignore[attr-defined]
    except Exception:
        return []


def _target_matches_entry(target: str, entry: str) -> bool:
    target_text = str(target or "").strip()
    entry_text = str(entry or "").strip()
    if not target_text or not entry_text:
        return False
    if entry_text.endswith("*"):
        return bool(entry_text[:-1]) and target_text.startswith(entry_text[:-1])
    return target_text == entry_text


def _validate_tmux_target(target: str) -> tuple[bool, str | None, dict[str, Any]]:
    allowlist = _tmux_entries("allow")
    blacklist = _tmux_entries("black")
    blacklist_match = next((entry for entry in blacklist if _target_matches_entry(target, entry)), None)
    if blacklist_match:
        return False, FAILURE_TARGET_BLOCKED, {
            "target": target,
            "allowlist_count": len(allowlist),
            "blacklist_match": blacklist_match,
        }
    allowlist_match = next((entry for entry in allowlist if _target_matches_entry(target, entry)), None)
    if not allowlist or not allowlist_match:
        return False, FAILURE_TARGET_NOT_ALLOWED, {
            "target": target,
            "allowlist_count": len(allowlist),
            "blacklist_count": len(blacklist),
        }
    return True, None, {
        "target": target,
        "allowlist_match": allowlist_match,
        "blacklist_count": len(blacklist),
    }


def _relay_live_enabled(env: CanonicalEnv | None = None) -> bool:
    if env is not None and "ISPEC_RELAY_LIVE_SEND_ENABLED" in env.values:
        return _is_truthy(env.values.get("ISPEC_RELAY_LIVE_SEND_ENABLED"))
    return _is_truthy(os.getenv("ISPEC_RELAY_LIVE_SEND_ENABLED"))


def _allowed_source_entries(env: CanonicalEnv | None = None) -> list[str]:
    if env is not None and "ISPEC_RELAY_ALLOWED_SOURCES" in env.values:
        raw = str(env.values.get("ISPEC_RELAY_ALLOWED_SOURCES") or "").strip()
    else:
        raw = str(os.getenv("ISPEC_RELAY_ALLOWED_SOURCES") or "").strip()
    if not raw:
        return []
    return [item.strip() for item in re.split(r"[\s,]+", raw) if item.strip()]


def _source_policy(request: dict[str, Any], *, env: CanonicalEnv | None = None) -> tuple[bool, dict[str, Any]]:
    source = request.get("source") if isinstance(request.get("source"), dict) else {}
    source_id = str(source.get("id") or "").strip()
    source_kind = str(source.get("kind") or "").strip()
    candidates = {value for value in (source_id, source_kind, f"{source_kind}:{source_id}" if source_kind and source_id else "") if value}
    allowed_entries = _allowed_source_entries(env)
    if not allowed_entries:
        return True, {
            "source": source,
            "allowed": True,
            "allowed_sources_configured": False,
        }
    matched = next((entry for entry in allowed_entries if entry in candidates), None)
    return bool(matched), {
        "source": source,
        "allowed": bool(matched),
        "allowed_sources_configured": True,
        "matched": matched,
    }


def _slack_timeout_seconds(env: CanonicalEnv) -> float:
    raw = str(env.values.get("ISPEC_SLACK_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 10.0
    try:
        return max(1.0, min(60.0, float(raw)))
    except ValueError:
        return 10.0


def _slack_api_call(
    *,
    token: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    post: Callable[..., Any] | None = None,
    as_form: bool = False,
) -> dict[str, Any]:
    post_fn = post or requests.post
    request_kwargs: dict[str, Any]
    if as_form:
        request_kwargs = {"data": payload}
        content_type = "application/x-www-form-urlencoded"
    else:
        request_kwargs = {"json": payload}
        content_type = "application/json; charset=utf-8"
    resp = post_fn(
        f"https://slack.com/api/{endpoint.lstrip('/')}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        },
        **request_kwargs,
        timeout=max(1.0, float(timeout_seconds)),
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _execute_slack_send(
    *,
    request: dict[str, Any],
    env: CanonicalEnv,
    destination: dict[str, Any],
    post: Callable[..., Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    token, token_meta = _token_info(env)
    if not token:
        return False, {"ok": False, "error_type": FAILURE_MISSING_TOKEN, "token": token_meta}, FAILURE_MISSING_TOKEN
    timeout_seconds = _slack_timeout_seconds(env)

    channel_result, channel_error = _resolve_slack_channel(
        token=token,
        destination=destination,
        timeout_seconds=timeout_seconds,
        post=post,
    )
    if channel_error:
        return False, channel_result, channel_error

    channel = str(channel_result.get("channel") or "").strip()
    user_id = str(channel_result.get("user_id") or "").strip()
    email = str(channel_result.get("email") or "").strip()
    if not channel:
        return False, {"ok": False, "error": "Missing channel after destination resolution."}, FAILURE_MISSING_TARGET

    message_payload: dict[str, Any] = {"channel": channel, "text": str(request.get("body") or "")}
    if request.get("thread_ts"):
        message_payload["thread_ts"] = str(request["thread_ts"])
    posted = _slack_api_call(
        token=token,
        endpoint="chat.postMessage",
        payload=message_payload,
        timeout_seconds=timeout_seconds,
        post=post,
    )
    if posted.get("ok") is not True:
        error = str(posted.get("error") or "unknown_error")
        return False, {"ok": False, "error": error, "slack": posted}, FAILURE_PROVIDER_ERROR
    return True, {
        "ok": True,
        "sent": True,
        "channel": channel,
        "user_id": user_id or None,
        "email": email or None,
        "thread_ts": request.get("thread_ts"),
        "slack": posted,
    }, None


def _resolve_slack_channel(
    *,
    token: str,
    destination: dict[str, Any],
    timeout_seconds: float,
    post: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    channel = str(destination.get("channel") or "").strip()
    user_id = str(destination.get("user_id") or "").strip()
    email = str(destination.get("email") or "").strip()
    if not channel and email:
        lookup = _slack_api_call(
            token=token,
            endpoint="users.lookupByEmail",
            payload={"email": email},
            timeout_seconds=timeout_seconds,
            post=post,
        )
        if lookup.get("ok") is not True:
            error = str(lookup.get("error") or "unknown_error")
            return {"ok": False, "error": error, "slack": lookup}, FAILURE_PROVIDER_ERROR
        user = lookup.get("user")
        if isinstance(user, dict):
            user_id = str(user.get("id") or "").strip()

    if not channel and user_id:
        opened = _slack_api_call(
            token=token,
            endpoint="conversations.open",
            payload={"users": user_id},
            timeout_seconds=timeout_seconds,
            post=post,
        )
        if opened.get("ok") is not True:
            error = str(opened.get("error") or "unknown_error")
            return {"ok": False, "error": error, "slack": opened}, FAILURE_PROVIDER_ERROR
        channel_obj = opened.get("channel")
        if isinstance(channel_obj, dict):
            channel = str(channel_obj.get("id") or "").strip()

    if not channel:
        return {"ok": False, "error": "Missing channel after destination resolution."}, FAILURE_MISSING_TARGET

    return {"ok": True, "channel": channel, "user_id": user_id or None, "email": email or None}, None


def _slack_upload_max_bytes(env: CanonicalEnv) -> int:
    raw = str(env.values.get("ISPEC_SLACK_UPLOAD_MAX_BYTES") or "").strip()
    if not raw:
        return 50 * 1024 * 1024
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 50 * 1024 * 1024


def _resolve_pdf_attachment(item: dict[str, Any], *, env: CanonicalEnv) -> tuple[dict[str, Any] | None, str | None]:
    raw_path = str(item.get("path") or "").strip()
    if not raw_path:
        return None, FAILURE_ATTACHMENT_MISSING
    try:
        file_path = Path(raw_path).expanduser().resolve()
    except Exception:
        file_path = Path(raw_path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return None, FAILURE_ATTACHMENT_MISSING
    if file_path.name.startswith(".env"):
        return None, FAILURE_ATTACHMENT_UNSUPPORTED
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    if file_path.suffix.lower() != ".pdf" and mime_type != "application/pdf":
        return None, FAILURE_ATTACHMENT_UNSUPPORTED
    size_bytes = int(file_path.stat().st_size)
    max_bytes = _slack_upload_max_bytes(env)
    if size_bytes > max_bytes:
        return None, FAILURE_ATTACHMENT_TOO_LARGE
    title = _truncate(item.get("title"), limit=240) or file_path.name
    return {
        "path": str(file_path),
        "filename": file_path.name,
        "title": title,
        "size_bytes": size_bytes,
        "mime_type": mime_type,
    }, None


def _upload_slack_file_external(
    *,
    token: str,
    channel: str,
    attachment: dict[str, Any],
    timeout_seconds: float,
    text: str | None,
    thread_ts: str | None,
    post: Callable[..., Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    get_url_response = _slack_api_call(
        token=token,
        endpoint="files.getUploadURLExternal",
        payload={
            "filename": attachment["filename"],
            "length": attachment["size_bytes"],
        },
        timeout_seconds=timeout_seconds,
        post=post,
        as_form=True,
    )
    if get_url_response.get("ok") is not True:
        return False, {
            "ok": False,
            "stage": "get_upload_url",
            "error": str(get_url_response.get("error") or "unknown_error"),
            "file": {key: attachment[key] for key in ("path", "filename", "size_bytes", "mime_type")},
            "slack": get_url_response,
        }, FAILURE_ATTACHMENT_UPLOAD_FAILED

    upload_url = str(get_url_response.get("upload_url") or "").strip()
    file_id = str(get_url_response.get("file_id") or "").strip()
    if not upload_url or not file_id:
        return False, {
            "ok": False,
            "stage": "get_upload_url",
            "error": "missing_upload_url_or_file_id",
            "file": {key: attachment[key] for key in ("path", "filename", "size_bytes", "mime_type")},
        }, FAILURE_ATTACHMENT_UPLOAD_FAILED

    post_fn = post or requests.post
    with Path(attachment["path"]).open("rb") as handle:
        upload_response = post_fn(
            upload_url,
            files={"file": (attachment["filename"], handle, attachment["mime_type"])},
            timeout=max(1.0, float(timeout_seconds)),
        )
    status_code = int(getattr(upload_response, "status_code", 200) or 200)
    if status_code < 200 or status_code >= 300:
        return False, {
            "ok": False,
            "stage": "upload_bytes",
            "error": f"upload_http_{status_code}",
            "file": {key: attachment[key] for key in ("path", "filename", "size_bytes", "mime_type")},
            "body": str(getattr(upload_response, "text", ""))[:500],
        }, FAILURE_ATTACHMENT_UPLOAD_FAILED

    complete_payload: dict[str, Any] = {
        "files": [{"id": file_id, "title": attachment["title"]}],
        "channel_id": channel,
    }
    if thread_ts:
        complete_payload["thread_ts"] = str(thread_ts).strip()
    if text:
        complete_payload["initial_comment"] = str(text).strip()
    complete_response = _slack_api_call(
        token=token,
        endpoint="files.completeUploadExternal",
        payload=complete_payload,
        timeout_seconds=timeout_seconds,
        post=post,
    )
    if complete_response.get("ok") is not True:
        return False, {
            "ok": False,
            "stage": "complete_upload",
            "error": str(complete_response.get("error") or "unknown_error"),
            "file": {key: attachment[key] for key in ("path", "filename", "size_bytes", "mime_type")},
            "slack": complete_response,
        }, FAILURE_ATTACHMENT_UPLOAD_FAILED

    return True, {
        "ok": True,
        "file_id": file_id,
        "file": {key: attachment[key] for key in ("path", "filename", "title", "size_bytes", "mime_type")},
        "slack": complete_response,
    }, None


def _execute_slack_uploads(
    *,
    request: dict[str, Any],
    env: CanonicalEnv,
    destination: dict[str, Any],
    post: Callable[..., Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    token, token_meta = _token_info(env)
    if not token:
        return False, {"ok": False, "error_type": FAILURE_MISSING_TOKEN, "token": token_meta}, FAILURE_MISSING_TOKEN
    timeout_seconds = _slack_timeout_seconds(env)
    channel_result, channel_error = _resolve_slack_channel(
        token=token,
        destination=destination,
        timeout_seconds=timeout_seconds,
        post=post,
    )
    if channel_error:
        return False, channel_result, channel_error

    channel = str(channel_result.get("channel") or "").strip()
    attachments: list[dict[str, Any]] = []
    for item in request.get("attachments") or []:
        if not isinstance(item, dict):
            return False, {"ok": False, "error": "Invalid attachment entry."}, FAILURE_ATTACHMENT_UNSUPPORTED
        attachment, attachment_error = _resolve_pdf_attachment(item, env=env)
        if attachment_error or attachment is None:
            return False, {"ok": False, "error_type": attachment_error, "attachment": item}, attachment_error
        attachments.append(attachment)

    uploaded: list[dict[str, Any]] = []
    for idx, attachment in enumerate(attachments):
        ok, result, error = _upload_slack_file_external(
            token=token,
            channel=channel,
            attachment=attachment,
            timeout_seconds=timeout_seconds,
            text=str(request.get("body") or "").strip() if idx == 0 else None,
            thread_ts=request.get("thread_ts"),
            post=post,
        )
        if not ok:
            return False, {
                "ok": False,
                "sent": bool(uploaded),
                "channel": channel,
                "uploaded": uploaded,
                "failed_upload": result,
            }, error or FAILURE_ATTACHMENT_UPLOAD_FAILED
        uploaded.append(result)

    return True, {
        "ok": True,
        "sent": True,
        "channel": channel,
        "user_id": channel_result.get("user_id"),
        "email": channel_result.get("email"),
        "thread_ts": request.get("thread_ts"),
        "attachments_uploaded": uploaded,
    }, None


def _execute_tmux_send(request: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    target = str((request.get("target") or {}).get("target") or "").strip()
    try:
        from ispec.assistant import tools as assistant_tools

        result = assistant_tools._tmux_send_text(  # type: ignore[attr-defined]
            target=target,
            text=str(request.get("body") or ""),
            press_enter=bool(request.get("press_enter")),
        )
        return True, {"ok": True, "sent": True, "tmux": result}, None
    except Exception as exc:
        return False, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, FAILURE_TMUX_SEND_FAILED


def _receipt_event(
    *,
    command_id: int | None,
    request: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    now = utcnow()
    with get_agent_session() as db:
        db.add(
            AgentEvent(
                agent_id=RELAY_AGENT_ID,
                event_type=EVENT_RELAY_RECEIPT,
                ts=now,
                received_at=now,
                name="relay_receipt",
                severity="info" if bool(receipt.get("ok")) else "warning",
                correlation_id=str(request.get("request_id") or ""),
                payload_json=stable_json(
                    {
                        "schema_version": RELAY_SCHEMA_VERSION,
                        "command_id": int(command_id) if command_id is not None else None,
                        "request": request,
                        "receipt": receipt,
                    }
                ),
            )
        )
        db.commit()


def dispatch_relay_request(
    payload: dict[str, Any],
    *,
    command_id: int | None = None,
    slack_post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    request, normalize_error = normalize_relay_request(payload)
    if request is None:
        receipt = {
            "ok": False,
            "schema_version": RELAY_SCHEMA_VERSION,
            "delivery_outcome": "failed",
            "sent": False,
            "error_type": normalize_error or FAILURE_INVALID_REQUEST,
        }
        _receipt_event(command_id=command_id, request={"request_id": None}, receipt=receipt)
        return receipt

    env = load_canonical_env()
    probe = relay_config_probe(
        target_alias=(request.get("target") or {}).get("alias"),
        message_type=request.get("message_type"),
    )
    kind = str(request.get("kind") or "")
    mode = str(request.get("mode") or "stage")
    body = str(request.get("body") or "").strip()

    receipt: dict[str, Any] = {
        "ok": True,
        "schema_version": RELAY_SCHEMA_VERSION,
        "request_id": request.get("request_id"),
        "kind": kind,
        "mode": mode,
        "delivery_outcome": "staged",
        "sent": False,
        "config_probe": probe,
        "policy": {
            "live_send_enabled": _relay_live_enabled(env),
            "confirm": bool(request.get("confirm") is True),
        },
        "target": request.get("target"),
        "provenance": request.get("provenance") if isinstance(request.get("provenance"), dict) else {},
        "metadata": request.get("metadata") if isinstance(request.get("metadata"), dict) else {},
    }
    source_allowed, source_policy = _source_policy(request, env=env)
    receipt["policy"]["source"] = source_policy
    if not source_allowed:
        receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_SOURCE_NOT_ALLOWED)
        _receipt_event(command_id=command_id, request=request, receipt=receipt)
        return receipt

    if kind in {KIND_SLACK_MESSAGE, KIND_TMUX_SEND} and not body:
        receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_MISSING_BODY)
        _receipt_event(command_id=command_id, request=request, receipt=receipt)
        return receipt

    if kind == KIND_STATUS_RECORD:
        receipt["delivery_outcome"] = "recorded"
        _receipt_event(command_id=command_id, request=request, receipt=receipt)
        return receipt

    if kind == KIND_SLACK_MESSAGE:
        target_alias = (request.get("target") or {}).get("alias")
        destination, destination_error = resolve_slack_destination(
            env,
            str(target_alias or ""),
            message_type=request.get("message_type"),
        )
        if destination_error or destination is None:
            receipt.update(ok=False, delivery_outcome="failed", error_type=destination_error or FAILURE_TARGET_NOT_ALLOWED)
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        receipt["resolved_target"] = destination
        if mode != "send":
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        if request.get("confirm") is not True:
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_CONFIRMATION_REQUIRED)
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        if not _relay_live_enabled(env):
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_LIVE_SEND_DISABLED)
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        if request.get("attachments"):
            ok, result, error = _execute_slack_uploads(
                request=request,
                env=env,
                destination=destination,
                post=slack_post,
            )
        else:
            ok, result, error = _execute_slack_send(request=request, env=env, destination=destination, post=slack_post)
        receipt.update(result)
        receipt["ok"] = bool(ok)
        receipt["delivery_outcome"] = "sent" if ok else "failed"
        if error:
            receipt["error_type"] = error
        _receipt_event(command_id=command_id, request=request, receipt=receipt)
        return receipt

    if kind == KIND_TMUX_SEND:
        target = str((request.get("target") or {}).get("target") or "").strip()
        if not target:
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_MISSING_TARGET)
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        allowed, target_error, policy_detail = _validate_tmux_target(target)
        receipt["policy"]["tmux"] = policy_detail
        if not allowed:
            receipt.update(ok=False, delivery_outcome="failed", error_type=target_error or FAILURE_TARGET_NOT_ALLOWED)
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        if mode != "send":
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        if request.get("confirm") is not True:
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_CONFIRMATION_REQUIRED)
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        if not _relay_live_enabled(env):
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_LIVE_SEND_DISABLED)
            _receipt_event(command_id=command_id, request=request, receipt=receipt)
            return receipt
        ok, result, error = _execute_tmux_send(request)
        receipt.update(result)
        receipt["ok"] = bool(ok)
        receipt["delivery_outcome"] = "sent" if ok else "failed"
        if error:
            receipt["error_type"] = error
        _receipt_event(command_id=command_id, request=request, receipt=receipt)
        return receipt

    receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_UNSUPPORTED_KIND)
    _receipt_event(command_id=command_id, request=request, receipt=receipt)
    return receipt
