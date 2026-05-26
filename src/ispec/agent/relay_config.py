from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ispec.agent.relay_constants import (
    FAILURE_MISSING_TARGET,
    FAILURE_TARGET_NOT_ALLOWED,
    RELAY_SCHEMA_VERSION,
)
from ispec.agent.relay_utils import is_truthy, slug
from ispec.cli.env import parse_env_file_text

try:  # Python 3.11+
    import tomllib
except Exception:  # pragma: no cover - Python <3.11 fallback
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CanonicalEnv:
    root: Path | None
    values: dict[str, str]
    sources: dict[str, str]
    loaded_files: list[str]
    errors: list[str]


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

    for key, value in os.environ.items():
        if key.startswith("ISPEC_") or key.startswith("SLACK_") or key in {"DEV_TMUX_SESSION"}:
            values[key] = str(value)
            sources[key] = f"process_env:{key}"

    return CanonicalEnv(root=root, values=values, sources=sources, loaded_files=loaded_files, errors=errors)


def token_info(env: CanonicalEnv) -> tuple[str, dict[str, Any]]:
    for key in ("ISPEC_SLACK_BOT_TOKEN", "SLACK_BOT_TOKEN"):
        token = str(env.values.get(key) or "").strip()
        if token:
            return token, {"present": True, "env_var": key, "source": env.sources.get(key)}
    return "", {"present": False, "env_var": None, "source": None}


def json_object_from_env(env: CanonicalEnv, key: str) -> Any:
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
        alias = slug(str(raw_key))
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


def load_destinations(env: CanonicalEnv) -> dict[str, dict[str, Any]]:
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
        parsed = json_object_from_env(env, key)
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
    requested = slug(alias or "staff")
    if not requested:
        return None, FAILURE_MISSING_TARGET
    destinations = load_destinations(env)
    entry = destinations.get(requested)
    if not isinstance(entry, dict):
        return None, FAILURE_TARGET_NOT_ALLOWED

    requested_type = slug(message_type)
    allowed_raw = entry.get("allowed_message_types")
    if requested_type and isinstance(allowed_raw, list) and allowed_raw:
        allowed = {slug(str(item)) for item in allowed_raw if slug(str(item))}
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


def relay_live_enabled(env: CanonicalEnv | None = None) -> bool:
    if env is not None and "ISPEC_RELAY_LIVE_SEND_ENABLED" in env.values:
        return is_truthy(env.values.get("ISPEC_RELAY_LIVE_SEND_ENABLED"))
    return is_truthy(os.getenv("ISPEC_RELAY_LIVE_SEND_ENABLED"))


def allowed_source_entries(env: CanonicalEnv | None = None) -> list[str]:
    if env is not None and "ISPEC_RELAY_ALLOWED_SOURCES" in env.values:
        raw = str(env.values.get("ISPEC_RELAY_ALLOWED_SOURCES") or "").strip()
    else:
        raw = str(os.getenv("ISPEC_RELAY_ALLOWED_SOURCES") or "").strip()
    if not raw:
        return []
    return [item.strip() for item in re.split(r"[\s,]+", raw) if item.strip()]


def source_policy(request: dict[str, Any], *, env: CanonicalEnv | None = None) -> tuple[bool, dict[str, Any]]:
    source = request.get("source") if isinstance(request.get("source"), dict) else {}
    source_id = str(source.get("id") or "").strip()
    source_kind = str(source.get("kind") or "").strip()
    candidates = {value for value in (source_id, source_kind, f"{source_kind}:{source_id}" if source_kind and source_id else "") if value}
    allowed_entries = allowed_source_entries(env)
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


def slack_timeout_seconds(env: CanonicalEnv) -> float:
    raw = str(env.values.get("ISPEC_SLACK_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 10.0
    try:
        return max(1.0, min(60.0, float(raw)))
    except ValueError:
        return 10.0


def slack_upload_max_bytes(env: CanonicalEnv) -> int:
    raw = str(env.values.get("ISPEC_SLACK_UPLOAD_MAX_BYTES") or "").strip()
    if not raw:
        return 50 * 1024 * 1024
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 50 * 1024 * 1024


def relay_config_probe(*, target_alias: str | None = None, message_type: str | None = None) -> dict[str, Any]:
    env = load_canonical_env()
    _token, token_meta = token_info(env)
    destinations = load_destinations(env)
    sources = allowed_source_entries(env)
    destination, destination_error = (None, None)
    if target_alias:
        destination, destination_error = resolve_slack_destination(env, target_alias, message_type=message_type)
    return {
        "schema_version": RELAY_SCHEMA_VERSION,
        "root": str(env.root) if env.root is not None else None,
        "env_files_loaded": list(env.loaded_files),
        "env_errors": list(env.errors),
        "relay": {
            "live_send_enabled": relay_live_enabled(env),
            "allowed_sources_configured": bool(sources),
            "allowed_sources_count": len(sources),
            "pdf_attachment_upload": {
                "supported": True,
                "max_bytes": slack_upload_max_bytes(env),
                "requires_mode_send": True,
                "requires_confirm": True,
            },
        },
        "slack": {
            "bot_token": token_meta,
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
