from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ispec.config.paths import resolve_log_dir
from ispec.logging import get_logger


logger = get_logger(__name__)


def inference_usage_logging_enabled() -> bool:
    raw = str(os.getenv("ISPEC_INFERENCE_USAGE_LOG_ENABLED") or "").strip().lower()
    if raw in {"", "1", "true", "yes", "on"}:
        return True
    return False


def resolve_inference_usage_log_dir() -> Path:
    raw = str(os.getenv("ISPEC_INFERENCE_USAGE_LOG_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    base = Path(resolve_log_dir().path or (Path.home() / ".ispec" / "logs")).expanduser()
    return base / "inference-usage"


def _jsonl_path(now: datetime | None = None) -> Path:
    current = now or datetime.now(UTC)
    return resolve_inference_usage_log_dir() / f"usage-{current.strftime('%Y%m%d')}.jsonl"


def _clean_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _clean_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            continue
        cleaned[key] = _clean_scalar(item)
    return cleaned or None


def record_inference_usage_event(
    *,
    provider: str | None,
    model: str | None,
    meta: dict[str, Any] | None,
    ok: bool,
    error: str | None = None,
    observability_context: dict[str, Any] | None = None,
) -> None:
    if not inference_usage_logging_enabled():
        return

    now = datetime.now(UTC)
    event: dict[str, Any] = {
        "ts_utc": now.isoformat(),
        "provider": (provider or "").strip() or None,
        "model": (model or "").strip() or None,
        "ok": bool(ok),
    }
    if error:
        event["error"] = str(error)

    meta_dict = meta if isinstance(meta, dict) else {}
    if "elapsed_ms" in meta_dict:
        event["elapsed_ms"] = _clean_scalar(meta_dict.get("elapsed_ms"))
    usage = _clean_usage(meta_dict.get("usage"))
    if usage is not None:
        event["usage"] = usage
    fallback = meta_dict.get("fallback")
    if isinstance(fallback, dict) and fallback:
        event["fallback"] = {
            str(key): _clean_scalar(value)
            for key, value in fallback.items()
            if isinstance(key, str) and key
        }
    for key in ("tool_call_dialect", "tool_parser_fallback_used", "tool_parser_fallback_shape"):
        if key in meta_dict:
            event[key] = _clean_scalar(meta_dict.get(key))

    context = observability_context if isinstance(observability_context, dict) else {}
    for key in (
        "surface",
        "session_id",
        "message_id",
        "run_id",
        "command_id",
        "task",
        "stage",
        "prompt_family",
        "prompt_version_num",
        "prompt_version_id",
        "prompt_sha256",
        "prompt_source_path",
        "prompt_binding",
    ):
        if key in context:
            event[key] = _clean_scalar(context.get(key))

    path = _jsonl_path(now)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        logger.exception("Failed to append inference usage event to %s", path)
