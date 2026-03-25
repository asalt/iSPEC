from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ASSISTANT_SCHEDULE_PATH_ENV = "ISPEC_ASSISTANT_SCHEDULE_PATH"
ASSISTANT_SCHEDULE_JSON_ENV = "ISPEC_ASSISTANT_SCHEDULE_JSON"
ASSISTANT_SCHEDULE_DEFAULT_TIMEZONE_ENV = "ISPEC_ASSISTANT_SCHEDULE_DEFAULT_TIMEZONE"


_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSY = {"0", "false", "no", "n", "off"}

_WEEKDAY_ALIASES: dict[str, int] = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}

_WEEKDAY_CANONICAL = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class AssistantSchedule:
    name: str
    weekday: int
    hour: int
    minute: int
    timezone: str
    prompt: str
    allowed_tools: tuple[str, ...]
    required_tool: str | None = None
    grace_seconds: int = 0
    priority: int = 0
    max_attempts: int = 1
    max_tool_calls: int = 4
    enabled: bool = True


@dataclass(frozen=True)
class AssistantScheduleStore:
    source: str
    path: Path | None
    raw_text: str
    error: str | None = None

    @property
    def writable(self) -> bool:
        return self.path is not None and self.error is None


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


def _safe_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    text = str(value).strip()
    if not text or not text.isdigit():
        return None
    parsed = int(text)
    return parsed if parsed >= 0 else None


def _clamp_int(value: int | None, *, default: int, min_value: int, max_value: int) -> int:
    if value is None:
        return default
    return max(min_value, min(max_value, int(value)))


def parse_weekday(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 6 else None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        parsed = int(raw)
        return parsed if 0 <= parsed <= 6 else None
    return _WEEKDAY_ALIASES.get(raw)


def format_weekday(value: int) -> str:
    if 0 <= int(value) <= 6:
        return _WEEKDAY_CANONICAL[int(value)]
    raise ValueError(f"Invalid weekday: {value!r}")


def parse_hhmm(value: Any) -> tuple[int, int] | None:
    raw = str(value).strip()
    if not raw or ":" not in raw:
        return None
    left, right = raw.split(":", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None
    try:
        hour = int(left)
        minute = int(right)
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def format_hhmm(hour: int, minute: int) -> str:
    return f"{int(hour):02d}:{int(minute):02d}"


def normalize_schedule_tool_names(value: Any) -> tuple[str, ...]:
    raw_items: list[str] = []
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if not item or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
    return tuple(cleaned)


def assistant_schedule_default_timezone() -> str:
    return (os.getenv(ASSISTANT_SCHEDULE_DEFAULT_TIMEZONE_ENV) or "").strip() or "UTC"


def assistant_schedule_path() -> Path | None:
    raw = (os.getenv(ASSISTANT_SCHEDULE_PATH_ENV) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def assistant_schedule_store() -> AssistantScheduleStore:
    path = assistant_schedule_path()
    if path is not None:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raw_text = ""
        except Exception as exc:
            return AssistantScheduleStore(source="path", path=path, raw_text="", error=f"{type(exc).__name__}: {exc}")
        return AssistantScheduleStore(source="path", path=path, raw_text=raw_text)

    raw = (os.getenv(ASSISTANT_SCHEDULE_JSON_ENV) or "").strip()
    if raw:
        return AssistantScheduleStore(source="env", path=None, raw_text=raw)
    return AssistantScheduleStore(source="none", path=None, raw_text="")


def _parse_schedule_json(raw_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    text = str(raw_text or "").strip()
    if not text:
        return [], []
    try:
        parsed = json.loads(text)
    except Exception as exc:
        return [], [f"Invalid schedule JSON: {type(exc).__name__}: {exc}"]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [], ["Schedule JSON must be an object or a list of objects."]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            errors.append(f"Entry {index} is not an object.")
            continue
        rows.append(dict(item))
    return rows, errors


def canonicalize_schedule_row(
    row: dict[str, Any],
    *,
    default_timezone: str | None = None,
    known_tool_names: set[str] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    default_tz = (default_timezone or "").strip() or "UTC"

    name = str(row.get("name") or "").strip()
    if not name:
        errors.append("name is required.")

    weekday = parse_weekday(row.get("weekday"))
    if weekday is None:
        errors.append("weekday must be 0-6 or a weekday name like tue.")

    hhmm = parse_hhmm(row.get("time"))
    if hhmm is None:
        errors.append("time must be HH:MM in 24-hour format.")

    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        errors.append("prompt is required.")

    allowed_tools = list(normalize_schedule_tool_names(row.get("allowed_tools")))
    required_tool = str(row.get("required_tool") or "").strip() or None
    if required_tool and required_tool not in allowed_tools:
        allowed_tools.append(required_tool)
    if not allowed_tools:
        errors.append("allowed_tools must include at least one tool.")

    if known_tool_names is not None and allowed_tools:
        unknown_tools = sorted({tool for tool in allowed_tools if tool not in known_tool_names})
        if unknown_tools:
            errors.append(f"Unknown allowed_tools: {unknown_tools}")
    if required_tool and known_tool_names is not None and required_tool not in known_tool_names:
        errors.append(f"Unknown required_tool: {required_tool}")

    timezone = str(row.get("timezone") or "").strip() or default_tz
    try:
        ZoneInfo(timezone)
    except Exception:
        errors.append(f"Invalid timezone: {timezone!r}")

    grace_seconds = _clamp_int(
        _safe_nonnegative_int(row.get("grace_seconds")),
        default=0,
        min_value=0,
        max_value=3600,
    )
    priority_raw = _safe_nonnegative_int(row.get("priority"))
    if priority_raw is None and row.get("priority") not in {None, ""}:
        try:
            priority_raw = int(str(row.get("priority")).strip())
        except Exception:
            priority_raw = None
    priority = _clamp_int(priority_raw, default=0, min_value=-10, max_value=10)
    max_attempts = _clamp_int(
        _safe_nonnegative_int(row.get("max_attempts")),
        default=1,
        min_value=1,
        max_value=10,
    )
    max_tool_calls = _clamp_int(
        _safe_nonnegative_int(row.get("max_tool_calls")),
        default=4,
        min_value=1,
        max_value=12,
    )
    enabled = _normalize_bool(row.get("enabled"), default=True)

    if errors:
        return None, errors

    hour, minute = hhmm or (0, 0)
    canonical: dict[str, Any] = {
        "name": name,
        "weekday": format_weekday(int(weekday)),
        "time": format_hhmm(hour, minute),
        "timezone": timezone,
        "prompt": prompt,
        "allowed_tools": allowed_tools,
        "max_tool_calls": int(max_tool_calls),
        "priority": int(priority),
        "grace_seconds": int(grace_seconds),
        "max_attempts": int(max_attempts),
        "enabled": bool(enabled),
    }
    if required_tool:
        canonical["required_tool"] = required_tool
    return canonical, []


def schedule_from_row(row: dict[str, Any]) -> AssistantSchedule | None:
    weekday = parse_weekday(row.get("weekday"))
    hhmm = parse_hhmm(row.get("time"))
    if weekday is None or hhmm is None:
        return None
    allowed_tools = tuple(normalize_schedule_tool_names(row.get("allowed_tools")))
    if not allowed_tools:
        return None
    required_tool = str(row.get("required_tool") or "").strip() or None
    hour, minute = hhmm
    return AssistantSchedule(
        name=str(row.get("name") or "").strip(),
        weekday=int(weekday),
        hour=int(hour),
        minute=int(minute),
        timezone=str(row.get("timezone") or "").strip() or assistant_schedule_default_timezone(),
        prompt=str(row.get("prompt") or "").strip(),
        allowed_tools=allowed_tools,
        required_tool=required_tool,
        grace_seconds=_clamp_int(_safe_nonnegative_int(row.get("grace_seconds")), default=0, min_value=0, max_value=3600),
        priority=_clamp_int(
            int(str(row.get("priority")).strip()) if str(row.get("priority") or "").strip().lstrip("-").isdigit() else None,
            default=0,
            min_value=-10,
            max_value=10,
        ),
        max_attempts=_clamp_int(_safe_nonnegative_int(row.get("max_attempts")), default=1, min_value=1, max_value=10),
        max_tool_calls=_clamp_int(_safe_nonnegative_int(row.get("max_tool_calls")), default=4, min_value=1, max_value=12),
        enabled=_normalize_bool(row.get("enabled"), default=True),
    )


def load_assistant_schedules(*, known_tool_names: set[str] | None = None) -> list[AssistantSchedule]:
    store = assistant_schedule_store()
    if store.error:
        return []

    rows, parse_errors = _parse_schedule_json(store.raw_text)
    if parse_errors:
        return []

    default_timezone = assistant_schedule_default_timezone()
    schedules: list[AssistantSchedule] = []
    seen: set[str] = set()
    for row in rows:
        canonical, errors = canonicalize_schedule_row(
            row,
            default_timezone=default_timezone,
            known_tool_names=known_tool_names,
        )
        if canonical is None or errors:
            continue
        name = str(canonical.get("name") or "").strip()
        if not name or name in seen:
            continue
        schedule = schedule_from_row(canonical)
        if schedule is None:
            continue
        schedules.append(schedule)
        seen.add(name)
    return schedules


def list_assistant_schedule_rows(*, known_tool_names: set[str] | None = None) -> tuple[list[dict[str, Any]], list[str], AssistantScheduleStore]:
    store = assistant_schedule_store()
    if store.error:
        return [], [store.error], store

    rows, parse_errors = _parse_schedule_json(store.raw_text)
    default_timezone = assistant_schedule_default_timezone()
    valid_rows: list[dict[str, Any]] = []
    errors = list(parse_errors)
    seen: set[str] = set()

    for index, row in enumerate(rows, start=1):
        canonical, row_errors = canonicalize_schedule_row(
            row,
            default_timezone=default_timezone,
            known_tool_names=known_tool_names,
        )
        if canonical is None:
            errors.extend(f"Entry {index}: {message}" for message in row_errors)
            continue
        name = str(canonical.get("name") or "").strip()
        if name in seen:
            errors.append(f"Entry {index}: duplicate schedule name {name!r}.")
            continue
        valid_rows.append(canonical)
        seen.add(name)

    valid_rows.sort(key=lambda item: (parse_weekday(item.get("weekday")) or 99, str(item.get("time") or ""), str(item.get("name") or "")))
    return valid_rows, errors, store


def load_assistant_schedule_rows_for_write(
    *,
    known_tool_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], Path | None]:
    path = assistant_schedule_path()
    if path is None:
        return [], [f"Set {ASSISTANT_SCHEDULE_PATH_ENV}=<path> to enable schedule editing."], None

    store = assistant_schedule_store()
    if store.error:
        return [], [store.error], path
    if store.source != "path":
        return [], [f"Set {ASSISTANT_SCHEDULE_PATH_ENV}=<path> to enable schedule editing."], path

    rows, errors, _ = list_assistant_schedule_rows(known_tool_names=known_tool_names)
    return rows, errors, path


def write_assistant_schedule_rows(rows: list[dict[str, Any]], *, path: Path) -> Path:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)
    return path
