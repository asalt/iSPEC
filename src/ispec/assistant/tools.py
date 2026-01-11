from __future__ import annotations

from datetime import date, datetime, time, timedelta
import enum
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ispec.assistant.context import person_summary, project_summary
from ispec.db.models import (
    AuthUser,
    E2G,
    Experiment,
    ExperimentRun,
    Person,
    Project,
    ProjectComment,
    UserRole,
)
from ispec.schedule.models import ScheduleRequest, ScheduleSlot


TOOL_CALL_PREFIX = "TOOL_CALL"
TOOL_RESULT_PREFIX = "TOOL_RESULT"

CENTRAL_TZ = ZoneInfo("America/Chicago")
UTC_TZ = ZoneInfo("UTC")

SCHEDULE_SLOT_STATUSES = {"available", "booked", "closed"}
SCHEDULE_REQUEST_STATUSES = {"requested", "confirmed", "declined", "cancelled"}
PROJECT_STATUSES = {"inquiry", "consultation", "waiting", "processing", "analysis", "summary", "closed", "hibernate"}


class ToolScope(str, enum.Enum):
    public = "public"
    user = "user"
    staff = "staff"
    admin = "admin"


_TOOL_SCOPES: dict[str, ToolScope] = {
    "project_counts_snapshot": ToolScope.staff,
    "latest_activity": ToolScope.staff,
    "billing_category_counts": ToolScope.staff,
    "db_file_stats": ToolScope.staff,
    "count_all_projects": ToolScope.staff,
    "count_current_projects": ToolScope.staff,
    "project_status_counts": ToolScope.staff,
    "latest_projects": ToolScope.staff,
    "latest_project_comments": ToolScope.staff,
    "search_projects": ToolScope.staff,
    "projects": ToolScope.staff,
    "get_project": ToolScope.staff,
    "search_api": ToolScope.user,
    "experiments_for_project": ToolScope.staff,
    "latest_experiments": ToolScope.staff,
    "get_experiment": ToolScope.staff,
    "latest_experiment_runs": ToolScope.staff,
    "get_experiment_run": ToolScope.staff,
    "e2g_search_genes_in_project": ToolScope.staff,
    "e2g_gene_in_project": ToolScope.staff,
    "search_people": ToolScope.staff,
    "get_person": ToolScope.staff,
    "list_schedule_slots": ToolScope.user,
    "list_schedule_requests": ToolScope.admin,
    "get_schedule_request": ToolScope.admin,
    "repo_list_files": ToolScope.staff,
    "repo_search": ToolScope.staff,
    "repo_read_file": ToolScope.staff,
    "create_project_comment": ToolScope.staff,
}

_WRITE_TOOL_NAMES: set[str] = {"create_project_comment"}


_REPO_TOOLS_ENV = "ISPEC_ASSISTANT_ENABLE_REPO_TOOLS"
_REPO_ROOT_ENV = "ISPEC_ASSISTANT_REPO_ROOT"
_REPO_TOOL_DEFAULT_PATH = "iSPEC/src"
_REPO_TOOL_DEFAULT_PATH_STANDALONE = "src"
_REPO_MAX_FILE_BYTES = 250_000
_REPO_DENY_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".venv", "venv"}
_REPO_DENY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".sqlite",
    ".db",
    ".pkl",
    ".parquet",
    ".pem",
    ".key",
}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _repo_tools_enabled() -> bool:
    return _is_truthy(os.getenv(_REPO_TOOLS_ENV))


def _assistant_repo_root() -> Path | None:
    env = (os.getenv(_REPO_ROOT_ENV) or "").strip()
    if env:
        try:
            return Path(env).expanduser().resolve()
        except Exception:
            return None

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "iSPEC" / "src" / "ispec").is_dir():
            return parent
        if (parent / "src" / "ispec").is_dir():
            return parent
    return None


def _repo_default_path(repo_root: Path) -> str:
    if (repo_root / "iSPEC" / "src" / "ispec").is_dir():
        return _REPO_TOOL_DEFAULT_PATH
    if (repo_root / "src" / "ispec").is_dir():
        return _REPO_TOOL_DEFAULT_PATH_STANDALONE
    return _REPO_TOOL_DEFAULT_PATH


def _safe_repo_rel_path(repo_root: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.startswith(("~", "/")):
        return None
    candidate = (repo_root / value).resolve()
    try:
        if not candidate.is_relative_to(repo_root):
            return None
    except Exception:
        return None
    return candidate


def _repo_path_denied(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection(_REPO_DENY_DIRS):
        return True
    name = path.name
    if name.startswith(".env"):
        return True
    suffix = path.suffix.lower()
    if suffix in _REPO_DENY_SUFFIXES:
        return True
    return False


def _iter_repo_files(base: Path, *, limit: int) -> list[Path]:
    if limit <= 0:
        return []
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(base, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in _REPO_DENY_DIRS]
        for filename in filenames:
            candidate = Path(root) / filename
            if _repo_path_denied(candidate):
                continue
            files.append(candidate)
            if len(files) >= limit:
                return files
    return files


def _repo_list_files(*, repo_root: Path, query: str | None, path: str | None, limit: int) -> dict[str, Any]:
    default_path = _repo_default_path(repo_root)
    base = _safe_repo_rel_path(repo_root, path) if path else repo_root / default_path
    if base is None:
        return {"ok": False, "tool": "repo_list_files", "error": "Invalid path."}
    if not base.exists():
        return {"ok": False, "tool": "repo_list_files", "error": "Path not found."}

    limit = _clamp_int(_safe_int(limit), default=200, min_value=1, max_value=2000)
    needle = (query or "").strip().lower() or None

    results: list[str] = []
    candidates = _iter_repo_files(base, limit=5000)
    for candidate in candidates:
        try:
            rel = candidate.resolve().relative_to(repo_root).as_posix()
        except Exception:
            continue
        if needle and needle not in rel.lower():
            continue
        results.append(rel)
        if len(results) >= limit:
            break

    results.sort()
    return {
        "ok": True,
        "tool": "repo_list_files",
        "result": {"repo_root": repo_root.name, "path": (path or default_path), "files": results},
    }


def _repo_search_python(
    *, repo_root: Path, query: str, base: Path, limit: int, regex: bool, ignore_case: bool
) -> dict[str, Any]:
    limit = _clamp_int(_safe_int(limit), default=50, min_value=1, max_value=500)

    pattern = None
    lowered_query = query.lower()
    if regex:
        try:
            flags = re.IGNORECASE if ignore_case else 0
            pattern = re.compile(query, flags=flags)
        except re.error as exc:
            return {"ok": False, "tool": "repo_search", "error": f"Invalid regex: {exc}"}

    matches: list[dict[str, Any]] = []
    scanned = 0
    candidates = _iter_repo_files(base, limit=10_000)
    for file_path in sorted(candidates):
        scanned += 1
        try:
            stat = file_path.stat()
        except Exception:
            continue
        if stat.st_size > _REPO_MAX_FILE_BYTES:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            haystack = line
            hit = False
            if pattern is not None:
                hit = pattern.search(haystack) is not None
            else:
                if ignore_case:
                    hit = lowered_query in haystack.lower()
                else:
                    hit = query in haystack
            if not hit:
                continue
            try:
                rel = file_path.resolve().relative_to(repo_root).as_posix()
            except Exception:
                rel = file_path.as_posix()
            matches.append({"path": rel, "line": line_no, "text": line[:300]})
            if len(matches) >= limit:
                return {
                    "ok": True,
                    "tool": "repo_search",
                    "result": {
                        "query": query,
                        "path": base.resolve().relative_to(repo_root).as_posix(),
                        "matches": matches,
                        "truncated": True,
                        "scanned_files": scanned,
                        "backend": "python",
                    },
                }

    return {
        "ok": True,
        "tool": "repo_search",
        "result": {
            "query": query,
            "path": base.resolve().relative_to(repo_root).as_posix(),
            "matches": matches,
            "truncated": False,
            "scanned_files": scanned,
            "backend": "python",
        },
    }


def _repo_search_rg(
    *, repo_root: Path, query: str, path: str, limit: int, regex: bool, ignore_case: bool
) -> dict[str, Any] | None:
    if shutil.which("rg") is None:
        return None
    limit = _clamp_int(_safe_int(limit), default=50, min_value=1, max_value=500)

    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color=never",
        "--max-columns=300",
        "--max-columns-preview",
        "--max-filesize",
        f"{_REPO_MAX_FILE_BYTES}",
    ]
    for denied in _REPO_DENY_DIRS:
        cmd.extend(["--glob", f"!**/{denied}/**"])
    cmd.extend(["--glob", "!**/.env*"])
    if ignore_case:
        cmd.append("--ignore-case")
    if not regex:
        cmd.append("--fixed-strings")
    cmd.extend(["--", query, path])

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode not in {0, 1}:
        return None

    matches: list[dict[str, Any]] = []
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            continue
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        file_part, line_part, text_part = parts[0], parts[1], parts[2]
        try:
            line_no = int(line_part)
        except ValueError:
            continue
        matches.append({"path": file_part, "line": line_no, "text": text_part[:300]})
        if len(matches) >= limit:
            break

    truncated = len(matches) >= limit and bool(completed.stdout)
    return {
        "ok": True,
        "tool": "repo_search",
        "result": {
            "query": query,
            "path": path,
            "matches": matches,
            "truncated": truncated,
            "backend": "rg",
        },
    }


def _repo_search(
    *, repo_root: Path, query: str, path: str | None, limit: int, regex: bool, ignore_case: bool
) -> dict[str, Any]:
    query_clean = (query or "").strip()
    if not query_clean:
        return {"ok": False, "tool": "repo_search", "error": "query is required."}

    base_path_raw = (path or "").strip() or _repo_default_path(repo_root)
    base = _safe_repo_rel_path(repo_root, base_path_raw)
    if base is None:
        return {"ok": False, "tool": "repo_search", "error": "Invalid path."}
    if not base.exists():
        return {"ok": False, "tool": "repo_search", "error": "Path not found."}
    if _repo_path_denied(base):
        return {"ok": False, "tool": "repo_search", "error": "Path not allowed."}

    rg_result = _repo_search_rg(
        repo_root=repo_root,
        query=query_clean,
        path=base_path_raw,
        limit=limit,
        regex=regex,
        ignore_case=ignore_case,
    )
    if rg_result is not None:
        return rg_result
    return _repo_search_python(
        repo_root=repo_root,
        query=query_clean,
        base=base,
        limit=limit,
        regex=regex,
        ignore_case=ignore_case,
    )


def _repo_read_file(*, repo_root: Path, path: str, start_line: int, max_lines: int) -> dict[str, Any]:
    candidate = _safe_repo_rel_path(repo_root, path)
    if candidate is None:
        return {"ok": False, "tool": "repo_read_file", "error": "Invalid path."}
    if _repo_path_denied(candidate):
        return {"ok": False, "tool": "repo_read_file", "error": "Path not allowed."}
    if not candidate.exists() or not candidate.is_file():
        return {"ok": False, "tool": "repo_read_file", "error": "File not found."}

    try:
        stat = candidate.stat()
    except Exception:
        stat = None
    if stat is not None and int(stat.st_size) > _REPO_MAX_FILE_BYTES:
        return {"ok": False, "tool": "repo_read_file", "error": "File too large."}

    start_line = _clamp_int(_safe_int(start_line), default=1, min_value=1, max_value=1_000_000)
    max_lines = _clamp_int(_safe_int(max_lines), default=200, min_value=1, max_value=500)

    try:
        raw = candidate.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "tool": "repo_read_file", "error": f"{type(exc).__name__}: {exc}"}

    lines = raw.splitlines()
    start_index = start_line - 1
    end_index = min(len(lines), start_index + max_lines)
    snippet_lines = lines[start_index:end_index] if start_index < len(lines) else []
    snippet = "\n".join(snippet_lines)
    rel = candidate.resolve().relative_to(repo_root).as_posix()
    return {
        "ok": True,
        "tool": "repo_read_file",
        "result": {
            "path": rel,
            "start_line": start_line,
            "end_line": (start_line + len(snippet_lines) - 1) if snippet_lines else start_line,
            "total_lines": len(lines),
            "content": snippet,
        },
    }


def _scope_error(scope: ToolScope, user: AuthUser | None) -> str | None:
    if user is None:
        return None
    if scope == ToolScope.public:
        return None
    if scope == ToolScope.user:
        return None
    if scope == ToolScope.staff:
        if user.role in {UserRole.viewer, UserRole.editor, UserRole.admin}:
            return None
        return "Staff access required."
    if scope == ToolScope.admin:
        if user.role == UserRole.admin:
            return None
        return "Admin access required."
    return "Access denied."


def tool_prompt() -> str:
    """Short tool list for the system prompt."""

    lines = [
        "Available tools:",
        "- (Most tools are read-only; create_project_comment writes project history.)",
        "- project_counts_snapshot(max_categories: int = 20)",
        "- latest_activity(limit: int = 20, kinds: list[str] | None = None, current_only: bool = false)",
        "- billing_category_counts(current_only: bool = false, limit: int = 20)",
        "- db_file_stats()  # show sqlite DB file sizes",
        "- count_all_projects()  # total projects across all statuses/flags",
        "- count_current_projects()  # current projects only",
        "- project_status_counts(current_only: bool = false)",
        "- latest_projects(sort: str = 'modified', limit: int = 10, current_only: bool = false)",
        "- latest_project_comments(limit: int = 10, project_id: int | None = None)",
        "- search_projects(query: str, limit: int = 5)",
        "- projects(project_id: int)  # alias for get_project",
        "- get_project(id: int)",
        "- search_api(query: str, limit: int = 10)  # search FastAPI/OpenAPI endpoints",
        "- create_project_comment(project_id: int, comment: str, comment_type: str | None = None, confirm: bool = true)  # write: requires explicit user request",
    ]

    if _repo_tools_enabled():
        lines.extend(
            [
                f"- repo_list_files(query: str | None = None, path: str | None = None, limit: int = 200)  # dev-only; set {_REPO_TOOLS_ENV}=1",
                f"- repo_search(query: str, path: str | None = None, limit: int = 50, regex: bool = false, ignore_case: bool = true)  # dev-only; set {_REPO_TOOLS_ENV}=1",
                f"- repo_read_file(path: str, start_line: int = 1, max_lines: int = 200)  # dev-only; set {_REPO_TOOLS_ENV}=1",
            ]
        )

    lines.extend(
        [
            "- experiments_for_project(project_id: int, limit: int = 20)",
            "- latest_experiments(limit: int = 5)",
            "- get_experiment(id: int)",
            "- latest_experiment_runs(limit: int = 5)",
            "- get_experiment_run(id: int)",
            "- e2g_search_genes_in_project(project_id: int, query: str, limit: int = 10)",
            "- e2g_gene_in_project(project_id: int, gene_id: int, limit: int = 50)",
            "- search_people(query: str, limit: int = 5)",
            "- get_person(id: int)",
            "- list_schedule_slots(start: YYYY-MM-DD, end: YYYY-MM-DD, status: str | None = None, limit: int = 50)",
            "- list_schedule_requests(limit: int = 20, status: str | None = None)  # admin-only",
            "- get_schedule_request(id: int)  # admin-only",
        ]
    )

    return "\n".join(lines) + "\n"


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed >= 0 else None
    return None


def _safe_str(value: Any, *, max_len: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _clamp_int(value: int | None, *, default: int, min_value: int, max_value: int) -> int:
    if value is None:
        return default
    return max(min_value, min(max_value, value))


def _safe_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _as_utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC_TZ)
    return value.astimezone(UTC_TZ)


def _range_bounds_local(start: date, end: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(start, time.min, tzinfo=CENTRAL_TZ)
    end_local = datetime.combine(end, time.max, tzinfo=CENTRAL_TZ)
    return start_local.astimezone(UTC_TZ).replace(tzinfo=None), end_local.astimezone(UTC_TZ).replace(
        tzinfo=None
    )


def _require_admin(user: AuthUser | None) -> str | None:
    if user is None:
        return "Not authenticated."
    if user.role != UserRole.admin:
        return "Admin access required."
    return None


def _human_bytes(size: int | None) -> str | None:
    if size is None:
        return None
    if size < 0:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit_index = 0
    while value >= 1024 and unit_index < (len(units) - 1):
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _sqlite_path_from_session(db: Session) -> Path | None:
    try:
        bind = db.get_bind()
        url = getattr(bind, "url", None)
        if url is None:
            return None
        if getattr(url, "get_backend_name", lambda: None)() != "sqlite":
            return None
        database = getattr(url, "database", None)
        if not database or database == ":memory:":
            return None
        return Path(str(database))
    except Exception:
        return None


def _stat_path(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "size_bytes": None, "size_human": None}
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"exists": False, "size_bytes": None, "size_human": None}
    except Exception as exc:
        return {"exists": False, "size_bytes": None, "size_human": None, "error": f"{type(exc).__name__}: {exc}"}
    size = int(stat.st_size)
    return {"exists": True, "size_bytes": size, "size_human": _human_bytes(size)}


def _coerce_str_list(value: Any, *, max_items: int) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return [text[:256]]
    if isinstance(value, list):
        items: list[str] = []
        for item in value[: max(0, max_items)]:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            items.append(text[:256])
        return items or None
    return None


def _extract_fenced_block(text: str, *, label: str) -> str | None:
    """Return the first fenced code block matching ``label`` (without the fences)."""

    if not text:
        return None

    start_marker = f"```{label}"
    in_block = False
    buffer: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not in_block:
            if stripped.startswith(start_marker):
                in_block = True
                continue
            continue
        if stripped.startswith("```"):
            break
        buffer.append(line)

    content = "\n".join(buffer).strip()
    return content or None


_TOOL_CALL_FUNC_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$")


def _parse_tool_call_func_args(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    args: dict[str, Any] = {}
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            return None
        key, value_raw = part.split("=", 1)
        key = key.strip()
        value_raw = value_raw.strip()
        if not key:
            return None

        value: Any
        lower = value_raw.lower()
        if lower in {"true", "false"}:
            value = lower == "true"
        elif lower in {"none", "null"}:
            value = None
        else:
            if (value_raw.startswith('"') and value_raw.endswith('"')) or (
                value_raw.startswith("'") and value_raw.endswith("'")
            ):
                value = value_raw[1:-1]
            else:
                try:
                    value = int(value_raw)
                except ValueError:
                    try:
                        value = float(value_raw)
                    except ValueError:
                        value = value_raw

        args[key] = value

    return args


def _parse_tool_call_fenced(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse common model outputs like:

      ```tool_calls
      search_projects(query="example")
      ```
    """

    block = _extract_fenced_block(text, label="tool_calls") or _extract_fenced_block(text, label="tool_call")
    if not block:
        return None

    stripped = block.strip()
    if stripped.startswith(("{", "[")):
        try:
            payload = json.loads(stripped)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("tool")
            args = payload.get("arguments") or payload.get("args") or {}
            if isinstance(name, str) and name.strip() and isinstance(args, dict):
                return name.strip(), args

        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                name = first.get("name") or first.get("tool")
                args = first.get("arguments") or first.get("args") or {}
                if isinstance(name, str) and name.strip() and isinstance(args, dict):
                    return name.strip(), args

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower().strip().rstrip(":")
        if lowered in {"tool_calls", "tool_call"}:
            continue
        match = _TOOL_CALL_FUNC_RE.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        args_raw = match.group(2).strip()
        args = _parse_tool_call_func_args(args_raw)
        if args is None:
            return None
        return name, args

    return None


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a TOOL_CALL request from assistant output.

    Expected format (single line, may appear anywhere in the response):
      TOOL_CALL {"name":"...","arguments":{...}}
    """

    line = extract_tool_call_line(text)
    if not line:
        return _parse_tool_call_fenced(text)

    raw = line[len(TOOL_CALL_PREFIX) :].strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    name = payload.get("name") or payload.get("tool")
    args = payload.get("arguments") or payload.get("args") or {}
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(args, dict):
        return None
    return name.strip(), args


def extract_tool_call_line(text: str) -> str | None:
    """Return the first TOOL_CALL line, if any.

    We intentionally support tool calls that appear alongside extra text
    because some models will not reliably output "tool call only" despite
    being instructed to do so.
    """

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(TOOL_CALL_PREFIX):
            return line
    return None


def format_tool_result_message(tool_name: str, payload: dict[str, Any]) -> str:
    return f"{TOOL_RESULT_PREFIX} {tool_name} (JSON):\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def run_tool(
    *,
    name: str,
    args: dict[str, Any],
    core_db: Session,
    schedule_db: Session | None = None,
    omics_db: Session | None = None,
    user: AuthUser | None = None,
    api_schema: dict[str, Any] | None = None,
    user_message: str | None = None,
) -> dict[str, Any]:
    # the large try block is not great - but these if statements for
    # resolving tool call results is probably fine
    # also not great to define closure inside as that makes it more difficult to test
    try:
        if name == "projects":
            project_id = _safe_int(args.get("id"))
            if project_id is None:
                project_id = _safe_int(args.get("project_id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id (or project_id)"}
            name = "get_project"
            args = {"id": project_id}

        scope = _TOOL_SCOPES.get(name, ToolScope.staff)
        scope_error = _scope_error(scope, user)
        if scope_error:
            return {"ok": False, "tool": name, "error": scope_error}

        if name in {"repo_list_files", "repo_search", "repo_read_file"}:
            if not _repo_tools_enabled():
                return {
                    "ok": False,
                    "tool": name,
                    "error": f"Repo tools are disabled. Set {_REPO_TOOLS_ENV}=1 to enable them.",
                }
            repo_root = _assistant_repo_root()
            if repo_root is None:
                return {"ok": False, "tool": name, "error": "Repo root not found."}

            if name == "repo_list_files":
                query = _safe_str(args.get("query"), max_len=256)
                path = _safe_str(args.get("path"), max_len=256)
                limit = _safe_int(args.get("limit")) or 200
                return _repo_list_files(repo_root=repo_root, query=query, path=path, limit=limit)

            if name == "repo_search":
                query_raw = _safe_str(args.get("query"), max_len=2048) or ""
                path = _safe_str(args.get("path"), max_len=256)
                limit = _safe_int(args.get("limit")) or 50
                regex = bool(args.get("regex"))
                ignore_case = True if args.get("ignore_case") is None else bool(args.get("ignore_case"))
                return _repo_search(
                    repo_root=repo_root,
                    query=query_raw,
                    path=path,
                    limit=limit,
                    regex=regex,
                    ignore_case=ignore_case,
                )

            file_path = _safe_str(args.get("path"), max_len=512) or ""
            start_line = _safe_int(args.get("start_line")) or 1
            max_lines = _safe_int(args.get("max_lines")) or 200
            return _repo_read_file(
                repo_root=repo_root,
                path=file_path,
                start_line=start_line,
                max_lines=max_lines,
            )

        if name == "project_counts_snapshot":
            max_categories = _clamp_int(
                _safe_int(args.get("max_categories")),
                default=20,
                min_value=1,
                max_value=100,
            )
            total_count = int(core_db.query(func.count(Project.id)).scalar() or 0)
            current_count = int(
                core_db.query(func.count(Project.id))
                .filter(Project.prj_Current_FLAG.is_(True))
                .scalar()
                or 0
            )

            status_rows = (
                core_db.query(Project.prj_Status, func.count(Project.id))
                .group_by(Project.prj_Status)
                .all()
            )
            status_total = [
                {"status": (status if status is not None else None), "count": int(count or 0)}
                for status, count in status_rows
            ]
            status_total.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["status"] is None else str(item["status"]),
                )
            )

            status_rows_current = (
                core_db.query(Project.prj_Status, func.count(Project.id))
                .filter(Project.prj_Current_FLAG.is_(True))
                .group_by(Project.prj_Status)
                .all()
            )
            status_current = [
                {"status": (status if status is not None else None), "count": int(count or 0)}
                for status, count in status_rows_current
            ]
            status_current.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["status"] is None else str(item["status"]),
                )
            )

            price_level_rows = (
                core_db.query(Project.prj_ProjectPriceLevel, func.count(Project.id))
                .group_by(Project.prj_ProjectPriceLevel)
                .all()
            )
            price_levels = [
                {
                    "category": (category.strip() if isinstance(category, str) and category.strip() else None),
                    "count": int(count or 0),
                }
                for category, count in price_level_rows
            ]
            price_levels.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["category"] is None else str(item["category"]),
                )
            )
            if len(price_levels) > max_categories:
                price_levels = price_levels[:max_categories]

            ready_to_bill_total = int(
                core_db.query(func.count(Project.id))
                .filter(Project.prj_Billing_ReadyToBill.is_(True))
                .scalar()
                or 0
            )
            ready_to_bill_current = int(
                core_db.query(func.count(Project.id))
                .filter(Project.prj_Current_FLAG.is_(True))
                .filter(Project.prj_Billing_ReadyToBill.is_(True))
                .scalar()
                or 0
            )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "generated_at": datetime.now(tz=UTC_TZ).isoformat(),
                    "projects": {
                        "total": total_count,
                        "current": current_count,
                        "status_counts_total": status_total,
                        "status_counts_current": status_current,
                        "billing_ready_to_bill": {
                            "total": ready_to_bill_total,
                            "current": ready_to_bill_current,
                        },
                        "billing_categories": {
                            "field": "prj_ProjectPriceLevel",
                            "max_categories": max_categories,
                            "items": price_levels,
                        },
                    },
                },
            }

        if name == "create_project_comment":
            if user is None:
                return {"ok": False, "tool": name, "error": "Not authenticated."}
            if user.role in {UserRole.viewer, UserRole.client}:
                return {"ok": False, "tool": name, "error": "Write access required."}

            confirm = args.get("confirm")
            if confirm is not True:
                return {"ok": False, "tool": name, "error": "confirm=true is required to write project history."}

            user_msg = (user_message or "").strip().lower()
            explicit = False
            if user_msg:
                if re.search(r"\b(save|log|record|add)\b", user_msg) and re.search(
                    r"\b(history|comment|note|meeting)\b", user_msg
                ):
                    explicit = True
            if not explicit:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "User did not explicitly request saving to project history.",
                }

            project_id = _safe_int(args.get("project_id"))
            if project_id is None or project_id <= 0:
                return {"ok": False, "tool": name, "error": "project_id is required."}

            comment_text = _safe_str(args.get("comment"), max_len=20_000)
            if comment_text is None or not comment_text.strip():
                return {"ok": False, "tool": name, "error": "comment text is required."}

            comment_type = _safe_str(args.get("comment_type"), max_len=64) or "assistant_note"

            project = core_db.get(Project, int(project_id))
            if project is None:
                return {"ok": False, "tool": name, "error": f"Project {project_id} not found."}

            person_id = _safe_int(args.get("person_id"))
            if person_id is not None and person_id > 0:
                person = core_db.get(Person, int(person_id))
                if person is None:
                    return {"ok": False, "tool": name, "error": f"Person {person_id} not found."}
            else:
                assistant_person = (
                    core_db.query(Person)
                    .filter(Person.ppl_Name_First == "iSPEC")
                    .filter(Person.ppl_Name_Last == "Assistant")
                    .first()
                )
                if assistant_person is None:
                    assistant_person = Person(
                        ppl_AddedBy=user.username,
                        ppl_Name_First="iSPEC",
                        ppl_Name_Last="Assistant",
                    )
                    core_db.add(assistant_person)
                    core_db.flush()
                person_id = int(assistant_person.id)

            comment = ProjectComment(
                project_id=int(project_id),
                person_id=int(person_id),
                com_Comment=comment_text,
                com_CommentType=comment_type,
                com_AddedBy=user.username,
            )
            core_db.add(comment)
            core_db.flush()
            core_db.commit()

            snippet = comment_text.strip().replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:239] + "…"

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": int(project_id),
                    "comment_id": int(comment.id),
                    "person_id": int(person_id),
                    "comment_type": comment_type,
                    "added_by": user.username,
                    "snippet": snippet,
                    "links": {"project_ui": f"/project/{project_id}"},
                },
            }

        if name == "latest_activity":
            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=200)
            current_only = bool(args.get("current_only"))
            kinds = _coerce_str_list(args.get("kinds"), max_items=8)
            allowed_kinds = {
                "project",
                "project_comment",
                "experiment",
                "experiment_run",
            }
            if kinds is None:
                selected_kinds = allowed_kinds
            else:
                selected_kinds = {kind.strip().lower() for kind in kinds if kind.strip()}
                unknown = sorted(selected_kinds - allowed_kinds)
                if unknown:
                    return {
                        "ok": False,
                        "tool": name,
                        "error": f"Unknown kinds: {unknown}. Allowed: {sorted(allowed_kinds)}",
                    }

            events: list[dict[str, Any]] = []

            if "project" in selected_kinds:
                query = core_db.query(Project)
                if current_only:
                    query = query.filter(Project.prj_Current_FLAG.is_(True))
                rows = (
                    query.order_by(Project.prj_ModificationTS.desc(), Project.id.desc())
                    .limit(limit)
                    .all()
                )
                for project in rows:
                    modified = getattr(project, "prj_ModificationTS", None)
                    created = getattr(project, "prj_CreationTS", None)
                    ts = modified or created
                    events.append(
                        {
                            "kind": "project",
                            "timestamp": ts.isoformat() if ts else None,
                            "id": int(project.id),
                            "title": project.prj_ProjectTitle,
                            "status": project.prj_Status,
                            "current": bool(project.prj_Current_FLAG),
                            "links": {"ui": f"/project/{project.id}", "api": f"/api/projects/{project.id}"},
                        }
                    )

            if "project_comment" in selected_kinds:
                query = (
                    core_db.query(ProjectComment, Project)
                    .join(Project, ProjectComment.project_id == Project.id)
                    .order_by(ProjectComment.com_CreationTS.desc(), ProjectComment.id.desc())
                )
                if current_only:
                    query = query.filter(Project.prj_Current_FLAG.is_(True))
                rows = query.limit(limit).all()
                for comment, project in rows:
                    created = getattr(comment, "com_CreationTS", None)
                    text = (comment.com_Comment or "").strip()
                    if len(text) > 240:
                        text = text[:239] + "…"
                    events.append(
                        {
                            "kind": "project_comment",
                            "timestamp": created.isoformat() if created else None,
                            "id": int(comment.id),
                            "project_id": int(project.id),
                            "project_title": project.prj_ProjectTitle,
                            "type": comment.com_CommentType,
                            "added_by": comment.com_AddedBy,
                            "comment": text or None,
                            "links": {"ui": f"/project/{project.id}", "api": f"/api/projects/{project.id}"},
                        }
                    )

            if "experiment" in selected_kinds:
                query = core_db.query(Experiment)
                if current_only:
                    query = query.join(Project, Experiment.project_id == Project.id).filter(
                        Project.prj_Current_FLAG.is_(True)
                    )
                rows = (
                    query.order_by(Experiment.Experiment_ModificationTS.desc(), Experiment.id.desc())
                    .limit(limit)
                    .all()
                )
                for experiment in rows:
                    modified = getattr(experiment, "Experiment_ModificationTS", None)
                    created = getattr(experiment, "Experiment_CreationTS", None)
                    ts = modified or created
                    events.append(
                        {
                            "kind": "experiment",
                            "timestamp": ts.isoformat() if ts else None,
                            "id": int(experiment.id),
                            "record_no": experiment.record_no,
                            "name": experiment.exp_Name,
                            "type": str(experiment.exp_Type) if experiment.exp_Type is not None else None,
                            "project_id": int(experiment.project_id)
                            if experiment.project_id is not None
                            else None,
                            "links": {"ui": f"/experiment/{experiment.id}", "api": f"/api/experiments/{experiment.id}"},
                        }
                    )

            if "experiment_run" in selected_kinds:
                query = core_db.query(ExperimentRun)
                if current_only:
                    query = (
                        query.join(Experiment, ExperimentRun.experiment_id == Experiment.id)
                        .join(Project, Experiment.project_id == Project.id)
                        .filter(Project.prj_Current_FLAG.is_(True))
                    )
                rows = (
                    query.order_by(
                        ExperimentRun.ExperimentRun_ModificationTS.desc(),
                        ExperimentRun.id.desc(),
                    )
                    .limit(limit)
                    .all()
                )
                for run in rows:
                    modified = getattr(run, "ExperimentRun_ModificationTS", None)
                    created = getattr(run, "ExperimentRun_CreationTS", None)
                    ts = modified or created
                    events.append(
                        {
                            "kind": "experiment_run",
                            "timestamp": ts.isoformat() if ts else None,
                            "id": int(run.id),
                            "experiment_id": int(run.experiment_id),
                            "run_no": int(run.run_no),
                            "search_no": int(run.search_no),
                            "search_engine": getattr(run, "search_engine", None),
                            "search_state": getattr(run, "search_state", None),
                            "links": {
                                "ui": f"/experiment-run/{run.id}",
                                "api": f"/api/experiment-runs/{run.id}",
                            },
                        }
                    )

            def sort_key(item: dict[str, Any]) -> tuple[int, str, int]:
                ts = item.get("timestamp")
                if isinstance(ts, str) and ts:
                    return (1, ts, int(item.get("id") or 0))
                return (0, "", int(item.get("id") or 0))

            events.sort(key=sort_key, reverse=True)
            events = events[:limit]

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "limit": limit,
                    "current_only": current_only,
                    "kinds": sorted(selected_kinds),
                    "count": len(events),
                    "events": events,
                },
            }

        if name == "billing_category_counts":
            current_only = bool(args.get("current_only"))
            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=100)

            query = core_db.query(Project.prj_ProjectPriceLevel, func.count(Project.id))
            if current_only:
                query = query.filter(Project.prj_Current_FLAG.is_(True))
            rows = query.group_by(Project.prj_ProjectPriceLevel).all()
            items = [
                {
                    "category": (category.strip() if isinstance(category, str) and category.strip() else None),
                    "count": int(count or 0),
                }
                for category, count in rows
            ]
            items.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["category"] is None else str(item["category"]),
                )
            )
            if len(items) > limit:
                items = items[:limit]
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "current_only": current_only,
                    "field": "prj_ProjectPriceLevel",
                    "count": len(items),
                    "items": items,
                },
            }

        if name == "db_file_stats":
            core_path = _sqlite_path_from_session(core_db)
            assistant_path = (core_path.parent / "ispec-assistant.db") if core_path is not None else None
            schedule_path = _sqlite_path_from_session(schedule_db) if schedule_db is not None else None
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "core_db": _stat_path(core_path),
                    "assistant_db": _stat_path(assistant_path),
                    "schedule_db": _stat_path(schedule_path) if schedule_db is not None else None,
                },
            }

        if name == "count_all_projects":
            count = int(core_db.query(func.count(Project.id)).scalar() or 0)
            return {
                "ok": True,
                "tool": name,
                "result": {"count": count, "scope": "all"},
            }

        if name == "count_current_projects":
            count = int(
                core_db.query(func.count(Project.id))
                .filter(Project.prj_Current_FLAG.is_(True))
                .scalar()
                or 0
            )
            return {
                "ok": True,
                "tool": name,
                "result": {"count": count, "scope": "current"},
            }

        if name == "project_status_counts":
            current_only = bool(args.get("current_only"))
            query = core_db.query(Project.prj_Status, func.count(Project.id))
            if current_only:
                query = query.filter(Project.prj_Current_FLAG.is_(True))
            rows = query.group_by(Project.prj_Status).all()
            items = [
                {"status": (status if status is not None else None), "count": int(count or 0)}
                for status, count in rows
            ]
            items.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["status"] is None else str(item["status"]),
                )
            )
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "current_only": current_only,
                    "total": int(sum(item["count"] for item in items)),
                    "items": items,
                },
            }

        if name == "latest_projects":
            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            current_only = bool(args.get("current_only"))
            sort_raw = (_safe_str(args.get("sort"), max_len=32) or "modified").strip().lower()

            query = core_db.query(Project)
            if current_only:
                query = query.filter(Project.prj_Current_FLAG.is_(True))

            if sort_raw in {"created", "creation"}:
                query = query.order_by(Project.prj_CreationTS.desc(), Project.id.desc())
                sort = "created"
            elif sort_raw in {"modified", "modification", "updated", "update"}:
                query = query.order_by(Project.prj_ModificationTS.desc(), Project.id.desc())
                sort = "modified"
            elif sort_raw in {"id"}:
                query = query.order_by(Project.id.desc())
                sort = "id"
            else:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "sort must be one of: created, modified, id",
                }

            rows = query.limit(limit).all()
            projects = []
            for project in rows:
                projects.append(
                    {
                        "id": int(project.id),
                        "title": project.prj_ProjectTitle,
                        "status": project.prj_Status,
                        "current": bool(project.prj_Current_FLAG),
                        "created": project.prj_CreationTS.isoformat()
                        if getattr(project, "prj_CreationTS", None)
                        else None,
                        "modified": project.prj_ModificationTS.isoformat()
                        if getattr(project, "prj_ModificationTS", None)
                        else None,
                        "links": {
                            "ui": f"/project/{project.id}",
                            "api": f"/api/projects/{project.id}",
                        },
                    }
                )
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "sort": sort,
                    "current_only": current_only,
                    "count": len(projects),
                    "projects": projects,
                },
            }

        if name == "latest_project_comments":
            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            project_id = _safe_int(args.get("project_id"))
            query = (
                core_db.query(ProjectComment, Project)
                .join(Project, ProjectComment.project_id == Project.id)
                .order_by(ProjectComment.com_CreationTS.desc(), ProjectComment.id.desc())
            )
            if project_id is not None:
                query = query.filter(ProjectComment.project_id == project_id)
            rows = query.limit(limit).all()

            comments: list[dict[str, Any]] = []
            for comment, project in rows:
                created = getattr(comment, "com_CreationTS", None)
                text = (comment.com_Comment or "").strip()
                if len(text) > 240:
                    text = text[:239] + "…"
                comments.append(
                    {
                        "id": int(comment.id),
                        "project_id": int(project.id),
                        "project_title": project.prj_ProjectTitle,
                        "type": comment.com_CommentType,
                        "added_by": comment.com_AddedBy,
                        "created": created.isoformat() if created else None,
                        "comment": text or None,
                        "links": {
                            "ui": f"/project/{project.id}",
                            "api": f"/api/projects/{project.id}",
                        },
                    }
                )
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": project_id,
                    "count": len(comments),
                    "comments": comments,
                },
            }

        if name == "search_api":
            query = _safe_str(args.get("query"), max_len=200)
            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            if not query:
                return {"ok": False, "tool": name, "error": "Missing string argument: query"}
            if not isinstance(api_schema, dict):
                return {"ok": False, "tool": name, "error": "API schema is not available."}

            matches = _search_openapi_schema(api_schema, query=query, limit=limit)
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "query": query,
                    "count": len(matches),
                    "matches": matches,
                    "docs_hint": "/docs (FastAPI Swagger UI), /openapi.json",
                },
            }

        if name == "experiments_for_project":
            project_id = _safe_int(args.get("project_id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: project_id"}
            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=100)

            rows = (
                core_db.query(Experiment)
                .filter(Experiment.project_id == project_id)
                .order_by(Experiment.id.desc())
                .limit(limit)
                .all()
            )

            experiments: list[dict[str, Any]] = []
            for experiment in rows:
                created = getattr(experiment, "Experiment_CreationTS", None)
                experiments.append(
                    {
                        "id": int(experiment.id),
                        "record_no": experiment.record_no,
                        "name": experiment.exp_Name,
                        "type": str(experiment.exp_Type) if experiment.exp_Type is not None else None,
                        "date": experiment.exp_Date.isoformat() if experiment.exp_Date else None,
                        "project_id": int(experiment.project_id)
                        if experiment.project_id is not None
                        else None,
                        "created": created.isoformat() if created else None,
                        "links": {
                            "ui": f"/experiment/{experiment.id}",
                            "api": f"/api/experiments/{experiment.id}",
                        },
                    }
                )
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": project_id,
                    "count": len(experiments),
                    "experiments": experiments,
                },
            }

        if name == "latest_experiments":
            limit = _clamp_int(_safe_int(args.get("limit")), default=5, min_value=1, max_value=50)
            rows = core_db.query(Experiment).order_by(Experiment.id.desc()).limit(limit).all()

            experiments: list[dict[str, Any]] = []
            for experiment in rows:
                created = getattr(experiment, "Experiment_CreationTS", None)
                experiments.append(
                    {
                        "id": int(experiment.id),
                        "record_no": experiment.record_no,
                        "name": experiment.exp_Name,
                        "type": str(experiment.exp_Type) if experiment.exp_Type is not None else None,
                        "date": experiment.exp_Date.isoformat() if experiment.exp_Date else None,
                        "project_id": int(experiment.project_id)
                        if experiment.project_id is not None
                        else None,
                        "created": created.isoformat() if created else None,
                        "links": {
                            "ui": f"/experiment/{experiment.id}",
                            "api": f"/api/experiments/{experiment.id}",
                        },
                    }
                )
            return {"ok": True, "tool": name, "result": {"experiments": experiments}}

        if name == "get_experiment":
            experiment_id = _safe_int(args.get("id"))
            if experiment_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            experiment = core_db.get(Experiment, experiment_id)
            if experiment is None:
                return {
                    "ok": False,
                    "tool": name,
                    "error": f"Experiment {experiment_id} not found.",
                }
            created = getattr(experiment, "Experiment_CreationTS", None)
            modified = getattr(experiment, "Experiment_ModificationTS", None)
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "id": int(experiment.id),
                    "record_no": experiment.record_no,
                    "name": experiment.exp_Name,
                    "type": str(experiment.exp_Type) if experiment.exp_Type is not None else None,
                    "date": experiment.exp_Date.isoformat() if experiment.exp_Date else None,
                    "project_id": int(experiment.project_id)
                    if experiment.project_id is not None
                    else None,
                    "created": created.isoformat() if created else None,
                    "modified": modified.isoformat() if modified else None,
                    "links": {
                        "ui": f"/experiment/{experiment.id}",
                        "api": f"/api/experiments/{experiment.id}",
                    },
                },
            }

        if name == "latest_experiment_runs":
            limit = _clamp_int(_safe_int(args.get("limit")), default=5, min_value=1, max_value=50)
            rows = core_db.query(ExperimentRun).order_by(ExperimentRun.id.desc()).limit(limit).all()

            runs: list[dict[str, Any]] = []
            for run in rows:
                created = getattr(run, "ExperimentRun_CreationTS", None)
                runs.append(
                    {
                        "id": int(run.id),
                        "experiment_id": int(run.experiment_id),
                        "run_no": int(run.run_no),
                        "search_no": int(run.search_no),
                        "search_engine": getattr(run, "search_engine", None),
                        "search_state": getattr(run, "search_state", None),
                        "created": created.isoformat() if created else None,
                        "links": {
                            "ui": f"/experiment-run/{run.id}",
                            "api": f"/api/experiment-runs/{run.id}",
                        },
                    }
                )
            return {"ok": True, "tool": name, "result": {"runs": runs}}

        if name == "get_experiment_run":
            run_id = _safe_int(args.get("id"))
            if run_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            run = core_db.get(ExperimentRun, run_id)
            if run is None:
                return {"ok": False, "tool": name, "error": f"ExperimentRun {run_id} not found."}
            created = getattr(run, "ExperimentRun_CreationTS", None)
            modified = getattr(run, "ExperimentRun_ModificationTS", None)
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "id": int(run.id),
                    "experiment_id": int(run.experiment_id),
                    "run_no": int(run.run_no),
                    "search_no": int(run.search_no),
                    "search_engine": getattr(run, "search_engine", None),
                    "search_state": getattr(run, "search_state", None),
                    "created": created.isoformat() if created else None,
                    "modified": modified.isoformat() if modified else None,
                    "links": {
                        "ui": f"/experiment-run/{run.id}",
                        "api": f"/api/experiment-runs/{run.id}",
                    },
                },
            }

        if name == "e2g_search_genes_in_project":
            project_id = _safe_int(args.get("project_id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: project_id"}
            if omics_db is None:
                return {"ok": False, "tool": name, "error": "Omics database session is not available."}

            query_text = _safe_str(args.get("query"), max_len=200)
            if not query_text:
                return {"ok": False, "tool": name, "error": "Missing string argument: query"}
            if query_text.strip().lower() in {"*", "all"}:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "Use e2g_gene_in_project for a specific GeneID, or provide a keyword (GeneSymbol/description).",
                }

            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            pattern = f"%{query_text}%"

            run_ids = [
                int(row[0])
                for row in core_db.query(ExperimentRun.id)
                .join(Experiment, ExperimentRun.experiment_id == Experiment.id)
                .filter(Experiment.project_id == project_id)
                .all()
            ]
            if not run_ids:
                return {
                    "ok": True,
                    "tool": name,
                    "result": {
                        "project_id": project_id,
                        "query": query_text,
                        "count": 0,
                        "matches": [],
                    },
                }

            rows = (
                omics_db.query(
                    E2G.gene,
                    E2G.gene_symbol,
                    E2G.description,
                    func.count(E2G.id).label("hits"),
                )
                .filter(E2G.experiment_run_id.in_(run_ids))
                .filter(
                    or_(
                        E2G.gene.ilike(pattern),
                        E2G.gene_symbol.ilike(pattern),
                        E2G.description.ilike(pattern),
                    )
                )
                .group_by(E2G.gene, E2G.gene_symbol, E2G.description)
                .order_by(func.count(E2G.id).desc(), E2G.gene.asc())
                .limit(limit)
                .all()
            )

            matches: list[dict[str, Any]] = []
            for gene, symbol, description, hits in rows:
                desc_text = (description or "").strip()
                if len(desc_text) > 240:
                    desc_text = desc_text[:239] + "…"
                gene_id = _safe_int(gene)
                matches.append(
                    {
                        "gene_id": gene_id,
                        "gene_symbol": (symbol.strip() if isinstance(symbol, str) and symbol.strip() else None),
                        "description": desc_text or None,
                        "hits": int(hits or 0),
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": project_id,
                    "query": query_text,
                    "count": len(matches),
                    "matches": matches,
                },
            }

        if name == "e2g_gene_in_project":
            project_id = _safe_int(args.get("project_id"))
            gene_id = _safe_int(args.get("gene_id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: project_id"}
            if gene_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: gene_id"}
            if omics_db is None:
                return {"ok": False, "tool": name, "error": "Omics database session is not available."}
            limit = _clamp_int(_safe_int(args.get("limit")), default=50, min_value=1, max_value=200)

            run_rows = (
                core_db.query(ExperimentRun, Experiment)
                .join(Experiment, ExperimentRun.experiment_id == Experiment.id)
                .filter(Experiment.project_id == project_id)
                .order_by(Experiment.id.desc(), ExperimentRun.id.desc())
                .all()
            )
            run_lookup: dict[int, tuple[Any, Any]] = {
                int(run.id): (run, experiment) for run, experiment in run_rows
            }
            run_ids = list(run_lookup.keys())
            if not run_ids:
                return {
                    "ok": True,
                    "tool": name,
                    "result": {
                        "project_id": project_id,
                        "gene_id": gene_id,
                        "count": 0,
                        "hits": [],
                    },
                }

            rows = (
                omics_db.query(E2G)
                .filter(E2G.experiment_run_id.in_(run_ids))
                .filter(E2G.geneidtype == "GeneID")
                .filter(E2G.gene == str(gene_id))
                .order_by(
                    func.coalesce(E2G.psms_u2g, E2G.psms, 0).desc(),
                    func.coalesce(E2G.iBAQ_dstrAdj, 0.0).desc(),
                    E2G.experiment_run_id.desc(),
                    E2G.id.asc(),
                )
                .limit(limit)
                .all()
            )

            hits: list[dict[str, Any]] = []
            for e2g in rows:
                run_id = int(getattr(e2g, "experiment_run_id"))
                run_obj, experiment_obj = run_lookup.get(run_id, (None, None))
                if run_obj is None or experiment_obj is None:
                    continue
                peptideprint = getattr(e2g, "peptideprint", None)
                peptideprint_len = len(peptideprint) if isinstance(peptideprint, str) else None
                peptideprint_preview = None
                if isinstance(peptideprint, str) and peptideprint:
                    peptideprint_preview = peptideprint[:800]
                    if len(peptideprint) > 800:
                        peptideprint_preview = peptideprint_preview[:799] + "…"

                hits.append(
                    {
                        "experiment_id": int(experiment_obj.id),
                        "experiment_record_no": experiment_obj.record_no,
                        "experiment_name": experiment_obj.exp_Name,
                        "experiment_run_id": int(run_obj.id),
                        "run_no": int(run_obj.run_no),
                        "search_no": int(run_obj.search_no),
                        "label": run_obj.label,
                        "gene_id": gene_id,
                        "gene_symbol": getattr(e2g, "gene_symbol", None),
                        "taxon_id": getattr(e2g, "taxon_id", None),
                        "sra": getattr(e2g, "sra", None),
                        "psms": getattr(e2g, "psms", None),
                        "psms_u2g": getattr(e2g, "psms_u2g", None),
                        "peptide_count": getattr(e2g, "peptide_count", None),
                        "peptide_count_u2g": getattr(e2g, "peptide_count_u2g", None),
                        "coverage": getattr(e2g, "coverage", None),
                        "coverage_u2g": getattr(e2g, "coverage_u2g", None),
                        "area_sum_u2g_all": getattr(e2g, "area_sum_u2g_all", None),
                        "iBAQ_dstrAdj": getattr(e2g, "iBAQ_dstrAdj", None),
                        "peptideprint_len": peptideprint_len,
                        "peptideprint_preview": peptideprint_preview,
                        "links": {
                            "project_ui": f"/project/{project_id}",
                            "experiment_ui": f"/experiment/{experiment_obj.id}",
                            "experiment_run_ui": f"/experiment-run/{run_obj.id}",
                        },
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": project_id,
                    "gene_id": gene_id,
                    "count": len(hits),
                    "hits": hits,
                },
            }

        if name == "get_project":
            project_id = _safe_int(args.get("id"))
            if project_id is None:
                project_id = _safe_int(args.get("project_id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id (or project_id)"}
            project = core_db.get(Project, project_id)
            if project is None:
                return {"ok": False, "tool": name, "error": f"Project {project_id} not found."}
            return {"ok": True, "tool": name, "result": project_summary(core_db, project)}

        if name == "search_projects":
            query = _safe_str(args.get("query"), max_len=200)
            limit = _clamp_int(_safe_int(args.get("limit")), default=5, min_value=1, max_value=20)
            if not query:
                return {"ok": False, "tool": name, "error": "Missing string argument: query"}
            if query.strip().lower() in {"*", "all"}:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "Use count_all_projects or count_current_projects to answer 'how many projects'. search_projects expects a keyword (title/PI/contact) or an id.",
                }

            matches: list[dict[str, Any]] = []
            project_id = _safe_int(query)
            if project_id is not None:
                project = core_db.get(Project, project_id)
                if project is not None:
                    matches.append(
                        {
                            "id": int(project.id),
                            "title": project.prj_ProjectTitle,
                            "status": project.prj_Status,
                            "current": bool(getattr(project, "prj_Current_FLAG", False)),
                        }
                    )
                return {"ok": True, "tool": name, "result": {"matches": matches}}

            pattern = f"%{query}%"
            rows = (
                core_db.query(Project)
                .filter(
                    or_(
                        Project.prj_ProjectTitle.ilike(pattern),
                        Project.prj_PI.ilike(pattern),
                        Project.prj_Project_LabContact.ilike(pattern),
                    )
                )
                .order_by(Project.id.asc())
                .limit(limit)
                .all()
            )
            for project in rows:
                matches.append(
                    {
                        "id": int(project.id),
                        "title": project.prj_ProjectTitle,
                        "status": project.prj_Status,
                        "current": bool(getattr(project, "prj_Current_FLAG", False)),
                    }
                )
            return {"ok": True, "tool": name, "result": {"matches": matches}}

        if name == "get_person":
            person_id = _safe_int(args.get("id"))
            if person_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            person = core_db.get(Person, person_id)
            if person is None:
                return {"ok": False, "tool": name, "error": f"Person {person_id} not found."}
            return {"ok": True, "tool": name, "result": person_summary(person)}

        if name == "search_people":
            query = _safe_str(args.get("query"), max_len=200)
            limit = _clamp_int(_safe_int(args.get("limit")), default=5, min_value=1, max_value=20)
            if not query:
                return {"ok": False, "tool": name, "error": "Missing string argument: query"}

            matches: list[dict[str, Any]] = []
            person_id = _safe_int(query)
            if person_id is not None:
                person = core_db.get(Person, person_id)
                if person is not None:
                    matches.append(person_summary(person))
                return {"ok": True, "tool": name, "result": {"matches": matches}}

            pattern = f"%{query}%"
            rows = (
                core_db.query(Person)
                .filter(
                    or_(
                        Person.ppl_Name_First.ilike(pattern),
                        Person.ppl_Name_Last.ilike(pattern),
                        Person.ppl_Email.ilike(pattern),
                    )
                )
                .order_by(Person.id.asc())
                .limit(limit)
                .all()
            )
            matches.extend(person_summary(person) for person in rows)
            return {"ok": True, "tool": name, "result": {"matches": matches}}

        if name == "list_schedule_slots":
            if schedule_db is None:
                return {"ok": False, "tool": name, "error": "Schedule DB is not available."}

            start = _safe_date(args.get("start"))
            end = _safe_date(args.get("end"))
            if start is None or end is None:
                today = datetime.now(CENTRAL_TZ).date()
                start = today
                end = today + timedelta(days=28)
            if end < start:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "end must be on or after start (YYYY-MM-DD).",
                }

            raw_status = _safe_str(args.get("status"), max_len=32)
            status: str | None = None
            if raw_status:
                normalized = raw_status.strip().lower()
                if normalized not in SCHEDULE_SLOT_STATUSES:
                    return {
                        "ok": False,
                        "tool": name,
                        "error": f"status must be one of {sorted(SCHEDULE_SLOT_STATUSES)}",
                    }
                status = normalized

            limit = _clamp_int(_safe_int(args.get("limit")), default=50, min_value=1, max_value=500)

            start_utc, end_utc = _range_bounds_local(start, end)
            query = (
                schedule_db.query(ScheduleSlot)
                .filter(ScheduleSlot.start_at >= start_utc)
                .filter(ScheduleSlot.start_at <= end_utc)
            )
            if status is not None:
                query = query.filter(ScheduleSlot.status == status)
            rows = query.order_by(ScheduleSlot.start_at.asc()).limit(limit).all()

            items: list[dict[str, Any]] = []
            for row in rows:
                start_utc_aware = _as_utc_aware(row.start_at)
                end_utc_aware = _as_utc_aware(row.end_at)
                items.append(
                    {
                        "id": int(row.id),
                        "status": row.status,
                        "start_at": start_utc_aware.isoformat(),
                        "end_at": end_utc_aware.isoformat(),
                        "start_at_central": start_utc_aware.astimezone(CENTRAL_TZ).isoformat(),
                        "end_at_central": end_utc_aware.astimezone(CENTRAL_TZ).isoformat(),
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "timezone": str(CENTRAL_TZ),
                    "count": len(items),
                    "items": items,
                },
            }

        if name == "list_schedule_requests":
            auth_error = _require_admin(user)
            if auth_error:
                return {"ok": False, "tool": name, "error": auth_error}
            if schedule_db is None:
                return {"ok": False, "tool": name, "error": "Schedule DB is not available."}

            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=200)
            raw_status = _safe_str(args.get("status"), max_len=32)
            status: str | None = None
            if raw_status:
                normalized = raw_status.strip().lower()
                if normalized not in SCHEDULE_REQUEST_STATUSES:
                    return {
                        "ok": False,
                        "tool": name,
                        "error": f"status must be one of {sorted(SCHEDULE_REQUEST_STATUSES)}",
                    }
                status = normalized

            query = schedule_db.query(ScheduleRequest)
            if status is not None:
                query = query.filter(ScheduleRequest.status == status)
            rows = query.order_by(ScheduleRequest.created_at.desc()).limit(limit).all()

            items: list[dict[str, Any]] = []
            for row in rows:
                slot_ids = [link.slot_id for link in sorted(row.slots, key=lambda link: link.rank)]
                items.append(
                    {
                        "id": int(row.id),
                        "status": row.status,
                        "created_at": _as_utc_aware(row.created_at).isoformat(),
                        "requester_name": row.requester_name,
                        "requester_email": row.requester_email,
                        "requester_org": row.requester_org,
                        "project_title": row.project_title,
                        "cancer_related": bool(row.cancer_related),
                        "slot_ids": slot_ids,
                    }
                )

            return {"ok": True, "tool": name, "result": {"count": len(items), "items": items}}

        if name == "get_schedule_request":
            auth_error = _require_admin(user)
            if auth_error:
                return {"ok": False, "tool": name, "error": auth_error}
            if schedule_db is None:
                return {"ok": False, "tool": name, "error": "Schedule DB is not available."}

            request_id = _safe_int(args.get("id"))
            if request_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            row = schedule_db.get(ScheduleRequest, request_id)
            if row is None:
                return {"ok": False, "tool": name, "error": f"Request {request_id} not found."}

            slot_ids = [link.slot_id for link in sorted(row.slots, key=lambda link: link.rank)]
            description = row.project_description or ""
            if len(description) > 2000:
                description = description[:1999] + "…"

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "id": int(row.id),
                    "status": row.status,
                    "created_at": _as_utc_aware(row.created_at).isoformat(),
                    "updated_at": _as_utc_aware(row.updated_at).isoformat(),
                    "requester_name": row.requester_name,
                    "requester_email": row.requester_email,
                    "requester_org": row.requester_org,
                    "requester_phone": row.requester_phone,
                    "project_title": row.project_title,
                    "project_description": description,
                    "cancer_related": bool(row.cancer_related),
                    "slot_ids": slot_ids,
                },
            }

        return {
            "ok": False,
            "tool": name,
            "error": f"Unknown tool '{name}'.",
            "available": [
                "project_counts_snapshot",
                "latest_activity",
                "billing_category_counts",
                "db_file_stats",
                "count_all_projects",
                "count_current_projects",
                "project_status_counts",
                "latest_projects",
                "latest_project_comments",
                "search_projects",
                "get_project",
                "search_api",
                "repo_list_files",
                "repo_search",
                "repo_read_file",
                "experiments_for_project",
                "latest_experiments",
                "get_experiment",
                "latest_experiment_runs",
                "get_experiment_run",
                "e2g_search_genes_in_project",
                "e2g_gene_in_project",
                "search_people",
                "get_person",
                "list_schedule_slots",
                "list_schedule_requests",
                "get_schedule_request",
            ],
        }
    except Exception as exc:
        return {
            "ok": False,
            "tool": name,
            "error": f"{type(exc).__name__}: {exc}",
        }


_OPENAI_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "project_counts_snapshot": {
        "type": "function",
        "function": {
            "name": "project_counts_snapshot",
            "description": "Return a snapshot of project counts (total/current/status/billing categories).",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_categories": {
                        "type": "integer",
                        "description": "Max billing categories to include.",
                    }
                },
            },
        },
    },
    "latest_activity": {
        "type": "function",
        "function": {
            "name": "latest_activity",
            "description": "Return recent iSPEC activity across projects, comments, experiments, and runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max events to return."},
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subset of kinds: project, project_comment, experiment, experiment_run.",
                    },
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, restrict activity to current projects.",
                    },
                },
            },
        },
    },
    "billing_category_counts": {
        "type": "function",
        "function": {
            "name": "billing_category_counts",
            "description": "Count projects grouped by billing category (ProjectPriceLevel).",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, only current projects.",
                    },
                    "limit": {"type": "integer", "description": "Max categories to return."},
                },
            },
        },
    },
    "db_file_stats": {
        "type": "function",
        "function": {
            "name": "db_file_stats",
            "description": "Return sqlite DB file sizes (core, assistant, and schedule DB).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "count_all_projects": {
        "type": "function",
        "function": {
            "name": "count_all_projects",
            "description": "Count total projects across all statuses/flags.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "count_current_projects": {
        "type": "function",
        "function": {
            "name": "count_current_projects",
            "description": "Count current projects only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "project_status_counts": {
        "type": "function",
        "function": {
            "name": "project_status_counts",
            "description": "Count projects by status (optionally restricted to current projects).",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, include only current projects.",
                    }
                },
            },
        },
    },
    "latest_projects": {
        "type": "function",
        "function": {
            "name": "latest_projects",
            "description": "Fetch the latest projects by modified/created timestamp (or id).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sort": {"type": "string", "description": "created, modified, or id."},
                    "limit": {"type": "integer", "description": "Max projects to return."},
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, only current projects.",
                    },
                },
            },
        },
    },
    "latest_project_comments": {
        "type": "function",
        "function": {
            "name": "latest_project_comments",
            "description": "Fetch latest project comments, optionally for a single project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max comments to return."},
                    "project_id": {"type": "integer", "description": "Optional project id filter."},
                },
            },
        },
    },
    "search_projects": {
        "type": "function",
        "function": {
            "name": "search_projects",
            "description": "Search projects by keyword (title/PI/contact) or id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword or id."},
                    "limit": {"type": "integer", "description": "Max matches to return."},
                },
                "required": ["query"],
            },
        },
    },
    "projects": {
        "type": "function",
        "function": {
            "name": "projects",
            "description": "Alias for get_project: fetch a project by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id."},
                    "id": {"type": "integer", "description": "Alias for project_id."},
                },
                "required": ["project_id"],
            },
        },
    },
    "get_project": {
        "type": "function",
        "function": {
            "name": "get_project",
            "description": "Fetch a project by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Project id."}},
                "required": ["id"],
            },
        },
    },
    "create_project_comment": {
        "type": "function",
        "function": {
            "name": "create_project_comment",
            "description": "Create a new project comment in project history (write). Use only when the user explicitly asks to save/log a note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id to attach the comment to."},
                    "comment": {"type": "string", "description": "Comment text to store."},
                    "comment_type": {
                        "type": "string",
                        "description": "Optional comment type label (e.g. meeting_note, assistant_note).",
                    },
                    "person_id": {
                        "type": "integer",
                        "description": "Optional person id for the FK; defaults to an 'iSPEC Assistant' person record.",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true; only call when user explicitly requests saving to project history.",
                    },
                },
                "required": ["project_id", "comment", "confirm"],
            },
        },
    },
    "search_api": {
        "type": "function",
        "function": {
            "name": "search_api",
            "description": "Search FastAPI/OpenAPI endpoints by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword query."},
                    "limit": {"type": "integer", "description": "Max matches to return."},
                },
                "required": ["query"],
            },
        },
    },
    "repo_list_files": {
        "type": "function",
        "function": {
            "name": "repo_list_files",
            "description": "Dev-only: list repo files (relative paths) to help locate code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional substring filter on the file path."},
                    "path": {
                        "type": "string",
                        "description": "Optional repo-relative directory (defaults to iSPEC/src).",
                    },
                    "limit": {"type": "integer", "description": "Max files to return."},
                },
            },
        },
    },
    "repo_search": {
        "type": "function",
        "function": {
            "name": "repo_search",
            "description": "Dev-only: grep the repo for a string (or regex) and return line matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text (or regex if regex=true)."},
                    "path": {
                        "type": "string",
                        "description": "Optional repo-relative directory (defaults to iSPEC/src).",
                    },
                    "limit": {"type": "integer", "description": "Max matches to return."},
                    "regex": {"type": "boolean", "description": "If true, treat query as regex."},
                    "ignore_case": {"type": "boolean", "description": "If true, case-insensitive search."},
                },
                "required": ["query"],
            },
        },
    },
    "repo_read_file": {
        "type": "function",
        "function": {
            "name": "repo_read_file",
            "description": "Dev-only: read a snippet from a repo file (use repo-relative paths).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative file path."},
                    "start_line": {"type": "integer", "description": "1-based start line."},
                    "max_lines": {"type": "integer", "description": "Max lines to return."},
                },
                "required": ["path"],
            },
        },
    },
    "experiments_for_project": {
        "type": "function",
        "function": {
            "name": "experiments_for_project",
            "description": "List experiments for a given project id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id."},
                    "limit": {"type": "integer", "description": "Max experiments to return."},
                },
                "required": ["project_id"],
            },
        },
    },
    "latest_experiments": {
        "type": "function",
        "function": {
            "name": "latest_experiments",
            "description": "Fetch latest experiments.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max experiments to return."}},
            },
        },
    },
    "get_experiment": {
        "type": "function",
        "function": {
            "name": "get_experiment",
            "description": "Fetch an experiment by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Experiment id."}},
                "required": ["id"],
            },
        },
    },
    "latest_experiment_runs": {
        "type": "function",
        "function": {
            "name": "latest_experiment_runs",
            "description": "Fetch latest experiment runs.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max runs to return."}},
            },
        },
    },
    "get_experiment_run": {
        "type": "function",
        "function": {
            "name": "get_experiment_run",
            "description": "Fetch an experiment run by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "ExperimentRun id."}},
                "required": ["id"],
            },
        },
    },
    "e2g_search_genes_in_project": {
        "type": "function",
        "function": {
            "name": "e2g_search_genes_in_project",
            "description": "Search E2G (experiment_to_gene) rows within a project by GeneID, GeneSymbol, or description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id."},
                    "query": {"type": "string", "description": "GeneID, gene symbol, or keyword."},
                    "limit": {"type": "integer", "description": "Max genes to return."},
                },
                "required": ["project_id", "query"],
            },
        },
    },
    "e2g_gene_in_project": {
        "type": "function",
        "function": {
            "name": "e2g_gene_in_project",
            "description": "Return E2G hits (per experiment run) for a GeneID within a project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id."},
                    "gene_id": {"type": "integer", "description": "GeneID (canonical identifier)."},
                    "limit": {"type": "integer", "description": "Max hits to return."},
                },
                "required": ["project_id", "gene_id"],
            },
        },
    },
    "search_people": {
        "type": "function",
        "function": {
            "name": "search_people",
            "description": "Search people by keyword (name/email) or id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword or id."},
                    "limit": {"type": "integer", "description": "Max matches to return."},
                },
                "required": ["query"],
            },
        },
    },
    "get_person": {
        "type": "function",
        "function": {
            "name": "get_person",
            "description": "Fetch a person by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Person id."}},
                "required": ["id"],
            },
        },
    },
    "list_schedule_slots": {
        "type": "function",
        "function": {
            "name": "list_schedule_slots",
            "description": "List schedule slots over a date range (America/Chicago).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date YYYY-MM-DD."},
                    "end": {"type": "string", "description": "End date YYYY-MM-DD."},
                    "status": {"type": "string", "description": "available, booked, or closed."},
                    "limit": {"type": "integer", "description": "Max slots to return."},
                },
            },
        },
    },
    "list_schedule_requests": {
        "type": "function",
        "function": {
            "name": "list_schedule_requests",
            "description": "Admin-only: list schedule requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max requests to return."},
                    "status": {
                        "type": "string",
                        "description": "requested, confirmed, declined, cancelled.",
                    },
                },
            },
        },
    },
    "get_schedule_request": {
        "type": "function",
        "function": {
            "name": "get_schedule_request",
            "description": "Admin-only: fetch a schedule request by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Request id."}},
                "required": ["id"],
            },
        },
    },
}


def openai_tools_for_user(user: AuthUser | None) -> list[dict[str, Any]]:
    """Return OpenAI-compatible tool schemas, filtered by tool scope."""

    tools: list[dict[str, Any]] = []
    repo_enabled = _repo_tools_enabled()
    for name, spec in _OPENAI_TOOL_SPECS.items():
        if name.startswith("repo_") and not repo_enabled:
            continue
        if name in _WRITE_TOOL_NAMES:
            if user is None:
                continue
            if user.role in {UserRole.viewer, UserRole.client}:
                continue
        scope = _TOOL_SCOPES.get(name, ToolScope.staff)
        if _scope_error(scope, user) is None:
            tools.append(spec)
    return tools


_OPENAPI_METHODS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "options",
    "head",
}


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _search_openapi_schema(api_schema: dict[str, Any], *, query: str, limit: int) -> list[dict[str, Any]]:
    """Search an OpenAPI schema for matching endpoints.

    Returns a list of {method, path, summary, tags, operation_id}.
    """

    tokens = _tokenize(query)
    paths = api_schema.get("paths")
    if not isinstance(paths, dict):
        return []

    scored: list[tuple[int, str, str, dict[str, Any]]] = []
    for path, methods in paths.items():
        if not isinstance(path, str) or not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if not isinstance(method, str) or method.lower() not in _OPENAPI_METHODS:
                continue
            if not isinstance(op, dict):
                continue

            summary = (op.get("summary") or "").strip()
            description = (op.get("description") or "").strip()
            operation_id = (op.get("operationId") or "").strip()
            tags = op.get("tags") if isinstance(op.get("tags"), list) else []
            tag_text = " ".join(str(t) for t in tags if t)

            haystack = " ".join(
                part
                for part in [
                    method.lower(),
                    path.lower(),
                    summary.lower(),
                    description.lower(),
                    operation_id.lower(),
                    tag_text.lower(),
                ]
                if part
            )
            score = 0
            for token in tokens:
                if token in haystack:
                    score += 1
                    if token in path.lower():
                        score += 1
                    if summary and token in summary.lower():
                        score += 1
            if score <= 0:
                continue
            scored.append((score, path, method.upper(), op))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    results: list[dict[str, Any]] = []
    for score, path, method, op in scored[: max(0, limit)]:
        summary = (op.get("summary") or "").strip() or None
        operation_id = (op.get("operationId") or "").strip() or None
        tags = op.get("tags") if isinstance(op.get("tags"), list) else []
        tags_clean = [str(tag) for tag in tags if isinstance(tag, str) and tag.strip()]
        results.append(
            {
                "method": method,
                "path": path,
                "summary": summary,
                "operation_id": operation_id,
                "tags": tags_clean,
                "score": score,
            }
        )
    return results
