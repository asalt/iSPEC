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

from sqlalchemy import Text, cast, func, or_
from sqlalchemy.orm import Session, defer

from ispec.assistant.context import person_summary, project_summary
from ispec.assistant.models import (
    SupportMemory,
    SupportMemoryEvidence,
    SupportMessage as AssistantSupportMessage,
    SupportSession,
    SupportSessionReview,
)
from ispec.assistant.prompt_header import build_prompt_header, header_legend
from ispec.agent.models import AgentCommand, AgentEvent, AgentRun, AgentStep
from ispec.db.models import (
    AuthUser,
    AuthUserProject,
    E2G,
    Experiment,
    ExperimentRun,
    Person,
    Project,
    ProjectComment,
    ProjectFile,
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
    "assistant_stats": ToolScope.staff,
    "assistant_recent_sessions": ToolScope.staff,
    "assistant_get_session_review": ToolScope.staff,
    "assistant_prompt_header": ToolScope.staff,
    "assistant_search_messages": ToolScope.admin,
    "assistant_get_message_context": ToolScope.admin,
    "assistant_list_digests": ToolScope.admin,
    "assistant_get_digest": ToolScope.admin,
    "assistant_search_digests": ToolScope.admin,
    "assistant_search_internal_logs": ToolScope.admin,
    "assistant_recent_agent_commands": ToolScope.admin,
    "assistant_recent_agent_steps": ToolScope.admin,
    "assistant_recent_session_reviews": ToolScope.admin,
    "assistant_get_agent_step": ToolScope.admin,
    "assistant_get_agent_command": ToolScope.admin,
    "assistant_get_agent_run": ToolScope.admin,
    "assistant_list_users": ToolScope.admin,
    "count_all_projects": ToolScope.staff,
    "count_current_projects": ToolScope.staff,
    "project_status_counts": ToolScope.staff,
    "latest_projects": ToolScope.staff,
    "latest_project_comments": ToolScope.staff,
    "search_projects": ToolScope.staff,
    "projects": ToolScope.staff,
    "get_project": ToolScope.staff,
    "my_projects": ToolScope.user,
    "project_files_for_project": ToolScope.user,
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
    "create_project_comment": ToolScope.user,
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


def tool_prompt(*, tool_names: set[str] | None = None) -> str:
    """Short tool list for the system prompt.

    When ``tool_names`` is provided, only those tools are listed (in the normal
    order). This is useful for multi-stage tool routing to keep prompts small.
    """

    allowed: set[str] | None = None
    if tool_names:
        allowed = {name.strip() for name in tool_names if isinstance(name, str) and name.strip()}

    lines = [
        "Available tools:",
        "- (Most tools are read-only; create_project_comment writes project history.)",
    ]

    def add(tool_name: str, line: str) -> None:
        if allowed is not None and tool_name not in allowed:
            return
        if tool_name.startswith("repo_") and not _repo_tools_enabled():
            return
        lines.append(line)

    add("project_counts_snapshot", "- project_counts_snapshot(max_categories: int = 20)")
    add("latest_activity", "- latest_activity(limit: int = 20, kinds: list[str] | None = None, current_only: bool = false)")
    add("billing_category_counts", "- billing_category_counts(current_only: bool = false, limit: int = 20)")
    add("db_file_stats", "- db_file_stats()  # show sqlite DB file sizes")
    add("assistant_stats", "- assistant_stats()  # assistant DB stats and review backlog")
    add("assistant_recent_sessions", "- assistant_recent_sessions(limit: int = 10)")
    add("assistant_get_session_review", "- assistant_get_session_review(session_id: str)")
    add(
        "assistant_prompt_header",
        "- assistant_prompt_header(session_id: str, user_message_id: int | None = None, include_legend: bool = false)",
    )
    add(
        "assistant_search_messages",
        "- assistant_search_messages(query: str, limit: int = 20, role: str | None = None, session_id: str | None = None, user_id: int | None = None)  # admin-only",
    )
    add(
        "assistant_get_message_context",
        "- assistant_get_message_context(message_id: int, before: int = 3, after: int = 3, max_chars: int = 800)  # admin-only",
    )
    add(
        "assistant_list_digests",
        "- assistant_list_digests(limit: int = 10, key: str | None = 'global', user_id: int | None = None)  # admin-only",
    )
    add("assistant_get_digest", "- assistant_get_digest(digest_id: int)  # admin-only")
    add(
        "assistant_search_digests",
        "- assistant_search_digests(query: str, limit: int = 20, key: str | None = None, user_id: int | None = None)  # admin-only",
    )
    add(
        "assistant_search_internal_logs",
        "- assistant_search_internal_logs(query: str, limit: int = 20)  # admin-only; searches agent runs/steps/commands/events",
    )
    add(
        "assistant_recent_agent_commands",
        "- assistant_recent_agent_commands(limit: int = 20, statuses: list[str] | None = None, command_types: list[str] | None = None, after_id: int | None = None)  # admin-only",
    )
    add(
        "assistant_recent_agent_steps",
        "- assistant_recent_agent_steps(limit: int = 20, kinds: list[str] | None = None, run_id: str | None = None, ok: bool | None = None, after_id: int | None = None)  # admin-only",
    )
    add(
        "assistant_recent_session_reviews",
        "- assistant_recent_session_reviews(limit: int = 20, session_id: str | None = None, user_id: int | None = None, after_id: int | None = None)  # admin-only",
    )
    add("assistant_get_agent_step", "- assistant_get_agent_step(step_id: int)  # admin-only")
    add("assistant_get_agent_command", "- assistant_get_agent_command(command_id: int)  # admin-only")
    add("assistant_get_agent_run", "- assistant_get_agent_run(run_id: str)  # admin-only")
    add("assistant_list_users", "- assistant_list_users(limit: int = 50, include_anonymous: bool = true)  # admin-only")
    add("count_all_projects", "- count_all_projects()  # total projects across all statuses/flags")
    add("count_current_projects", "- count_current_projects()  # current projects only")
    add("project_status_counts", "- project_status_counts(current_only: bool = false)")
    add("latest_projects", "- latest_projects(sort: str = 'modified', limit: int = 10, current_only: bool = false)")
    add("latest_project_comments", "- latest_project_comments(limit: int = 10, project_id: int | None = None)")
    add("search_projects", "- search_projects(query: str, limit: int = 5)")
    add("projects", "- projects(project_id: int)  # alias for get_project")
    add("get_project", "- get_project(id: int)")
    add("my_projects", "- my_projects(limit: int = 50, query: str | None = None, current_only: bool = false)")
    add(
        "project_files_for_project",
        "- project_files_for_project(project_id: int, limit: int = 50, query: str | None = None)",
    )
    add("search_api", "- search_api(query: str, limit: int = 10)  # search FastAPI/OpenAPI endpoints")
    add(
        "create_project_comment",
        "- create_project_comment(project_id: int, comment: str, comment_type: str | None = None, confirm: bool = true)  # write: requires explicit user request; client notes saved as client_note",
    )

    add(
        "repo_list_files",
        f"- repo_list_files(query: str | None = None, path: str | None = None, limit: int = 200)  # dev-only; set {_REPO_TOOLS_ENV}=1",
    )
    add(
        "repo_search",
        f"- repo_search(query: str, path: str | None = None, limit: int = 50, regex: bool = false, ignore_case: bool = true)  # dev-only; set {_REPO_TOOLS_ENV}=1",
    )
    add(
        "repo_read_file",
        f"- repo_read_file(path: str, start_line: int = 1, max_lines: int = 200)  # dev-only; set {_REPO_TOOLS_ENV}=1",
    )

    add("experiments_for_project", "- experiments_for_project(project_id: int, limit: int = 20)")
    add("latest_experiments", "- latest_experiments(limit: int = 5)")
    add("get_experiment", "- get_experiment(id: int)")
    add("latest_experiment_runs", "- latest_experiment_runs(limit: int = 5)")
    add("get_experiment_run", "- get_experiment_run(id: int)")
    add("e2g_search_genes_in_project", "- e2g_search_genes_in_project(project_id: int, query: str, limit: int = 10)")
    add("e2g_gene_in_project", "- e2g_gene_in_project(project_id: int, gene_id: int, limit: int = 50)")
    add("search_people", "- search_people(query: str, limit: int = 5)")
    add("get_person", "- get_person(id: int)")
    add(
        "list_schedule_slots",
        "- list_schedule_slots(start: YYYY-MM-DD, end: YYYY-MM-DD, status: str | None = None, limit: int = 50)",
    )
    add(
        "list_schedule_requests",
        "- list_schedule_requests(limit: int = 20, status: str | None = None)  # admin-only",
    )
    add("get_schedule_request", "- get_schedule_request(id: int)  # admin-only")

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


def _snippet_for_query(text: str | None, query: str, *, window: int = 80, max_len: int = 320) -> str:
    haystack = (text or "").strip()
    needle = (query or "").strip()
    if not haystack or not needle:
        return _safe_str(haystack, max_len=max_len) or ""

    lower_haystack = haystack.lower()
    lower_needle = needle.lower()
    idx = lower_haystack.find(lower_needle)
    if idx < 0:
        return _safe_str(haystack, max_len=max_len) or ""

    start = max(0, idx - max(0, int(window)))
    end = min(len(haystack), idx + len(needle) + max(0, int(window)))
    snippet = haystack[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(haystack):
        snippet = snippet + "…"
    if len(snippet) > max_len:
        snippet = snippet[: max_len - 1] + "…"
    return snippet


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
    assistant_db: Session | None = None,
    agent_db: Session | None = None,
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
            if user.role == UserRole.viewer and not bool(getattr(user, "can_write_project_comments", False)):
                return {"ok": False, "tool": name, "error": "Write access required."}

            is_client_user = user.role == UserRole.client

            confirm = args.get("confirm")
            if confirm is not True:
                return {"ok": False, "tool": name, "error": "confirm=true is required to write project history."}

            user_msg = (user_message or "").strip().lower()
            if user_msg and (
                re.search(r"\b(do not|don't|dont)\s+(save|log|record|add|commit)\b", user_msg)
                or re.search(r"\bnot\s+(?:save|commit)\b", user_msg)
            ):
                return {
                    "ok": False,
                    "tool": name,
                    "error": "User requested not to save yet.",
                }
            explicit = False
            if user_msg:
                verbs = r"(?:save|log|record|add|write|create|make|leave|commit)"
                nouns = r"(?:history|comment|comments|note|notes|meeting|memo)"
                if re.search(rf"\b{verbs}\b", user_msg) and re.search(rf"\b{nouns}\b", user_msg):
                    explicit = True
                else:
                    normalized = re.sub(r"[^\w\s]", "", user_msg)
                    normalized = re.sub(r"\s+", " ", normalized).strip()
                    if re.fullmatch(r"(?:yes|y|yeah|yep|yup|ok|okay|sure|confirm)(?: please)?", normalized):
                        explicit = True
                    elif normalized in {"go ahead", "do it", "please do", "please do it"}:
                        explicit = True
                    else:
                        tokens = normalized.split()
                        if len(tokens) <= 6:
                            negative = {"no", "n", "nope", "nah"}
                            if any(token in negative for token in tokens):
                                explicit = False
                            else:
                                affirmative = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm"}
                                if "commit" in tokens and ("it" in tokens or any(token in affirmative for token in tokens)):
                                    explicit = True
                                elif normalized in {"commit", "commit it", "please commit", "please commit it"}:
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
            if is_client_user:
                comment_type = "client_note"

            if is_client_user:
                project = (
                    core_db.query(Project)
                    .join(AuthUserProject, AuthUserProject.project_id == Project.id)
                    .filter(Project.id == int(project_id), AuthUserProject.user_id == user.id)
                    .first()
                )
            else:
                project = core_db.get(Project, int(project_id))
            if project is None:
                if is_client_user:
                    return {
                        "ok": False,
                        "tool": name,
                        "error": f"Project {project_id} not found or not accessible.",
                    }
                return {"ok": False, "tool": name, "error": f"Project {project_id} not found."}

            person_id = _safe_int(args.get("person_id")) if not is_client_user else None
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

        if name == "assistant_prompt_header":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}

            session_id = _safe_str(args.get("session_id"), max_len=256) or _safe_str(args.get("sessionId"), max_len=256)
            if not session_id:
                return {"ok": False, "tool": name, "error": "Missing argument: session_id"}

            include_legend = bool(args.get("include_legend"))
            user_message_id = _safe_int(args.get("user_message_id"))
            if user_message_id is None:
                user_message_id = _safe_int(args.get("message_id"))
            if user_message_id is None:
                user_message_id = _safe_int(args.get("userMessageId"))

            session = assistant_db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
            if session is None:
                return {"ok": False, "tool": name, "error": "Session not found.", "session_id": session_id}

            state: dict[str, Any] = {}
            raw_state = getattr(session, "state_json", None)
            if raw_state:
                try:
                    parsed_state = json.loads(raw_state)
                    if isinstance(parsed_state, dict):
                        state = parsed_state
                except Exception:
                    state = {}

            if user_message_id is None:
                last_user_message = (
                    assistant_db.query(AssistantSupportMessage)
                    .filter(AssistantSupportMessage.session_pk == int(session.id))
                    .filter(AssistantSupportMessage.role == "user")
                    .order_by(AssistantSupportMessage.id.desc())
                    .first()
                )
                if last_user_message is not None:
                    user_message_id = int(last_user_message.id)

            tool_protocol = (os.getenv("ISPEC_ASSISTANT_TOOL_PROTOCOL") or "line").strip()
            header = build_prompt_header(
                session_id=session.session_id,
                user_role=user.role if user is not None else None,
                user_id=int(user.id) if user is not None else None,
                session_state=state,
                user_message_id=user_message_id,
                tools_available=True,
                tool_protocol=tool_protocol,
                compare_mode=False,
                forced_tool_choice=False,
                repo_tools_enabled=_repo_tools_enabled(),
            )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "session_id": session.session_id,
                    "header_line": header.line,
                    "header_fields": header.fields,
                    "legend": header_legend() if include_legend else None,
                },
            }

        if name == "assistant_stats":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}

            sessions_total = int(assistant_db.query(func.count(SupportSession.id)).scalar() or 0)
            messages_total = int(assistant_db.query(func.count(AssistantSupportMessage.id)).scalar() or 0)
            memories_total = int(assistant_db.query(func.count(SupportMemory.id)).scalar() or 0)

            session_rows = (
                assistant_db.query(SupportSession)
                .order_by(SupportSession.updated_at.desc(), SupportSession.id.desc())
                .limit(250)
                .all()
            )
            last_message_rows = (
                assistant_db.query(AssistantSupportMessage.session_pk, func.max(AssistantSupportMessage.id))
                .group_by(AssistantSupportMessage.session_pk)
                .all()
            )
            last_message_by_session = {int(session_pk): int(last_id or 0) for session_pk, last_id in last_message_rows}
            last_message_ids = [int(last_id or 0) for _, last_id in last_message_rows if last_id]
            last_role_by_id = {}
            if last_message_ids:
                role_rows = (
                    assistant_db.query(AssistantSupportMessage.id, AssistantSupportMessage.role)
                    .filter(AssistantSupportMessage.id.in_(last_message_ids))
                    .all()
                )
                last_role_by_id = {int(message_id): role for message_id, role in role_rows}
            last_role_by_session = {
                int(session_pk): last_role_by_id.get(int(last_id or 0))
                for session_pk, last_id in last_message_rows
            }
            last_assistant_rows = (
                assistant_db.query(AssistantSupportMessage.session_pk, func.max(AssistantSupportMessage.id))
                .filter(AssistantSupportMessage.role == "assistant")
                .group_by(AssistantSupportMessage.session_pk)
                .all()
            )
            last_assistant_by_session = {int(session_pk): int(last_id or 0) for session_pk, last_id in last_assistant_rows}

            session_pks = [int(session.id) for session in session_rows]
            reviewed_up_to_by_session: dict[int, int] = {}
            if session_pks:
                review_rows = (
                    assistant_db.query(SupportSessionReview.session_pk, func.max(SupportSessionReview.target_message_id))
                    .filter(SupportSessionReview.session_pk.in_(session_pks))
                    .group_by(SupportSessionReview.session_pk)
                    .all()
                )
                reviewed_up_to_by_session = {int(session_pk): int(max_id or 0) for session_pk, max_id in review_rows}

            needs_review = 0
            reviewed = 0
            pending_user_turn = 0
            no_assistant_messages = 0
            for session in session_rows:
                session_pk = int(session.id)
                last_id = int(last_message_by_session.get(session_pk, 0))
                last_role = last_role_by_session.get(session_pk)
                last_assistant_id = int(last_assistant_by_session.get(session_pk, 0))
                if last_assistant_id <= 0:
                    no_assistant_messages += 1
                    continue
                if last_role != "assistant":
                    pending_user_turn += 1
                    continue
                reviewed_up_to = reviewed_up_to_by_session.get(session_pk)
                if reviewed_up_to is None:
                    state: dict[str, Any] = {}
                    raw_state = getattr(session, "state_json", None)
                    if raw_state:
                        try:
                            parsed_state = json.loads(raw_state)
                            if isinstance(parsed_state, dict):
                                state = parsed_state
                        except Exception:
                            state = {}
                    reviewed_up_to = _safe_int(state.get("conversation_review_up_to_id")) or 0
                if last_id > 0 and reviewed_up_to >= last_assistant_id:
                    reviewed += 1
                elif last_assistant_id > reviewed_up_to:
                    needs_review += 1

            queued_commands: int | None = None
            running_commands: int | None = None
            last_run: AgentRun | None = None
            orchestrator_state = None
            if agent_db is not None:
                queued_commands = int(
                    agent_db.query(func.count(AgentCommand.id))
                    .filter(AgentCommand.status == "queued")
                    .scalar()
                    or 0
                )
                running_commands = int(
                    agent_db.query(func.count(AgentCommand.id))
                    .filter(AgentCommand.status == "running")
                    .scalar()
                    or 0
                )
                last_run = (
                    agent_db.query(AgentRun)
                    .filter(AgentRun.kind == "supervisor")
                    .order_by(AgentRun.id.desc())
                    .first()
                )
                if last_run is not None and isinstance(last_run.summary_json, dict):
                    orchestrator_state = last_run.summary_json.get("orchestrator")

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "sessions_total": sessions_total,
                    "messages_total": messages_total,
                    "memories_total": memories_total,
                    "sessions_needing_review": needs_review,
                    "sessions_reviewed": reviewed,
                    "sessions_pending_user_turn": pending_user_turn,
                    "sessions_without_assistant_messages": no_assistant_messages,
                    "agent_commands": (
                        {"queued": queued_commands, "running": running_commands}
                        if queued_commands is not None and running_commands is not None
                        else None
                    ),
                    "latest_supervisor_run": {
                        "run_id": getattr(last_run, "run_id", None),
                        "agent_id": getattr(last_run, "agent_id", None),
                        "status": getattr(last_run, "status", None),
                        "updated_at": getattr(last_run, "updated_at", None).isoformat() if getattr(last_run, "updated_at", None) else None,
                        "orchestrator": orchestrator_state,
                    }
                    if last_run is not None
                    else None,
                },
            }

        if name == "assistant_recent_agent_commands":
            if agent_db is None:
                return {"ok": False, "tool": name, "error": "agent_db is required."}

            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=200)
            statuses = _coerce_str_list(args.get("statuses"), max_items=10) or _coerce_str_list(args.get("status"), max_items=10)
            command_types = _coerce_str_list(args.get("command_types"), max_items=20) or _coerce_str_list(args.get("command_type"), max_items=20)
            after_id = _safe_int(args.get("after_id"))
            order = (_safe_str(args.get("order"), max_len=8) or "desc").lower()
            if order not in {"asc", "desc"}:
                return {"ok": False, "tool": name, "error": "order must be 'asc' or 'desc'."}
            include_payload = bool(args.get("include_payload"))
            include_result = bool(args.get("include_result"))

            query = agent_db.query(AgentCommand)
            if statuses:
                normalized_statuses = [s.strip().lower() for s in statuses if s.strip()]
                query = query.filter(AgentCommand.status.in_(normalized_statuses))
            if command_types:
                normalized_types = [s.strip() for s in command_types if s.strip()]
                query = query.filter(AgentCommand.command_type.in_(normalized_types))
            if after_id is not None and after_id > 0:
                query = query.filter(AgentCommand.id > int(after_id))

            if order == "asc":
                query = query.order_by(AgentCommand.id.asc())
            else:
                query = query.order_by(AgentCommand.id.desc())

            rows = query.limit(limit).all()
            items: list[dict[str, Any]] = []
            for row in rows:
                payload_json = row.payload_json if isinstance(row.payload_json, dict) else {}
                result_json = row.result_json if isinstance(row.result_json, dict) else {}
                item: dict[str, Any] = {
                    "id": int(row.id),
                    "command_type": row.command_type,
                    "status": row.status,
                    "priority": int(row.priority or 0),
                    "attempts": int(row.attempts or 0),
                    "max_attempts": int(row.max_attempts or 0),
                    "available_at": row.available_at.isoformat() if row.available_at else None,
                    "claimed_at": row.claimed_at.isoformat() if row.claimed_at else None,
                    "claimed_by_agent_id": row.claimed_by_agent_id,
                    "claimed_by_run_id": row.claimed_by_run_id,
                    "started_at": row.started_at.isoformat() if row.started_at else None,
                    "ended_at": row.ended_at.isoformat() if row.ended_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "error": row.error,
                    "payload_keys": sorted(payload_json.keys())[:50],
                    "result_keys": sorted(result_json.keys())[:50],
                }
                if include_payload:
                    item["payload_json"] = payload_json
                if include_result:
                    item["result_json"] = result_json
                items.append(item)

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "limit": limit,
                    "order": order,
                    "after_id": after_id,
                    "filters": {"statuses": statuses, "command_types": command_types},
                    "count": len(items),
                    "commands": items,
                },
            }

        if name == "assistant_recent_agent_steps":
            if agent_db is None:
                return {"ok": False, "tool": name, "error": "agent_db is required."}

            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=200)
            kinds = _coerce_str_list(args.get("kinds"), max_items=20) or _coerce_str_list(args.get("kind"), max_items=20)
            run_id = _safe_str(args.get("run_id"), max_len=128)
            ok_filter = args.get("ok")
            if ok_filter is not None and not isinstance(ok_filter, bool):
                ok_filter = bool(ok_filter)
            after_id = _safe_int(args.get("after_id"))
            order = (_safe_str(args.get("order"), max_len=8) or "desc").lower()
            if order not in {"asc", "desc"}:
                return {"ok": False, "tool": name, "error": "order must be 'asc' or 'desc'."}
            include_prompt = bool(args.get("include_prompt"))
            include_response = bool(args.get("include_response"))
            include_state = bool(args.get("include_state"))

            query = agent_db.query(AgentStep, AgentRun.run_id).join(AgentRun, AgentRun.id == AgentStep.run_pk)
            if run_id:
                query = query.filter(AgentRun.run_id == run_id)
            if kinds:
                normalized = [k.strip() for k in kinds if k.strip()]
                query = query.filter(AgentStep.kind.in_(normalized))
            if ok_filter is not None:
                query = query.filter(AgentStep.ok.is_(bool(ok_filter)))
            if after_id is not None and after_id > 0:
                query = query.filter(AgentStep.id > int(after_id))

            if order == "asc":
                query = query.order_by(AgentStep.id.asc())
            else:
                query = query.order_by(AgentStep.id.desc())

            rows = query.limit(limit).all()
            items: list[dict[str, Any]] = []
            for step, step_run_id in rows:
                chosen = step.chosen_json if isinstance(step.chosen_json, dict) else {}
                command_id = _safe_int(chosen.get("command_id"))
                command_type = str(chosen.get("command_type") or "").strip() or None
                item: dict[str, Any] = {
                    "id": int(step.id),
                    "run_id": step_run_id,
                    "run_pk": int(step.run_pk),
                    "step_index": int(step.step_index or 0),
                    "kind": step.kind,
                    "ok": bool(step.ok),
                    "severity": step.severity,
                    "duration_ms": int(step.duration_ms) if step.duration_ms is not None else None,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "ended_at": step.ended_at.isoformat() if step.ended_at else None,
                    "error": step.error,
                    "command_id": command_id,
                    "command_type": command_type,
                }
                if include_prompt:
                    item["prompt_json"] = step.prompt_json
                if include_response:
                    item["response_json"] = step.response_json
                if include_state:
                    item["state_before_json"] = step.state_before_json
                    item["state_after_json"] = step.state_after_json
                items.append(item)

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "limit": limit,
                    "order": order,
                    "after_id": after_id,
                    "filters": {"kinds": kinds, "run_id": run_id, "ok": ok_filter},
                    "count": len(items),
                    "steps": items,
                },
            }

        if name == "assistant_recent_session_reviews":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}

            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=200)
            session_id = _safe_str(args.get("session_id"), max_len=256)
            user_id = _safe_int(args.get("user_id"))
            after_id = _safe_int(args.get("after_id"))
            order = (_safe_str(args.get("order"), max_len=8) or "desc").lower()
            if order not in {"asc", "desc"}:
                return {"ok": False, "tool": name, "error": "order must be 'asc' or 'desc'."}
            include_review_json = bool(args.get("include_review_json"))

            query = assistant_db.query(SupportSessionReview, SupportSession.session_id, SupportSession.user_id).join(
                SupportSession, SupportSession.id == SupportSessionReview.session_pk
            )
            if session_id:
                query = query.filter(SupportSession.session_id == session_id)
            if user_id is not None:
                query = query.filter(SupportSession.user_id == int(user_id))
            if after_id is not None and after_id > 0:
                query = query.filter(SupportSessionReview.id > int(after_id))

            if order == "asc":
                query = query.order_by(SupportSessionReview.id.asc())
            else:
                query = query.order_by(SupportSessionReview.id.desc())

            rows = query.limit(limit).all()
            items: list[dict[str, Any]] = []
            for review, review_session_id, review_user_id in rows:
                review_json = review.review_json if isinstance(review.review_json, dict) else {}
                summary = review_json.get("summary")
                if not isinstance(summary, str):
                    summary = ""
                issues = review_json.get("issues")
                issue_count = len(issues) if isinstance(issues, list) else 0
                followups = review_json.get("followups")
                followup_count = len(followups) if isinstance(followups, list) else 0
                repo_queries = review_json.get("repo_search_queries")
                repo_query_count = len(repo_queries) if isinstance(repo_queries, list) else 0

                item: dict[str, Any] = {
                    "id": int(review.id),
                    "session_id": review_session_id,
                    "session_pk": int(review.session_pk),
                    "user_id": int(review_user_id) if review_user_id is not None else None,
                    "target_message_id": int(review.target_message_id),
                    "updated_at": review.updated_at.isoformat() if review.updated_at else None,
                    "issues_count": issue_count,
                    "followups_count": followup_count,
                    "repo_search_queries_count": repo_query_count,
                    "summary": summary[:400] + ("…" if isinstance(summary, str) and len(summary) > 400 else ""),
                }
                if include_review_json:
                    item["review_json"] = review_json
                items.append(item)

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "limit": limit,
                    "order": order,
                    "after_id": after_id,
                    "filters": {"session_id": session_id, "user_id": user_id},
                    "count": len(items),
                    "reviews": items,
                },
            }

        if name == "assistant_recent_sessions":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}

            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            sessions = (
                assistant_db.query(SupportSession)
                .order_by(SupportSession.updated_at.desc(), SupportSession.id.desc())
                .limit(limit)
                .all()
            )
            session_pks = [int(session.id) for session in sessions]
            review_by_session_pk: dict[int, dict[str, Any]] = {}
            if session_pks:
                review_rows = (
                    assistant_db.query(
                        SupportSessionReview.session_pk,
                        SupportSessionReview.target_message_id,
                        SupportSessionReview.updated_at,
                    )
                    .filter(SupportSessionReview.session_pk.in_(session_pks))
                    .order_by(
                        SupportSessionReview.session_pk.asc(),
                        SupportSessionReview.target_message_id.desc(),
                        SupportSessionReview.updated_at.desc(),
                        SupportSessionReview.id.desc(),
                    )
                    .all()
                )
                for session_pk, target_id, updated_at in review_rows:
                    key = int(session_pk)
                    if key in review_by_session_pk:
                        continue
                    review_by_session_pk[key] = {
                        "reviewed_up_to_id": int(target_id or 0),
                        "review_updated_at": updated_at.isoformat() if updated_at else None,
                    }

            items: list[dict[str, Any]] = []
            for session in sessions:
                msg_count = int(
                    assistant_db.query(func.count(AssistantSupportMessage.id))
                    .filter(AssistantSupportMessage.session_pk == session.id)
                    .scalar()
                    or 0
                )
                last_message = (
                    assistant_db.query(AssistantSupportMessage)
                    .filter(AssistantSupportMessage.session_pk == session.id)
                    .order_by(AssistantSupportMessage.id.desc())
                    .first()
                )
                last_user = (
                    assistant_db.query(AssistantSupportMessage)
                    .filter(AssistantSupportMessage.session_pk == session.id)
                    .filter(AssistantSupportMessage.role == "user")
                    .order_by(AssistantSupportMessage.id.desc())
                    .first()
                )
                last_assistant = (
                    assistant_db.query(AssistantSupportMessage)
                    .filter(AssistantSupportMessage.session_pk == session.id)
                    .filter(AssistantSupportMessage.role == "assistant")
                    .order_by(AssistantSupportMessage.id.desc())
                    .first()
                )
                state: dict[str, Any] = {}
                raw_state = getattr(session, "state_json", None)
                if raw_state:
                    try:
                        parsed_state = json.loads(raw_state)
                        if isinstance(parsed_state, dict):
                            state = parsed_state
                    except Exception:
                        state = {}
                review_info = review_by_session_pk.get(int(session.id))
                if isinstance(review_info, dict):
                    reviewed_up_to_id = _safe_int(review_info.get("reviewed_up_to_id")) or 0
                    review_updated_at = review_info.get("review_updated_at")
                else:
                    reviewed_up_to_id = _safe_int(state.get("conversation_review_up_to_id")) or 0
                    review_updated_at = state.get("conversation_review_updated_at")
                items.append(
                    {
                        "session_id": session.session_id,
                        "user_id": int(session.user_id) if session.user_id is not None else None,
                        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                        "message_count": msg_count,
                        "last_message_id": int(last_message.id) if last_message is not None else 0,
                        "last_message_role": getattr(last_message, "role", None),
                        "last_assistant_message_id": int(last_assistant.id) if last_assistant is not None else 0,
                        "last_user_message": _safe_str(getattr(last_user, "content", None), max_len=240),
                        "reviewed_up_to_id": reviewed_up_to_id,
                        "review_updated_at": review_updated_at,
                        "memory_up_to_id": _safe_int(state.get("conversation_memory_up_to_id")) or 0,
                    }
                )

            return {"ok": True, "tool": name, "result": {"limit": limit, "sessions": items}}

        if name == "assistant_get_session_review":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}
            session_id = _safe_str(args.get("session_id"), max_len=256) or _safe_str(args.get("id"), max_len=256)
            if not session_id:
                return {"ok": False, "tool": name, "error": "Missing argument: session_id"}
            session = assistant_db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
            if session is None:
                return {"ok": False, "tool": name, "error": "Session not found.", "session_id": session_id}
            review_row = (
                assistant_db.query(SupportSessionReview)
                .filter(SupportSessionReview.session_pk == int(session.id))
                .order_by(
                    SupportSessionReview.target_message_id.desc(),
                    SupportSessionReview.updated_at.desc(),
                    SupportSessionReview.id.desc(),
                )
                .first()
            )
            if review_row is not None:
                review = review_row.review_json if isinstance(review_row.review_json, dict) else None
                reviewed_up_to_id = int(review_row.target_message_id or 0)
                review_updated_at = review_row.updated_at.isoformat() if review_row.updated_at else None
            else:
                state: dict[str, Any] = {}
                raw_state = getattr(session, "state_json", None)
                if raw_state:
                    try:
                        parsed_state = json.loads(raw_state)
                        if isinstance(parsed_state, dict):
                            state = parsed_state
                    except Exception:
                        state = {}
                review = state.get("conversation_review") if isinstance(state.get("conversation_review"), dict) else None
                reviewed_up_to_id = _safe_int(state.get("conversation_review_up_to_id")) or 0
                review_updated_at = state.get("conversation_review_updated_at")
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "session_id": session.session_id,
                    "reviewed_up_to_id": reviewed_up_to_id,
                    "review_updated_at": review_updated_at,
                    "review": review,
                },
            }

        if name == "assistant_search_messages":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}
            query_text = _safe_str(args.get("query"), max_len=512) or _safe_str(args.get("q"), max_len=512)
            if not query_text:
                return {"ok": False, "tool": name, "error": "Missing argument: query"}

            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=100)
            role = _safe_str(args.get("role"), max_len=16)
            session_id = _safe_str(args.get("session_id"), max_len=256)
            user_id = _safe_int(args.get("user_id"))

            pattern = f"%{query_text}%"
            query = (
                assistant_db.query(
                    AssistantSupportMessage,
                    SupportSession.session_id,
                    SupportSession.user_id,
                )
                .join(SupportSession, AssistantSupportMessage.session_pk == SupportSession.id)
                .filter(AssistantSupportMessage.content.ilike(pattern))
            )
            if role:
                query = query.filter(AssistantSupportMessage.role == role)
            if session_id:
                query = query.filter(SupportSession.session_id == session_id)
            if user_id is not None:
                query = query.filter(SupportSession.user_id == int(user_id))

            rows = (
                query.order_by(AssistantSupportMessage.id.desc())
                .limit(limit)
                .all()
            )
            matches: list[dict[str, Any]] = []
            for message, sess_id, sess_user_id in rows:
                content = getattr(message, "content", "") or ""
                matches.append(
                    {
                        "message_id": int(message.id),
                        "session_id": sess_id,
                        "user_id": int(sess_user_id) if sess_user_id is not None else None,
                        "role": getattr(message, "role", None),
                        "created_at": message.created_at.isoformat() if getattr(message, "created_at", None) else None,
                        "snippet": _snippet_for_query(content, query_text, window=80, max_len=320),
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "query": query_text,
                    "limit": limit,
                    "role": role,
                    "session_id": session_id,
                    "user_id": int(user_id) if user_id is not None else None,
                    "count": len(matches),
                    "matches": matches,
                },
            }

        if name == "assistant_get_message_context":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}
            message_id = _safe_int(args.get("message_id"))
            if message_id is None:
                message_id = _safe_int(args.get("id"))
            if message_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: message_id (or id)"}

            before = _clamp_int(_safe_int(args.get("before")), default=3, min_value=0, max_value=20)
            after = _clamp_int(_safe_int(args.get("after")), default=3, min_value=0, max_value=20)
            max_chars = _clamp_int(_safe_int(args.get("max_chars")), default=800, min_value=50, max_value=5000)

            anchor = assistant_db.query(AssistantSupportMessage).filter(AssistantSupportMessage.id == int(message_id)).first()
            if anchor is None:
                return {"ok": False, "tool": name, "error": "Message not found.", "message_id": int(message_id)}

            session = assistant_db.query(SupportSession).filter(SupportSession.id == int(anchor.session_pk)).first()
            session_id = session.session_id if session is not None else None
            session_user_id = int(session.user_id) if session is not None and session.user_id is not None else None

            previous_rows = (
                assistant_db.query(AssistantSupportMessage)
                .filter(AssistantSupportMessage.session_pk == int(anchor.session_pk))
                .filter(AssistantSupportMessage.id < int(anchor.id))
                .order_by(AssistantSupportMessage.id.desc())
                .limit(before)
                .all()
            )
            previous_rows.reverse()
            next_rows = (
                assistant_db.query(AssistantSupportMessage)
                .filter(AssistantSupportMessage.session_pk == int(anchor.session_pk))
                .filter(AssistantSupportMessage.id > int(anchor.id))
                .order_by(AssistantSupportMessage.id.asc())
                .limit(after)
                .all()
            )

            rows = [*previous_rows, anchor, *next_rows]
            messages = [
                {
                    "id": int(row.id),
                    "role": row.role,
                    "created_at": row.created_at.isoformat() if getattr(row, "created_at", None) else None,
                    "content": _safe_str(getattr(row, "content", None), max_len=max_chars) or "",
                }
                for row in rows
            ]

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "message_id": int(anchor.id),
                    "session_id": session_id,
                    "user_id": session_user_id,
                    "before": before,
                    "after": after,
                    "count": len(messages),
                    "messages": messages,
                },
            }

        if name == "assistant_list_digests":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}

            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=100)
            key = _safe_str(args.get("key"), max_len=128) if args.get("key") is not None else "global"
            user_id = _safe_int(args.get("user_id"))

            query = (
                assistant_db.query(SupportMemory, SupportSession.session_id)
                .outerjoin(SupportSession, SupportMemory.session_pk == SupportSession.id)
                .filter(SupportMemory.kind == "digest")
            )
            if key:
                query = query.filter(SupportMemory.key == key)
            if user_id is not None:
                query = query.filter(SupportMemory.user_id == int(user_id))

            rows = (
                query.order_by(SupportMemory.created_at.desc(), SupportMemory.id.desc())
                .limit(limit)
                .all()
            )
            items: list[dict[str, Any]] = []
            for digest, session_id in rows:
                digest_json: dict[str, Any] | None = None
                raw_value = getattr(digest, "value_json", None)
                if isinstance(raw_value, str) and raw_value.strip():
                    try:
                        parsed_value = json.loads(raw_value)
                        if isinstance(parsed_value, dict):
                            digest_json = parsed_value
                    except Exception:
                        digest_json = None
                summary_preview = ""
                if isinstance(digest_json, dict):
                    summary_preview = _safe_str(digest_json.get("summary"), max_len=240) or ""
                if not summary_preview and isinstance(raw_value, str):
                    summary_preview = _safe_str(raw_value, max_len=240) or ""
                items.append(
                    {
                        "digest_id": int(digest.id),
                        "kind": digest.kind,
                        "key": digest.key,
                        "user_id": int(digest.user_id) if getattr(digest, "user_id", None) is not None else 0,
                        "session_id": session_id,
                        "created_at": digest.created_at.isoformat() if getattr(digest, "created_at", None) else None,
                        "updated_at": digest.updated_at.isoformat() if getattr(digest, "updated_at", None) else None,
                        "summary": summary_preview,
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "limit": limit,
                    "key": key,
                    "user_id": int(user_id) if user_id is not None else None,
                    "count": len(items),
                    "digests": items,
                },
            }

        if name == "assistant_get_digest":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}

            digest_id = _safe_int(args.get("digest_id"))
            if digest_id is None:
                digest_id = _safe_int(args.get("id"))
            if digest_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: digest_id (or id)"}

            digest = (
                assistant_db.query(SupportMemory)
                .filter(SupportMemory.id == int(digest_id))
                .filter(SupportMemory.kind == "digest")
                .first()
            )
            if digest is None:
                return {"ok": False, "tool": name, "error": "Digest not found.", "digest_id": int(digest_id)}

            session_id = None
            if getattr(digest, "session_pk", None) is not None:
                session = assistant_db.query(SupportSession).filter(SupportSession.id == int(digest.session_pk)).first()
                session_id = session.session_id if session is not None else None

            parsed_value: dict[str, Any] | None = None
            raw_value = getattr(digest, "value_json", None)
            if isinstance(raw_value, str) and raw_value.strip():
                try:
                    payload_value = json.loads(raw_value)
                    if isinstance(payload_value, dict):
                        parsed_value = payload_value
                except Exception:
                    parsed_value = None

            evidence_rows = (
                assistant_db.query(SupportMemoryEvidence.message_id)
                .filter(SupportMemoryEvidence.memory_id == int(digest.id))
                .order_by(SupportMemoryEvidence.message_id.asc())
                .all()
            )
            evidence_message_ids = [int(row[0]) for row in evidence_rows if row and int(row[0]) > 0]

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "digest_id": int(digest.id),
                    "key": digest.key,
                    "user_id": int(digest.user_id) if getattr(digest, "user_id", None) is not None else 0,
                    "session_id": session_id,
                    "created_at": digest.created_at.isoformat() if getattr(digest, "created_at", None) else None,
                    "updated_at": digest.updated_at.isoformat() if getattr(digest, "updated_at", None) else None,
                    "digest": parsed_value,
                    "raw": raw_value if parsed_value is None else None,
                    "evidence_message_ids": evidence_message_ids,
                },
            }

        if name == "assistant_search_digests":
            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}
            query_text = _safe_str(args.get("query"), max_len=512) or _safe_str(args.get("q"), max_len=512)
            if not query_text:
                return {"ok": False, "tool": name, "error": "Missing argument: query"}

            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=100)
            key = _safe_str(args.get("key"), max_len=128)
            user_id = _safe_int(args.get("user_id"))

            pattern = f"%{query_text}%"
            query = assistant_db.query(SupportMemory).filter(SupportMemory.kind == "digest").filter(
                cast(SupportMemory.value_json, Text).ilike(pattern)
            )
            if key:
                query = query.filter(SupportMemory.key == key)
            if user_id is not None:
                query = query.filter(SupportMemory.user_id == int(user_id))

            rows = (
                query.order_by(SupportMemory.created_at.desc(), SupportMemory.id.desc())
                .limit(limit)
                .all()
            )
            matches: list[dict[str, Any]] = []
            for digest in rows:
                raw_value = getattr(digest, "value_json", "") or ""
                matches.append(
                    {
                        "digest_id": int(digest.id),
                        "key": digest.key,
                        "user_id": int(digest.user_id) if getattr(digest, "user_id", None) is not None else 0,
                        "created_at": digest.created_at.isoformat() if getattr(digest, "created_at", None) else None,
                        "snippet": _snippet_for_query(raw_value, query_text, window=80, max_len=320),
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "query": query_text,
                    "limit": limit,
                    "key": key,
                    "user_id": int(user_id) if user_id is not None else None,
                    "count": len(matches),
                    "matches": matches,
                },
            }

        if name == "assistant_list_users":
            include_anonymous = True if args.get("include_anonymous") is None else bool(args.get("include_anonymous"))
            limit = _clamp_int(_safe_int(args.get("limit")), default=50, min_value=1, max_value=200)
            linkage_session_limit = _clamp_int(
                _safe_int(args.get("linkage_session_limit")),
                default=50,
                min_value=1,
                max_value=500,
            )

            if assistant_db is None:
                return {"ok": False, "tool": name, "error": "assistant_db is required."}

            session_counts_query = (
                assistant_db.query(
                    SupportSession.user_id,
                    func.count(SupportSession.id),
                    func.max(SupportSession.updated_at),
                )
                .group_by(SupportSession.user_id)
                .order_by(func.max(SupportSession.updated_at).desc())
            )
            if not include_anonymous:
                session_counts_query = session_counts_query.filter(SupportSession.user_id.isnot(None))

            session_counts_rows = session_counts_query.limit(limit).all()
            user_ids: list[int] = [
                int(user_id)
                for user_id, _, _ in session_counts_rows
                if user_id is not None
            ]

            user_rows = []
            if user_ids:
                user_rows = core_db.query(AuthUser).filter(AuthUser.id.in_(user_ids)).all()
            user_by_id = {int(user.id): user for user in user_rows}

            message_counts_query = (
                assistant_db.query(
                    SupportSession.user_id,
                    AssistantSupportMessage.role,
                    func.count(AssistantSupportMessage.id),
                )
                .join(SupportSession, AssistantSupportMessage.session_pk == SupportSession.id)
                .group_by(SupportSession.user_id, AssistantSupportMessage.role)
            )
            if not include_anonymous:
                message_counts_query = message_counts_query.filter(SupportSession.user_id.isnot(None))
            message_counts_rows = message_counts_query.all()
            message_counts: dict[int | None, dict[str, int]] = {}
            for raw_user_id, role, count in message_counts_rows:
                key = int(raw_user_id) if raw_user_id is not None else None
                bucket = message_counts.get(key)
                if bucket is None:
                    bucket = {}
                    message_counts[key] = bucket
                bucket[str(role or "unknown")] = int(count or 0)

            users: list[dict[str, Any]] = []
            for raw_user_id, sessions_count, last_updated_at in session_counts_rows:
                key = int(raw_user_id) if raw_user_id is not None else None
                auth_user = user_by_id.get(int(raw_user_id)) if raw_user_id is not None else None

                sample_sessions_query = (
                    assistant_db.query(SupportSession)
                    .filter(SupportSession.user_id == raw_user_id)
                    .order_by(SupportSession.updated_at.desc(), SupportSession.id.desc())
                    .limit(5)
                )
                if raw_user_id is None:
                    sample_sessions_query = (
                        assistant_db.query(SupportSession)
                        .filter(SupportSession.user_id.is_(None))
                        .order_by(SupportSession.updated_at.desc(), SupportSession.id.desc())
                        .limit(5)
                    )
                sample_sessions = sample_sessions_query.all()
                sample_session_ids = [str(s.session_id) for s in sample_sessions]

                linkage_sessions_query = (
                    assistant_db.query(SupportSession)
                    .filter(SupportSession.user_id == raw_user_id)
                    .order_by(SupportSession.updated_at.desc(), SupportSession.id.desc())
                    .limit(linkage_session_limit)
                )
                if raw_user_id is None:
                    linkage_sessions_query = (
                        assistant_db.query(SupportSession)
                        .filter(SupportSession.user_id.is_(None))
                        .order_by(SupportSession.updated_at.desc(), SupportSession.id.desc())
                        .limit(linkage_session_limit)
                    )
                linkage_sessions = linkage_sessions_query.all()

                project_counts: dict[int, int] = {}
                route_counts: dict[str, int] = {}
                for session in linkage_sessions:
                    raw_state = getattr(session, "state_json", None)
                    if not raw_state:
                        continue
                    try:
                        state = json.loads(raw_state)
                    except Exception:
                        continue
                    if not isinstance(state, dict):
                        continue

                    for key_name in ("current_project_id", "ui_project_id"):
                        project_id = state.get(key_name)
                        if isinstance(project_id, int) and project_id > 0:
                            project_counts[project_id] = project_counts.get(project_id, 0) + 1

                    ui_route = state.get("ui_route")
                    if isinstance(ui_route, dict):
                        route_name = ui_route.get("name")
                        if isinstance(route_name, str) and route_name.strip():
                            cleaned = route_name.strip()
                            route_counts[cleaned] = route_counts.get(cleaned, 0) + 1

                project_items = [{"project_id": pid, "count": cnt} for pid, cnt in project_counts.items()]
                project_items.sort(key=lambda item: (-int(item["count"]), int(item["project_id"])))
                route_items = [{"route": route, "count": cnt} for route, cnt in route_counts.items()]
                route_items.sort(key=lambda item: (-int(item["count"]), str(item["route"])))

                users.append(
                    {
                        "user_id": key,
                        "username": getattr(auth_user, "username", None) if auth_user is not None else None,
                        "role": str(getattr(auth_user, "role", "")) if auth_user is not None else None,
                        "sessions_count": int(sessions_count or 0),
                        "messages_count": message_counts.get(key, {}),
                        "last_session_updated_at": last_updated_at.isoformat() if last_updated_at else None,
                        "sample_session_ids": sample_session_ids,
                        "linkage": {
                            "projects": project_items[:20],
                            "ui_routes": route_items[:20],
                            "sessions_scanned": len(linkage_sessions),
                        },
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "limit": limit,
                    "include_anonymous": include_anonymous,
                    "count": len(users),
                    "users": users,
                },
            }

        if name == "assistant_search_internal_logs":
            if agent_db is None:
                return {"ok": False, "tool": name, "error": "agent_db is required."}
            query_text = _safe_str(args.get("query"), max_len=512) or _safe_str(args.get("q"), max_len=512)
            if not query_text:
                return {"ok": False, "tool": name, "error": "Missing argument: query"}
            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=100)

            pattern = f"%{query_text}%"
            step_rows = (
                agent_db.query(AgentStep, AgentRun.run_id, AgentRun.agent_id)
                .join(AgentRun, AgentStep.run_pk == AgentRun.id)
                .filter(
                    or_(
                        AgentStep.kind.ilike(pattern),
                        AgentStep.error.ilike(pattern),
                        cast(AgentStep.prompt_json, Text).ilike(pattern),
                        cast(AgentStep.response_json, Text).ilike(pattern),
                        cast(AgentStep.tool_results_json, Text).ilike(pattern),
                        cast(AgentStep.state_before_json, Text).ilike(pattern),
                        cast(AgentStep.state_after_json, Text).ilike(pattern),
                        cast(AgentStep.summary_before_json, Text).ilike(pattern),
                        cast(AgentStep.summary_after_json, Text).ilike(pattern),
                    )
                )
                .order_by(AgentStep.id.desc())
                .limit(limit)
                .all()
            )
            steps: list[dict[str, Any]] = []
            for step, run_id, agent_id in step_rows:
                response_preview = ""
                if isinstance(step.response_json, dict):
                    response_preview = _snippet_for_query(json.dumps(step.response_json, ensure_ascii=False), query_text)
                steps.append(
                    {
                        "step_id": int(step.id),
                        "run_id": run_id,
                        "agent_id": agent_id,
                        "kind": step.kind,
                        "step_index": int(step.step_index or 0),
                        "ok": bool(step.ok),
                        "severity": step.severity,
                        "error": step.error,
                        "started_at": step.started_at.isoformat() if getattr(step, "started_at", None) else None,
                        "ended_at": step.ended_at.isoformat() if getattr(step, "ended_at", None) else None,
                        "response_preview": response_preview,
                    }
                )

            command_rows = (
                agent_db.query(AgentCommand)
                .filter(
                    or_(
                        AgentCommand.command_type.ilike(pattern),
                        AgentCommand.status.ilike(pattern),
                        AgentCommand.error.ilike(pattern),
                        cast(AgentCommand.payload_json, Text).ilike(pattern),
                        cast(AgentCommand.result_json, Text).ilike(pattern),
                    )
                )
                .order_by(AgentCommand.id.desc())
                .limit(limit)
                .all()
            )
            commands: list[dict[str, Any]] = []
            for cmd in command_rows:
                payload_preview = ""
                if isinstance(cmd.payload_json, dict):
                    payload_preview = _snippet_for_query(json.dumps(cmd.payload_json, ensure_ascii=False), query_text)
                result_preview = ""
                if isinstance(cmd.result_json, dict):
                    result_preview = _snippet_for_query(json.dumps(cmd.result_json, ensure_ascii=False), query_text)
                commands.append(
                    {
                        "command_id": int(cmd.id),
                        "command_type": cmd.command_type,
                        "status": cmd.status,
                        "priority": int(cmd.priority or 0),
                        "available_at": cmd.available_at.isoformat() if getattr(cmd, "available_at", None) else None,
                        "claimed_by_run_id": cmd.claimed_by_run_id,
                        "error": cmd.error,
                        "payload_preview": payload_preview,
                        "result_preview": result_preview,
                    }
                )

            run_rows = (
                agent_db.query(AgentRun)
                .filter(
                    or_(
                        AgentRun.run_id.ilike(pattern),
                        AgentRun.agent_id.ilike(pattern),
                        AgentRun.kind.ilike(pattern),
                        AgentRun.status.ilike(pattern),
                        AgentRun.last_error.ilike(pattern),
                        AgentRun.status_bar.ilike(pattern),
                        cast(AgentRun.summary_json, Text).ilike(pattern),
                        cast(AgentRun.state_json, Text).ilike(pattern),
                        cast(AgentRun.config_json, Text).ilike(pattern),
                    )
                )
                .order_by(AgentRun.id.desc())
                .limit(limit)
                .all()
            )
            runs: list[dict[str, Any]] = []
            for run in run_rows:
                summary_preview = ""
                if isinstance(run.summary_json, dict):
                    summary_preview = _snippet_for_query(json.dumps(run.summary_json, ensure_ascii=False), query_text)
                runs.append(
                    {
                        "run_id": run.run_id,
                        "agent_id": run.agent_id,
                        "kind": run.kind,
                        "status": run.status,
                        "updated_at": run.updated_at.isoformat() if getattr(run, "updated_at", None) else None,
                        "status_bar": run.status_bar,
                        "last_error": run.last_error,
                        "summary_preview": summary_preview,
                    }
                )

            event_rows = (
                agent_db.query(AgentEvent)
                .filter(
                    or_(
                        AgentEvent.agent_id.ilike(pattern),
                        AgentEvent.event_type.ilike(pattern),
                        AgentEvent.name.ilike(pattern),
                        AgentEvent.severity.ilike(pattern),
                        AgentEvent.payload_json.ilike(pattern),
                    )
                )
                .order_by(AgentEvent.id.desc())
                .limit(limit)
                .all()
            )
            events: list[dict[str, Any]] = []
            for event in event_rows:
                payload_preview = _snippet_for_query(event.payload_json, query_text)
                events.append(
                    {
                        "event_id": int(event.id),
                        "agent_id": event.agent_id,
                        "event_type": event.event_type,
                        "name": event.name,
                        "severity": event.severity,
                        "ts": event.ts.isoformat() if getattr(event, "ts", None) else None,
                        "payload_preview": payload_preview,
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "query": query_text,
                    "limit": limit,
                    "steps": steps,
                    "commands": commands,
                    "runs": runs,
                    "events": events,
                },
            }

        if name == "assistant_get_agent_step":
            if agent_db is None:
                return {"ok": False, "tool": name, "error": "agent_db is required."}
            step_id = _safe_int(args.get("step_id"))
            if step_id is None:
                step_id = _safe_int(args.get("id"))
            if step_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: step_id (or id)"}

            row = (
                agent_db.query(AgentStep, AgentRun.run_id, AgentRun.agent_id)
                .join(AgentRun, AgentStep.run_pk == AgentRun.id)
                .filter(AgentStep.id == int(step_id))
                .first()
            )
            if row is None:
                return {"ok": False, "tool": name, "error": "Agent step not found.", "step_id": int(step_id)}
            step, run_id, agent_id = row
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "step_id": int(step.id),
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "kind": step.kind,
                    "step_index": int(step.step_index or 0),
                    "started_at": step.started_at.isoformat() if getattr(step, "started_at", None) else None,
                    "ended_at": step.ended_at.isoformat() if getattr(step, "ended_at", None) else None,
                    "duration_ms": int(step.duration_ms) if step.duration_ms is not None else None,
                    "ok": bool(step.ok),
                    "severity": step.severity,
                    "error": step.error,
                    "prompt_json": step.prompt_json,
                    "response_json": step.response_json,
                    "tool_calls_json": step.tool_calls_json,
                    "tool_results_json": step.tool_results_json,
                    "state_before_json": step.state_before_json,
                    "state_after_json": step.state_after_json,
                },
            }

        if name == "assistant_get_agent_command":
            if agent_db is None:
                return {"ok": False, "tool": name, "error": "agent_db is required."}
            command_id = _safe_int(args.get("command_id"))
            if command_id is None:
                command_id = _safe_int(args.get("id"))
            if command_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: command_id (or id)"}

            cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).first()
            if cmd is None:
                return {"ok": False, "tool": name, "error": "Agent command not found.", "command_id": int(command_id)}
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "command_id": int(cmd.id),
                    "command_type": cmd.command_type,
                    "status": cmd.status,
                    "priority": int(cmd.priority or 0),
                    "created_at": cmd.created_at.isoformat() if getattr(cmd, "created_at", None) else None,
                    "updated_at": cmd.updated_at.isoformat() if getattr(cmd, "updated_at", None) else None,
                    "available_at": cmd.available_at.isoformat() if getattr(cmd, "available_at", None) else None,
                    "claimed_at": cmd.claimed_at.isoformat() if getattr(cmd, "claimed_at", None) else None,
                    "claimed_by_agent_id": cmd.claimed_by_agent_id,
                    "claimed_by_run_id": cmd.claimed_by_run_id,
                    "started_at": cmd.started_at.isoformat() if getattr(cmd, "started_at", None) else None,
                    "ended_at": cmd.ended_at.isoformat() if getattr(cmd, "ended_at", None) else None,
                    "attempts": int(cmd.attempts or 0),
                    "max_attempts": int(cmd.max_attempts or 0),
                    "payload_json": cmd.payload_json,
                    "result_json": cmd.result_json,
                    "error": cmd.error,
                },
            }

        if name == "assistant_get_agent_run":
            if agent_db is None:
                return {"ok": False, "tool": name, "error": "agent_db is required."}
            run_id = _safe_str(args.get("run_id"), max_len=128) or _safe_str(args.get("id"), max_len=128)
            if not run_id:
                return {"ok": False, "tool": name, "error": "Missing argument: run_id"}
            run = agent_db.query(AgentRun).filter(AgentRun.run_id == run_id).first()
            if run is None:
                return {"ok": False, "tool": name, "error": "Agent run not found.", "run_id": run_id}
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "run_id": run.run_id,
                    "agent_id": run.agent_id,
                    "kind": run.kind,
                    "status": run.status,
                    "created_at": run.created_at.isoformat() if getattr(run, "created_at", None) else None,
                    "updated_at": run.updated_at.isoformat() if getattr(run, "updated_at", None) else None,
                    "ended_at": run.ended_at.isoformat() if getattr(run, "ended_at", None) else None,
                    "step_index": int(run.step_index or 0),
                    "status_bar": run.status_bar,
                    "last_error": run.last_error,
                    "config_json": run.config_json,
                    "state_json": run.state_json,
                    "summary_json": run.summary_json,
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

        if name == "my_projects":
            query_text = _safe_str(args.get("query"), max_len=200)
            limit = _clamp_int(_safe_int(args.get("limit")), default=50, min_value=1, max_value=500)
            current_only = bool(args.get("current_only"))

            query = core_db.query(Project)
            if user is not None and user.role == UserRole.client:
                query = query.join(
                    AuthUserProject,
                    AuthUserProject.project_id == Project.id,
                ).filter(AuthUserProject.user_id == user.id)

            if current_only:
                query = query.filter(Project.prj_Current_FLAG.is_(True))

            if query_text:
                pattern = f"%{query_text}%"
                query = query.filter(
                    or_(
                        Project.prj_ProjectTitle.ilike(pattern),
                        Project.prj_PI.ilike(pattern),
                        Project.prj_Project_LabContact.ilike(pattern),
                    )
                )

            try:
                total = int(query.order_by(None).count())
            except Exception:
                total = int(query.count())

            rows = query.order_by(Project.id.asc()).limit(limit).all()
            projects = [
                {
                    "id": int(project.id),
                    "title": project.prj_ProjectTitle,
                    "status": project.prj_Status,
                    "current": bool(getattr(project, "prj_Current_FLAG", False)),
                    "to_be_billed": bool(getattr(project, "prj_Billing_ReadyToBill", False)),
                    "links": {
                        "ui": f"/project/{project.id}",
                        "api": f"/api/projects/{project.id}",
                    },
                }
                for project in rows
            ]
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "count": len(projects),
                    "total": total,
                    "projects": projects,
                },
            }

        if name == "project_files_for_project":
            project_id = _safe_int(args.get("project_id"))
            if project_id is None:
                project_id = _safe_int(args.get("id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: project_id (or id)"}

            query_text = _safe_str(args.get("query"), max_len=200)
            limit = _clamp_int(_safe_int(args.get("limit")), default=50, min_value=1, max_value=500)

            project_query = core_db.query(Project).filter(Project.id == project_id)
            if user is not None and user.role == UserRole.client:
                project_query = project_query.join(
                    AuthUserProject,
                    AuthUserProject.project_id == Project.id,
                ).filter(AuthUserProject.user_id == user.id)
            project = project_query.first()
            if project is None:
                return {"ok": False, "tool": name, "error": f"Project {project_id} not found."}

            def display_path(filename: str | None) -> str:
                raw = (filename or "").strip()
                if not raw:
                    return ""
                return raw.replace("__", "/")

            def analysis_key(filename: str | None) -> str:
                display = display_path(filename)
                if not display:
                    return "Ungrouped"
                parts = [part for part in display.split("/") if part]
                return parts[0] if parts else "Ungrouped"

            def relative_path(filename: str | None) -> str:
                display = display_path(filename)
                if not display:
                    return ""
                parts = [part for part in display.split("/") if part]
                if len(parts) <= 1:
                    return display
                return "/".join(parts[1:])

            def file_payload(row: ProjectFile) -> dict[str, Any]:
                analysis = analysis_key(getattr(row, "prjfile_FileName", None))
                display = display_path(getattr(row, "prjfile_FileName", None))
                return {
                    "id": int(row.id),
                    "project_id": int(row.project_id),
                    "analysis": analysis,
                    "path": display,
                    "name": relative_path(getattr(row, "prjfile_FileName", None)),
                    "filename_raw": getattr(row, "prjfile_FileName", None),
                    "content_type": getattr(row, "prjfile_ContentType", None),
                    "size_bytes": int(getattr(row, "prjfile_SizeBytes", 0) or 0),
                    "sha256": getattr(row, "prjfile_Sha256", None),
                    "added_by": getattr(row, "prjfile_AddedBy", None),
                    "created": row.prjfile_CreationTS.isoformat() if row.prjfile_CreationTS else None,
                    "modified": row.prjfile_ModificationTS.isoformat() if row.prjfile_ModificationTS else None,
                    "links": {
                        "download": f"/api/projects/{row.project_id}/files/{row.id}",
                        "preview": f"/api/projects/{row.project_id}/files/{row.id}/preview",
                    },
                }

            file_query = (
                core_db.query(ProjectFile)
                .options(defer(ProjectFile.prjfile_Data))
                .filter(ProjectFile.project_id == project_id)
            )
            if query_text:
                file_query = file_query.filter(ProjectFile.prjfile_FileName.ilike(f"%{query_text}%"))

            try:
                total = int(file_query.order_by(None).count())
            except Exception:
                total = int(file_query.count())

            rows = file_query.order_by(ProjectFile.id.asc()).limit(limit).all()
            files = [file_payload(row) for row in rows]

            by_analysis: dict[str, int] = {}
            for item in files:
                key = str(item.get("analysis") or "Ungrouped")
                by_analysis[key] = by_analysis.get(key, 0) + 1

            pca_hint = re.compile(r"\bpca\b|pc1|pc2|biplot|pcaplot", re.IGNORECASE)
            highlights = [
                item
                for item in files
                if pca_hint.search(str(item.get("path") or ""))
                or pca_hint.search(str(item.get("filename_raw") or ""))
            ][:10]

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project": {
                        "id": int(project.id),
                        "title": project.prj_ProjectTitle,
                        "status": project.prj_Status,
                        "links": {"ui": f"/project/{project.id}", "api": f"/api/projects/{project.id}"},
                    },
                    "count": len(files),
                    "total": total,
                    "by_analysis": by_analysis,
                    "highlights": highlights,
                    "files": files,
                },
            }

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
    "assistant_stats": {
        "type": "function",
        "function": {
            "name": "assistant_stats",
            "description": "Return internal assistant/support-session stats (sessions, messages, memory, review queue).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "assistant_recent_sessions": {
        "type": "function",
        "function": {
            "name": "assistant_recent_sessions",
            "description": "List recent support sessions with message counts and review/memory progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max sessions to return (default 10)."},
                },
            },
        },
    },
    "assistant_get_session_review": {
        "type": "function",
        "function": {
            "name": "assistant_get_session_review",
            "description": "Fetch the latest internal conversation review for a support session, if present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Support session_id to fetch."},
                },
                "required": ["session_id"],
            },
        },
    },
    "assistant_prompt_header": {
        "type": "function",
        "function": {
            "name": "assistant_prompt_header",
            "description": "Return the compact @h1 prompt header (pointers + bitmasks) for a support session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Support session_id to summarize."},
                    "user_message_id": {"type": "integer", "description": "Optional user message id to include in the header."},
                    "include_legend": {"type": "boolean", "description": "If true, include bit legend mappings (default false)."},
                },
                "required": ["session_id"],
            },
        },
    },
    "assistant_search_messages": {
        "type": "function",
        "function": {
            "name": "assistant_search_messages",
            "description": "Search support chat logs (support_message.content) across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for."},
                    "limit": {"type": "integer", "description": "Max matches to return (default 20)."},
                    "role": {"type": "string", "description": "Optional role filter: user|assistant|system."},
                    "session_id": {"type": "string", "description": "Optional session_id to restrict search."},
                    "user_id": {"type": "integer", "description": "Optional user_id to restrict search."},
                },
                "required": ["query"],
            },
        },
    },
    "assistant_get_message_context": {
        "type": "function",
        "function": {
            "name": "assistant_get_message_context",
            "description": "Fetch a support message plus nearby context messages in the same session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "integer", "description": "Support message id."},
                    "before": {"type": "integer", "description": "Messages before (default 3)."},
                    "after": {"type": "integer", "description": "Messages after (default 3)."},
                    "max_chars": {"type": "integer", "description": "Max chars per message content (default 800)."},
                },
                "required": ["message_id"],
            },
        },
    },
    "assistant_list_digests": {
        "type": "function",
        "function": {
            "name": "assistant_list_digests",
            "description": "List stored internal support digests (support_memory kind='digest').",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max digests to return (default 10)."},
                    "key": {"type": "string", "description": "Optional digest key filter (default 'global')."},
                    "user_id": {"type": "integer", "description": "Optional user_id filter."},
                },
            },
        },
    },
    "assistant_get_digest": {
        "type": "function",
        "function": {
            "name": "assistant_get_digest",
            "description": "Fetch a stored support digest by id (includes evidence message ids when available).",
            "parameters": {
                "type": "object",
                "properties": {
                    "digest_id": {"type": "integer", "description": "support_memory.id for kind='digest'."},
                },
                "required": ["digest_id"],
            },
        },
    },
    "assistant_search_digests": {
        "type": "function",
        "function": {
            "name": "assistant_search_digests",
            "description": "Search stored support digests (support_memory kind='digest') by substring match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for."},
                    "limit": {"type": "integer", "description": "Max matches to return (default 20)."},
                    "key": {"type": "string", "description": "Optional digest key filter."},
                    "user_id": {"type": "integer", "description": "Optional user_id filter."},
                },
                "required": ["query"],
            },
        },
    },
    "assistant_list_users": {
        "type": "function",
        "function": {
            "name": "assistant_list_users",
            "description": "List users who have support sessions, including basic linkage signals (projects/routes) from session state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max users to return (default 50)."},
                    "include_anonymous": {"type": "boolean", "description": "Include sessions with user_id=null (default true)."},
                    "linkage_session_limit": {"type": "integer", "description": "Max sessions per user to scan for linkage (default 50)."},
                },
            },
        },
    },
    "assistant_search_internal_logs": {
        "type": "function",
        "function": {
            "name": "assistant_search_internal_logs",
            "description": "Search internal agent/supervisor logs (agent_run/agent_step/agent_command/agent_event).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for."},
                    "limit": {"type": "integer", "description": "Max matches per table (default 20)."},
                },
                "required": ["query"],
            },
        },
    },
    "assistant_recent_agent_commands": {
        "type": "function",
        "function": {
            "name": "assistant_recent_agent_commands",
            "description": "List recent agent_command rows (queue state for the supervisor/agents).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max commands to return (default 20)."},
                    "statuses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional status filter (queued|running|succeeded|failed).",
                    },
                    "command_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional command_type filter.",
                    },
                    "after_id": {"type": "integer", "description": "Only return commands with id > after_id."},
                    "order": {"type": "string", "description": "asc or desc (default desc)."},
                    "include_payload": {"type": "boolean", "description": "Include payload_json (default false)."},
                    "include_result": {"type": "boolean", "description": "Include result_json (default false)."},
                },
            },
        },
    },
    "assistant_recent_agent_steps": {
        "type": "function",
        "function": {
            "name": "assistant_recent_agent_steps",
            "description": "List recent agent_step rows (supervisor action and command execution logs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max steps to return (default 20)."},
                    "kinds": {"type": "array", "items": {"type": "string"}, "description": "Optional step kind filter."},
                    "run_id": {"type": "string", "description": "Optional agent_run.run_id filter."},
                    "ok": {"type": "boolean", "description": "Optional ok filter."},
                    "after_id": {"type": "integer", "description": "Only return steps with id > after_id."},
                    "order": {"type": "string", "description": "asc or desc (default desc)."},
                    "include_prompt": {"type": "boolean", "description": "Include prompt_json (default false)."},
                    "include_response": {"type": "boolean", "description": "Include response_json (default false)."},
                    "include_state": {"type": "boolean", "description": "Include state_before_json/state_after_json (default false)."},
                },
            },
        },
    },
    "assistant_recent_session_reviews": {
        "type": "function",
        "function": {
            "name": "assistant_recent_session_reviews",
            "description": "List recent support_session_review rows (internal QA reviews).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max reviews to return (default 20)."},
                    "session_id": {"type": "string", "description": "Optional support_session.session_id filter."},
                    "user_id": {"type": "integer", "description": "Optional support_session.user_id filter."},
                    "after_id": {"type": "integer", "description": "Only return reviews with id > after_id."},
                    "order": {"type": "string", "description": "asc or desc (default desc)."},
                    "include_review_json": {"type": "boolean", "description": "Include full review_json (default false)."},
                },
            },
        },
    },
    "assistant_get_agent_step": {
        "type": "function",
        "function": {
            "name": "assistant_get_agent_step",
            "description": "Fetch an agent_step row (prompt/response/tool results/state) by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer", "description": "agent_step.id"},
                },
                "required": ["step_id"],
            },
        },
    },
    "assistant_get_agent_command": {
        "type": "function",
        "function": {
            "name": "assistant_get_agent_command",
            "description": "Fetch an agent_command row (payload/result/status) by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command_id": {"type": "integer", "description": "agent_command.id"},
                },
                "required": ["command_id"],
            },
        },
    },
    "assistant_get_agent_run": {
        "type": "function",
        "function": {
            "name": "assistant_get_agent_run",
            "description": "Fetch an agent_run row (state/summary) by run_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "agent_run.run_id"},
                },
                "required": ["run_id"],
            },
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
    "my_projects": {
        "type": "function",
        "function": {
            "name": "my_projects",
            "description": "List projects the current authenticated user can access.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max projects to return."},
                    "query": {"type": "string", "description": "Optional keyword filter."},
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, include only current projects.",
                    },
                },
            },
        },
    },
    "project_files_for_project": {
        "type": "function",
        "function": {
            "name": "project_files_for_project",
            "description": "List uploaded files attached to a project (metadata only; no file bytes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id."},
                    "limit": {"type": "integer", "description": "Max files to return."},
                    "query": {"type": "string", "description": "Optional filename filter."},
                },
                "required": ["project_id"],
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
                        "description": "Optional comment type label (e.g. meeting_note, assistant_note). For client users this is stored as client_note.",
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
            if user.role == UserRole.viewer:
                if name != "create_project_comment" or not bool(
                    getattr(user, "can_write_project_comments", False)
                ):
                    continue
            if user.role == UserRole.client and name != "create_project_comment":
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
