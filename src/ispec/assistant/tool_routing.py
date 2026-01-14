from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from ispec.assistant.service import AssistantReply


@dataclass(frozen=True)
class ToolGroup:
    name: str
    description: str
    tool_names: tuple[str, ...]


_GROUP_DEFS: tuple[ToolGroup, ...] = (
    ToolGroup(
        name="projects",
        description="Create/list/count/search projects and project comments/metadata.",
        tool_names=(
            "project_counts_snapshot",
            "count_all_projects",
            "count_current_projects",
            "project_status_counts",
            "billing_category_counts",
            "latest_projects",
            "latest_project_comments",
            "search_projects",
            "projects",
            "get_project",
            "create_project_comment",
        ),
    ),
    ToolGroup(
        name="experiments",
        description="List/get experiments and experiment runs (and experiments in a project).",
        tool_names=(
            "experiments_for_project",
            "latest_experiments",
            "get_experiment",
            "latest_experiment_runs",
            "get_experiment_run",
        ),
    ),
    ToolGroup(
        name="omics",
        description="Gene-level E2G QC/quant lookups scoped to a project.",
        tool_names=("e2g_search_genes_in_project", "e2g_gene_in_project"),
    ),
    ToolGroup(
        name="people",
        description="Search/get people records.",
        tool_names=("search_people", "get_person"),
    ),
    ToolGroup(
        name="schedule",
        description="Scheduling slots/requests (requests tools are admin-only).",
        tool_names=("list_schedule_slots", "list_schedule_requests", "get_schedule_request"),
    ),
    ToolGroup(
        name="repo",
        description="Dev-only repo inspection tools (list/search/read files).",
        tool_names=("repo_list_files", "repo_search", "repo_read_file"),
    ),
    ToolGroup(
        name="misc",
        description="Other utilities (assistant DB stats, API schema search, DB file stats, recent activity).",
        tool_names=(
            "search_api",
            "db_file_stats",
            "latest_activity",
            "assistant_stats",
            "assistant_recent_sessions",
            "assistant_get_session_review",
        ),
    ),
)


def tool_groups_for_available_tools(available_tool_names: set[str]) -> list[ToolGroup]:
    """Return tool groups with tool lists restricted to ``available_tool_names``."""

    available = {name.strip() for name in available_tool_names if isinstance(name, str) and name.strip()}
    grouped: list[ToolGroup] = []
    covered: set[str] = set()
    for group in _GROUP_DEFS:
        names = tuple(name for name in group.tool_names if name in available)
        if not names:
            continue
        grouped.append(ToolGroup(name=group.name, description=group.description, tool_names=names))
        covered.update(names)

    leftovers = sorted(available - covered)
    if leftovers:
        grouped.append(
            ToolGroup(
                name="other",
                description="Fallback group for tools not categorized elsewhere.",
                tool_names=tuple(leftovers),
            )
        )

    return grouped


def tool_router_schema(group_names: list[str]) -> dict[str, Any]:
    names = [name.strip() for name in group_names if isinstance(name, str) and name.strip()]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "primary": {"type": "string", "enum": names},
            "secondary": {
                "type": "array",
                "items": {"type": "string", "enum": names},
                "minItems": 0,
                "maxItems": 2,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "clarify": {"type": "boolean"},
        },
        "required": ["primary", "secondary", "confidence", "clarify"],
    }


def _tool_router_system_prompt(groups: list[ToolGroup]) -> str:
    lines = [
        "Select the best tool group(s) for the user request.",
        "Return only a JSON object that matches the provided schema.",
        'If you cannot follow the schema exactly, return: {"groups":["<primary>","<optional-secondary>",...]}',
        "",
        "Groups:",
    ]
    for group in groups:
        lines.append(f"- {group.name}: {group.description}")
    return "\n".join(lines).strip()


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _validate_router_decision(decision: dict[str, Any], *, group_names: set[str]) -> dict[str, Any] | None:
    primary = decision.get("primary")
    secondary: Any = decision.get("secondary")
    confidence: Any = decision.get("confidence")
    clarify: Any = decision.get("clarify")

    if not isinstance(primary, str) or not primary.strip():
        groups = decision.get("groups")
        if isinstance(groups, list) and groups and isinstance(groups[0], str) and groups[0].strip():
            primary = groups[0].strip()
            if secondary is None:
                secondary = [item for item in groups[1:] if isinstance(item, str) and item.strip()]
        else:
            group = decision.get("group")
            if isinstance(group, str) and group.strip():
                primary = group.strip()

    if not isinstance(primary, str) or primary not in group_names:
        return None

    secondary_list: list[str] = []
    if secondary is None:
        secondary_list = []
    elif isinstance(secondary, str):
        secondary_list = [secondary]
    elif isinstance(secondary, list):
        if any((not isinstance(item, str)) for item in secondary):
            return None
        secondary_list = list(secondary)
    else:
        return None

    secondary_clean = [item for item in secondary_list if item in group_names and item != primary]
    if len(secondary_clean) > 2:
        secondary_clean = secondary_clean[:2]
    try:
        confidence_value = float(confidence) if confidence is not None else 0.7
    except Exception:
        confidence_value = 0.7
    if confidence_value < 0:
        confidence_value = 0.0
    if confidence_value > 1:
        confidence_value = 1.0
    clarify_value = bool(False if clarify is None else clarify)
    return {
        "primary": primary,
        "secondary": secondary_clean,
        "confidence": confidence_value,
        "clarify": clarify_value,
    }


def route_tool_groups_vllm(
    *,
    user_message: str,
    groups: list[ToolGroup],
    generate_reply_fn: Callable[..., AssistantReply],
) -> tuple[dict[str, Any] | None, AssistantReply]:
    """Route a user request to a tool group using vLLM guided JSON output."""

    group_names = [group.name for group in groups]
    schema = tool_router_schema(group_names)
    router_messages = [
        {"role": "system", "content": _tool_router_system_prompt(groups)},
        {"role": "user", "content": f'User request: "{user_message}"'},
    ]
    reply = generate_reply_fn(
        messages=router_messages,
        tools=None,
        vllm_extra_body={
            "guided_json": schema,
            "max_tokens": 200,
            "temperature": 0,
        },
    )
    parsed = _parse_json_object(reply.content)
    validated = (
        _validate_router_decision(parsed, group_names=set(group_names)) if isinstance(parsed, dict) else None
    )
    return validated, reply


def tool_names_for_groups(*, groups: list[ToolGroup], primary: str, secondary: list[str]) -> set[str]:
    by_group = {group.name: set(group.tool_names) for group in groups}
    selected: set[str] = set()
    selected.update(by_group.get(primary, set()))
    for name in secondary:
        selected.update(by_group.get(name, set()))
    return selected
