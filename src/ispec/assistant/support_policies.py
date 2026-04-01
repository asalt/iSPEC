from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from ispec.assistant.context import extract_project_ids


_COUNT_PROJECTS_RE = re.compile(
    r"\bhow\s+many\s+(?:(?:total|all|overall)\s+)?(?:projects?|prjs?|projs?)\b"
    r"|\bnumber\s+of\s+(?:(?:total|all|overall)\s+)?(?:projects?|prjs?|projs?)\b"
    r"|\bcount\s+(?:(?:total|all|overall)\s+)?(?:projects?|prjs?|projs?)\b",
    re.IGNORECASE,
)
_COUNT_CURRENT_PROJECTS_RE = re.compile(
    r"\b(current|active|ongoing)\s+(?:projects?|prjs?|projs?)\b"
    r"|\b(?:projects?|prjs?|projs?)\s+(current|active|ongoing)\b",
    re.IGNORECASE,
)
_LIST_MY_PROJECTS_RE = re.compile(
    r"\bmy\s+(?:projects?|prjs?|projs?)\b"
    r"|\b(?:projects?|prjs?|projs?)\s+(?:can\s+i|do\s+i)\s+(?:view|see|access)\b"
    r"|\b(?:what|which|show|list)\s+(?:projects?|prjs?|projs?)\b",
    re.IGNORECASE,
)
_PROJECT_EXISTENCE_LOOKUP_RE = re.compile(
    r"\b(?:do|does)\s+(?:we\s+)?have\s+(?:project|prj|proj)\s+#?\d+\b"
    r"|\bdoes\s+(?:project|prj|proj)\s+#?\d+\s+exist\b"
    r"|\bis\s+(?:project|prj|proj)\s+#?\d+\s+(?:available|present|there)\b"
    r"|\b(?:project|prj|proj)\s+#?\d+\s+(?:exists?|available|present)\b",
    re.IGNORECASE,
)
_FILE_ROUTER_HINT_RE = re.compile(
    r"\b("
    r"files?|results?|directory|directories|folder|folders|plots?|pca|biplot|"
    r"cluster(?:plot)?s?|heatmap|volcano|pdfs?|images?|removed|filtered"
    r")\b",
    re.IGNORECASE,
)
_PROJECT_ROUTER_HINT_RE = re.compile(
    r"\b("
    r"meeting|prepare|prep|important|focus|understand|background|question|goal|summary|"
    r"sample(?:s)?|geno(?:type)?s?|genders?|mmps?|proteins?|analysis|results?"
    r")\b",
    re.IGNORECASE,
)
_TMUX_ROUTER_HINT_RE = re.compile(
    r"\b("
    r"tmux|pane(?:s)?|window(?:s)?|session\s+group|codex"
    r")\b",
    re.IGNORECASE,
)
_TMUX_SESSION_NAME_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]{0,63})\s+tmux\s+(?:session|session\s+group)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SupportPolicyContext:
    message: str
    focused_project_id: int | None = None


@dataclass(frozen=True)
class SupportToolPolicySelection:
    rule_name: str
    tool_name: str
    args: dict[str, Any]
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class SupportToolPolicyRule:
    name: str
    matches: Callable[[SupportPolicyContext], bool]
    tool_name: Callable[[SupportPolicyContext], str]
    build_args: Callable[[SupportPolicyContext], dict[str, Any]]
    build_messages: Callable[[SupportPolicyContext], list[dict[str, str]]]


@dataclass(frozen=True)
class SupportGroupHintRule:
    name: str
    collect_groups: Callable[[SupportPolicyContext], set[str]]


def _project_ids(ctx: SupportPolicyContext) -> list[int]:
    return extract_project_ids(ctx.message or "")


def _tmux_session_args(ctx: SupportPolicyContext) -> dict[str, Any]:
    match = _TMUX_SESSION_NAME_RE.search(ctx.message or "")
    if not match:
        return {}
    session_name = str(match.group(1) or "").strip()
    return {"session_name": session_name} if session_name else {}


def _project_existence_messages(_ctx: SupportPolicyContext) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "The user is asking for project existence/availability. "
                "After checking the project, answer directly whether it exists. "
                "Keep the answer concise. Do not draft a comment, do not claim anything was saved, "
                "and do not summarize project details or previous comments unless the user asks."
            ),
        }
    ]


_TOOL_POLICY_RULES: tuple[SupportToolPolicyRule, ...] = (
    SupportToolPolicyRule(
        name="count_current_projects",
        matches=lambda ctx: bool(_COUNT_PROJECTS_RE.search(ctx.message or ""))
        and bool(_COUNT_CURRENT_PROJECTS_RE.search(ctx.message or "")),
        tool_name=lambda _ctx: "count_current_projects",
        build_args=lambda _ctx: {},
        build_messages=lambda _ctx: [],
    ),
    SupportToolPolicyRule(
        name="count_all_projects",
        matches=lambda ctx: bool(_COUNT_PROJECTS_RE.search(ctx.message or ""))
        and not bool(_COUNT_CURRENT_PROJECTS_RE.search(ctx.message or "")),
        tool_name=lambda _ctx: "count_all_projects",
        build_args=lambda _ctx: {},
        build_messages=lambda _ctx: [],
    ),
    SupportToolPolicyRule(
        name="my_projects",
        matches=lambda ctx: bool(_LIST_MY_PROJECTS_RE.search(ctx.message or "")),
        tool_name=lambda _ctx: "my_projects",
        build_args=lambda _ctx: {},
        build_messages=lambda _ctx: [],
    ),
    SupportToolPolicyRule(
        name="project_existence_lookup",
        matches=lambda ctx: len(_project_ids(ctx)) == 1 and bool(_PROJECT_EXISTENCE_LOOKUP_RE.search(ctx.message or "")),
        tool_name=lambda _ctx: "get_project",
        build_args=lambda ctx: {"id": int(_project_ids(ctx)[0])} if len(_project_ids(ctx)) == 1 else {},
        build_messages=_project_existence_messages,
    ),
    SupportToolPolicyRule(
        name="tmux_inspection",
        matches=lambda ctx: bool(_TMUX_ROUTER_HINT_RE.search(ctx.message or "")),
        tool_name=lambda _ctx: "assistant_list_tmux_panes",
        build_args=_tmux_session_args,
        build_messages=lambda _ctx: [],
    ),
)


_GROUP_HINT_RULES: tuple[SupportGroupHintRule, ...] = (
    SupportGroupHintRule(
        name="tmux_text_hint",
        collect_groups=lambda ctx: {"tmux"} if _TMUX_ROUTER_HINT_RE.search(ctx.message or "") else set(),
    ),
    SupportGroupHintRule(
        name="project_id_presence",
        collect_groups=lambda ctx: {"projects"} if _project_ids(ctx) else set(),
    ),
    SupportGroupHintRule(
        name="focused_project_project_hint",
        collect_groups=lambda ctx: (
            {"projects"}
            if isinstance(ctx.focused_project_id, int)
            and ctx.focused_project_id > 0
            and _PROJECT_ROUTER_HINT_RE.search(ctx.message or "")
            else set()
        ),
    ),
    SupportGroupHintRule(
        name="focused_project_file_hint",
        collect_groups=lambda ctx: (
            {"projects", "files"}
            if isinstance(ctx.focused_project_id, int)
            and ctx.focused_project_id > 0
            and _FILE_ROUTER_HINT_RE.search(ctx.message or "")
            else set()
        ),
    ),
)


def select_support_tool_policy(*, message: str, focused_project_id: int | None = None) -> SupportToolPolicySelection | None:
    ctx = SupportPolicyContext(message=str(message or ""), focused_project_id=focused_project_id)
    for rule in _TOOL_POLICY_RULES:
        if not rule.matches(ctx):
            continue
        return SupportToolPolicySelection(
            rule_name=rule.name,
            tool_name=rule.tool_name(ctx),
            args=dict(rule.build_args(ctx) or {}),
            messages=list(rule.build_messages(ctx) or []),
        )
    return None


def hinted_support_tool_groups(*, message: str, focused_project_id: int | None = None) -> set[str]:
    ctx = SupportPolicyContext(message=str(message or ""), focused_project_id=focused_project_id)
    hinted: set[str] = set()
    for rule in _GROUP_HINT_RULES:
        hinted.update(rule.collect_groups(ctx))
    return hinted


def _comment_intent_message(text: str, *, confidence: float | None = None, reason: str | None = None) -> list[dict[str, str]]:
    hint = text
    if isinstance(confidence, (int, float)):
        hint = hint.replace("NOTE:", f"NOTE: (confidence={float(confidence):.2f})", 1)
    if isinstance(reason, str) and reason.strip():
        hint += f" Reason: {reason.strip()}"
    return [{"role": "system", "content": hint}]


_WRITE_MODE_HINT_TEXT: dict[str, str] = {
    "draft_only": (
        "NOTE: comment_intent=draft_only. "
        "Treat this as a hint: draft the comment, then ask for confirmation or tweaks before saving."
    ),
    "save_now": (
        "NOTE: comment_intent=save_now. "
        "Treat this as a hint: the user appears to want the note saved now if you have enough content."
    ),
    "confirm_save": (
        "NOTE: comment_intent=confirm_save. "
        "Treat this as a hint: the user appears to be confirming a previously drafted note should now be saved."
    ),
    "none": (
        "NOTE: comment_intent=other. "
        "Treat this as a hint only; do not infer that a project comment write is needed."
    ),
}


def comment_intent_messages_for_write_mode(
    *,
    write_mode: str | None,
    missing_comment_text: bool = False,
    confidence: float | None = None,
    reason: str | None = None,
) -> list[dict[str, str]]:
    normalized = str(write_mode or "").strip() or "none"
    if normalized == "save_now" and missing_comment_text:
        return _comment_intent_message(
            "NOTE: The user appears to want a project comment saved, but the actual comment text is still missing. "
            "Do not save yet; ask for the comment text first.",
            confidence=confidence,
            reason=reason,
        )
    text = _WRITE_MODE_HINT_TEXT.get(normalized)
    if not text:
        return []
    return _comment_intent_message(text, confidence=confidence, reason=reason)

