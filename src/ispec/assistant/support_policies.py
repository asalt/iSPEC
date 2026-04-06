from __future__ import annotations

import json
from dataclasses import dataclass
import re
from typing import Any, Callable

from ispec.assistant.classifier_service import generate_classifier_reply
from ispec.prompt import load_bound_prompt, prompt_binding, prompt_observability_context
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
_TMUX_LISTISH_RE = re.compile(
    r"\b(list|which|show)\b.*\b(panes?|windows?|sessions?)\b"
    r"|\bwhat\s+(?:panes?|windows?|sessions?)\b"
    r"|\btmux\s+ls\b",
    re.IGNORECASE,
)
_TMUX_RAW_CAPTURE_RE = re.compile(
    r"\b(raw|verbatim|exact|full|entire|complete)\b.*\b(output|text|transcript|capture|pane|logs?|traceback|stack\s+trace)\b"
    r"|\b(show|paste|quote)\b.*\b(raw|exact|full|verbatim)\b"
    r"|\b(stdout|stderr|traceback|stack\s+trace|logs?)\b",
    re.IGNORECASE,
)
_TMUX_PANE_NUMBER_RE = re.compile(r"\bpane\s+(\d{1,4})\b", re.IGNORECASE)
_TMUX_CLASSIFIER_MIN_CONFIDENCE = 0.6
_TMUX_CLASSIFIER_MAX_CANDIDATES = 8
_TMUX_STOPWORDS = {
    "a",
    "all",
    "for",
    "go",
    "hello",
    "hows",
    "how",
    "in",
    "is",
    "me",
    "on",
    "pane",
    "panes",
    "please",
    "session",
    "sessions",
    "show",
    "tell",
    "the",
    "this",
    "tmux",
    "what",
    "whats",
    "which",
    "window",
    "windows",
    "with",
}


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
    force_tool_choice: bool = False
    override_tool_args: bool = False
    meta: dict[str, Any] | None = None


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


def _tmux_policy_panes() -> list[dict[str, Any]]:
    try:
        from ispec.assistant.tools import _tmux_list_allowed_panes

        rows = _tmux_list_allowed_panes()
    except Exception:
        rows = []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _tmux_unique_strings(values: list[Any]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                text = str(nested or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    items.append(text)
            continue
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            items.append(text)
    return items


def _tmux_session_aliases(row: dict[str, Any]) -> list[str]:
    return _tmux_unique_strings([row.get("session_group"), row.get("session"), row.get("session_names")])


def _tmux_word_alias_match(message: str, alias: str) -> bool:
    alias_text = str(alias or "").strip().lower()
    if not alias_text:
        return False
    message_text = str(message or "").strip().lower()
    if not message_text:
        return False
    if re.fullmatch(r"[a-z0-9_-]+", alias_text):
        pattern = rf"(?<![a-z0-9_-]){re.escape(alias_text)}(?![a-z0-9_-])"
        return re.search(pattern, message_text) is not None
    return alias_text in message_text


def _tmux_message_tokens(message: str, *, ignored: set[str]) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9_/-]+", str(message or "").lower()) if token]
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _TMUX_STOPWORDS or token in ignored or len(token) < 2:
            continue
        if token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _tmux_descriptor_haystack(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("window_name") or ""),
            str(row.get("pane_title") or ""),
            str(row.get("current_command") or ""),
            " ".join(_tmux_unique_strings([row.get("target_aliases")])),
            " ".join(_tmux_unique_strings([row.get("window_aliases")])),
            str(row.get("preferred_alias") or ""),
        ]
    ).lower()


def _tmux_list_session_name(panes: list[dict[str, Any]], *, message: str) -> str | None:
    if not panes:
        return None
    session_groups = {
        str(row.get("session_group") or "").strip()
        for row in panes
        if str(row.get("session_group") or "").strip()
    }
    sessions = {
        str(row.get("session") or "").strip()
        for row in panes
        if str(row.get("session") or "").strip()
    }
    session_names = {
        name
        for row in panes
        for name in _tmux_unique_strings([row.get("session_names")])
        if name
    }
    for alias in sorted(session_groups):
        if _tmux_word_alias_match(message, alias):
            return alias
    if len(session_groups) == 1:
        return next(iter(session_groups))
    for alias in sorted(session_names):
        if _tmux_word_alias_match(message, alias):
            return alias
    if len(sessions) == 1:
        return next(iter(sessions))
    return None


def _tmux_selection_messages(*, mode: str, scope_label: str | None = None) -> list[dict[str, str]]:
    if mode == "capture":
        content = (
            "The tmux resolver already selected the best real allowlisted pane for this request. "
            "Use the capture result directly. By default, summarize the pane's current state in 1-3 concise sentences "
            "or a few short bullets. Prefer structured fields like activity_summary, last_nonempty_line, current_command, "
            "pane_title, and preferred_alias over dumping raw pane content. Quote or paste raw pane text only when the user "
            "explicitly asks for exact output, raw text, logs, traceback, or a transcript. Do not rename the session, do not invent "
            "another pane handle, and do not claim you inspected a different pane."
        )
        return [{"role": "system", "content": content}]
    content = (
        "The tmux resolver did not find one unique pane to inspect. "
        "Use the returned tmux pane list and ask the user to choose from the real handles in that list, "
        "preferably preferred_alias, capture_target, or pane_id. Do not invent session names or pane handles."
    )
    if scope_label:
        content += f" The list is narrowed to {scope_label}."
    return [{"role": "system", "content": content}]


def _tmux_prefers_list(message: str) -> bool:
    message_text = str(message or "")
    if _tmux_prefers_raw_capture(message_text):
        return False
    return bool(_TMUX_LISTISH_RE.search(message_text))


def _tmux_prefers_raw_capture(message: str) -> bool:
    return bool(_TMUX_RAW_CAPTURE_RE.search(str(message or "")))


def _tmux_capture_args(pane: dict[str, Any], *, message: str) -> dict[str, Any]:
    target = str(pane.get("capture_target") or pane.get("pane_id") or pane.get("preferred_alias") or "")
    if _tmux_prefers_raw_capture(message):
        return {"target": target, "lines": 120}
    return {"target": target, "lines": 40}


def _tmux_explicit_pane_matches(message: str, panes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    message_text = str(message or "").strip().lower()
    if not message_text:
        return []
    matches: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    pane_numbers = {int(raw) for raw in _TMUX_PANE_NUMBER_RE.findall(message_text)}
    for row in panes:
        pane_id = str(row.get("pane_id") or "").strip()
        pane_number = row.get("pane_number")
        matched = False
        if pane_id and pane_id.lower() in message_text:
            matched = True
        if not matched and isinstance(pane_number, int) and pane_number in pane_numbers:
            matched = True
        if not matched:
            for alias in _tmux_unique_strings(
                [
                    row.get("capture_target"),
                    row.get("preferred_alias"),
                    row.get("target_aliases"),
                    row.get("window_aliases"),
                ]
            ):
                alias_text = str(alias or "").strip()
                if not alias_text or alias_text.isdigit():
                    continue
                if _tmux_word_alias_match(message_text, alias_text):
                    matched = True
                    break
        if not matched:
            continue
        key = str(row.get("pane_id") or row.get("capture_target") or row.get("preferred_alias") or "")
        if key and key not in seen_keys:
            seen_keys.add(key)
            matches.append(row)
    return matches


def _tmux_descriptor_matches(message: str, panes: list[dict[str, Any]], *, ignored_tokens: set[str]) -> list[dict[str, Any]]:
    tokens = _tmux_message_tokens(message, ignored=ignored_tokens)
    if not tokens:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in panes:
        haystack = _tmux_descriptor_haystack(row)
        score = sum(1 for token in tokens if token in haystack)
        if score > 0:
            scored.append((score, row))
    if not scored:
        return []
    max_score = max(score for score, _row in scored)
    return [row for score, row in scored if score == max_score]


def _tmux_choice_schema(candidate_keys: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_key": {"type": "string", "enum": ["none", *candidate_keys]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["candidate_key", "confidence", "reason"],
    }


@prompt_binding("assistant.tmux_target_choice.classifier")
def _tmux_choice_prompt() -> str:
    return load_bound_prompt(_tmux_choice_prompt).text


def _tmux_classify_candidate_choice(
    *,
    message: str,
    panes: list[dict[str, Any]],
    generate_reply_fn: Callable[..., Any] | None,
) -> tuple[str | None, float, str]:
    if generate_reply_fn is None or not panes or len(panes) > _TMUX_CLASSIFIER_MAX_CANDIDATES:
        return None, 0.0, "classifier_not_used"
    candidate_keys = [f"c{index}" for index in range(1, len(panes) + 1)]
    candidate_payload = []
    for key, row in zip(candidate_keys, panes, strict=False):
        candidate_payload.append(
            {
                "candidate_key": key,
                "preferred_alias": str(row.get("preferred_alias") or ""),
                "capture_target": str(row.get("capture_target") or ""),
                "session": str(row.get("session") or ""),
                "session_group": str(row.get("session_group") or ""),
                "window_name": str(row.get("window_name") or ""),
                "pane_title": str(row.get("pane_title") or ""),
                "current_command": str(row.get("current_command") or ""),
            }
        )
    prompt = load_bound_prompt(_tmux_choice_prompt)
    reply = generate_classifier_reply(
        base_generate_reply_fn=generate_reply_fn,
        messages=[
            {"role": "system", "content": prompt.text},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_message": str(message or ""),
                        "candidates": candidate_payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        vllm_extra_body={"structured_outputs": {"json": _tmux_choice_schema(candidate_keys)}},
        observability_context=prompt_observability_context(
            prompt,
            extra={
                "surface": "support_chat",
                "stage": "tmux_target_resolution",
                "classifier": "tmux_target_choice",
            },
        ),
    )
    try:
        parsed = json.loads(str(reply.content or "").strip())
    except Exception:
        parsed = {}
    candidate_key = str(parsed.get("candidate_key") or "").strip()
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    reason = str(parsed.get("reason") or "").strip()
    if candidate_key not in {"none", *candidate_keys}:
        return None, 0.0, reason or "invalid_candidate_key"
    if candidate_key == "none":
        return None, confidence, reason or "no_clear_candidate"
    return candidate_key, confidence, reason


def _select_tmux_support_tool_policy(
    *,
    ctx: SupportPolicyContext,
    generate_reply_fn: Callable[..., Any] | None = None,
) -> SupportToolPolicySelection | None:
    if not _TMUX_ROUTER_HINT_RE.search(ctx.message or ""):
        return None

    panes = _tmux_policy_panes()
    if not panes:
        return SupportToolPolicySelection(
            rule_name="tmux_list_choices",
            tool_name="assistant_list_tmux_panes",
            args={},
            messages=_tmux_selection_messages(mode="list"),
            force_tool_choice=True,
            override_tool_args=True,
            meta={
                "resolver": "tmux_inventory",
                "strategy": "list_all",
                "candidate_count": 0,
                "selected_target": None,
                "selected_session_name": None,
                "classifier_used": False,
            },
        )

    explicit_matches = _tmux_explicit_pane_matches(ctx.message, panes)
    if len(explicit_matches) == 1:
        selected = explicit_matches[0]
        return SupportToolPolicySelection(
            rule_name="tmux_capture_unique_pane",
            tool_name="assistant_capture_tmux_pane",
            args=_tmux_capture_args(selected, message=ctx.message),
            messages=_tmux_selection_messages(mode="capture"),
            force_tool_choice=True,
            override_tool_args=True,
            meta={
                "resolver": "tmux_inventory",
                "strategy": "explicit_pane_match",
                "candidate_count": 1,
                "candidate_aliases": [str(selected.get("preferred_alias") or selected.get("capture_target") or "")],
                "selected_target": str(selected.get("capture_target") or ""),
                "selected_session_name": _tmux_list_session_name([selected], message=ctx.message),
                "classifier_used": False,
            },
        )

    session_alias_matches = {
        alias.lower(): alias
        for row in panes
        for alias in _tmux_session_aliases(row)
        if _tmux_word_alias_match(ctx.message, alias)
    }
    session_filtered = [
        row
        for row in panes
        if session_alias_matches
        and any(str(alias or "").strip().lower() in session_alias_matches for alias in _tmux_session_aliases(row))
    ]
    candidate_panes = explicit_matches or session_filtered or panes
    ignored_tokens = set(session_alias_matches) | {
        str(row.get("session") or "").strip().lower()
        for row in candidate_panes
        if str(row.get("session") or "").strip()
    }
    descriptor_matches = _tmux_descriptor_matches(ctx.message, candidate_panes, ignored_tokens=ignored_tokens)
    narrowed_panes = descriptor_matches or candidate_panes

    if not _tmux_prefers_list(ctx.message) and len(narrowed_panes) == 1:
        selected = narrowed_panes[0]
        return SupportToolPolicySelection(
            rule_name="tmux_capture_unique_pane",
            tool_name="assistant_capture_tmux_pane",
            args=_tmux_capture_args(selected, message=ctx.message),
            messages=_tmux_selection_messages(mode="capture"),
            force_tool_choice=True,
            override_tool_args=True,
            meta={
                "resolver": "tmux_inventory",
                "strategy": "unique_candidate",
                "candidate_count": 1,
                "candidate_aliases": [str(selected.get("preferred_alias") or selected.get("capture_target") or "")],
                "selected_target": str(selected.get("capture_target") or ""),
                "selected_session_name": _tmux_list_session_name([selected], message=ctx.message),
                "classifier_used": False,
            },
        )

    classifier_used = False
    classifier_reason: str | None = None
    classifier_confidence: float | None = None
    if not _tmux_prefers_list(ctx.message) and 1 < len(narrowed_panes) <= _TMUX_CLASSIFIER_MAX_CANDIDATES:
        candidate_key, classifier_confidence, classifier_reason = _tmux_classify_candidate_choice(
            message=ctx.message,
            panes=narrowed_panes,
            generate_reply_fn=generate_reply_fn,
        )
        classifier_used = classifier_reason != "classifier_not_used"
        if candidate_key is not None and classifier_confidence >= _TMUX_CLASSIFIER_MIN_CONFIDENCE:
            selected = narrowed_panes[int(candidate_key[1:]) - 1]
            return SupportToolPolicySelection(
                rule_name="tmux_capture_classifier_choice",
                tool_name="assistant_capture_tmux_pane",
                args=_tmux_capture_args(selected, message=ctx.message),
                messages=_tmux_selection_messages(mode="capture"),
                force_tool_choice=True,
                override_tool_args=True,
                meta={
                    "resolver": "tmux_inventory",
                    "strategy": "classifier_choice",
                    "candidate_count": len(narrowed_panes),
                    "candidate_aliases": [
                        str(row.get("preferred_alias") or row.get("capture_target") or "")
                        for row in narrowed_panes
                    ],
                    "selected_target": str(selected.get("capture_target") or ""),
                    "selected_session_name": _tmux_list_session_name([selected], message=ctx.message),
                    "classifier_used": True,
                    "classifier_confidence": classifier_confidence,
                    "classifier_reason": classifier_reason,
                },
            )

    list_session_name = _tmux_list_session_name(narrowed_panes, message=ctx.message)
    list_args = {"session_name": list_session_name} if list_session_name else {}
    scope_label = f"session {list_session_name!r}" if list_session_name else None
    return SupportToolPolicySelection(
        rule_name="tmux_list_choices",
        tool_name="assistant_list_tmux_panes",
        args=list_args,
        messages=_tmux_selection_messages(mode="list", scope_label=scope_label),
        force_tool_choice=True,
        override_tool_args=True,
        meta={
            "resolver": "tmux_inventory",
            "strategy": "list_session" if list_session_name else "list_all",
            "candidate_count": len(narrowed_panes),
            "candidate_aliases": [
                str(row.get("preferred_alias") or row.get("capture_target") or "")
                for row in narrowed_panes[:_TMUX_CLASSIFIER_MAX_CANDIDATES]
            ],
            "selected_target": None,
            "selected_session_name": list_session_name,
            "classifier_used": classifier_used,
            "classifier_confidence": classifier_confidence,
            "classifier_reason": classifier_reason,
        },
    )


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
        build_args=lambda _ctx: {},
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


def select_support_tool_policy(
    *,
    message: str,
    focused_project_id: int | None = None,
    generate_reply_fn: Callable[..., Any] | None = None,
) -> SupportToolPolicySelection | None:
    ctx = SupportPolicyContext(message=str(message or ""), focused_project_id=focused_project_id)
    tmux_selection = _select_tmux_support_tool_policy(ctx=ctx, generate_reply_fn=generate_reply_fn)
    if tmux_selection is not None:
        return tmux_selection
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
