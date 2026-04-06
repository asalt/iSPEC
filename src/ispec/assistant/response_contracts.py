from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from jinja2 import Environment, FileSystemLoader

from ispec.assistant.service import AssistantReply
from ispec.prompt import load_bound_prompt, prompt_binding, prompt_observability_context


ResponseContractName = Literal[
    "direct",
    "brief_explainer",
    "structured_explainer",
    "deep_dive",
]
ResponseContractMode = Literal["off", "shadow"]

_CONTRACT_ORDER: tuple[ResponseContractName, ...] = (
    "direct",
    "brief_explainer",
    "structured_explainer",
    "deep_dive",
)


@dataclass(frozen=True)
class ResponseContractSpec:
    name: ResponseContractName
    intent: str
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]
    optional_priority: tuple[str, ...]
    max_optional: int
    template_name: str
    text_sentence_limits: dict[str, int] = field(default_factory=dict)
    text_char_limits: dict[str, int] = field(default_factory=dict)
    min_points: int = 0
    max_points: int = 0
    point_char_limit: int = 160


@dataclass(frozen=True)
class ResponseContractPolicy:
    force_contract: ResponseContractName | None
    contract_cap: ResponseContractName | None
    allowed_contracts: tuple[ResponseContractName, ...]
    warnings: tuple[str, ...] = ()


@dataclass
class ResponseContractResult:
    ok: bool
    enabled: bool = True
    applied: bool = False
    rendered_content: str | None = None
    selected_contract: ResponseContractName | None = None
    policy: ResponseContractPolicy | None = None
    selection: dict[str, Any] | None = None
    raw_slots: dict[str, Any] | None = None
    normalized_slots: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    repair_applied: bool = False
    hard_violations: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    selector_reply_meta: dict[str, Any] | None = None
    fill_reply_meta: dict[str, Any] | None = None
    repair_reply_meta: dict[str, Any] | None = None

    def as_meta(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "applied": self.applied,
            "selected_contract": self.selected_contract,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "repair_applied": self.repair_applied,
            "hard_violations": list(self.hard_violations),
            "skipped_reason": self.skipped_reason,
        }
        if self.policy is not None:
            payload["policy"] = {
                "force": self.policy.force_contract,
                "cap": self.policy.contract_cap,
                "allowed_contracts": list(self.policy.allowed_contracts),
                "warnings": list(self.policy.warnings),
            }
        if self.selection is not None:
            payload["selection"] = self.selection
        if self.raw_slots is not None:
            payload["raw_slots"] = self.raw_slots
        if self.normalized_slots is not None:
            payload["slots"] = self.normalized_slots
        if self.rendered_content is not None:
            payload["candidate_content"] = self.rendered_content
        if self.selector_reply_meta is not None:
            payload["selector"] = self.selector_reply_meta
        if self.fill_reply_meta is not None:
            payload["fill"] = self.fill_reply_meta
        if self.repair_reply_meta is not None:
            payload["repair"] = self.repair_reply_meta
        return payload


CONTRACTS: dict[ResponseContractName, ResponseContractSpec] = {
    "direct": ResponseContractSpec(
        name="direct",
        intent="Short factual reply, correction, or straightforward answer.",
        required_slots=("answer",),
        optional_slots=("caveat", "next_step"),
        optional_priority=("caveat", "next_step"),
        max_optional=1,
        template_name="direct.j2",
        text_sentence_limits={"answer": 2, "caveat": 2, "next_step": 2},
        text_char_limits={"answer": 280, "caveat": 220, "next_step": 220},
    ),
    "brief_explainer": ResponseContractSpec(
        name="brief_explainer",
        intent="Short explanation with one core reason and at most one supporting extra.",
        required_slots=("answer", "reason"),
        optional_slots=("example", "caveat"),
        optional_priority=("example", "caveat"),
        max_optional=1,
        template_name="brief_explainer.j2",
        text_sentence_limits={"answer": 2, "reason": 2, "example": 2, "caveat": 2},
        text_char_limits={"answer": 320, "reason": 320, "example": 260, "caveat": 240},
    ),
    "structured_explainer": ResponseContractSpec(
        name="structured_explainer",
        intent="Bounded explanation with a few organized supporting points.",
        required_slots=("answer", "points"),
        optional_slots=("example", "caveat", "next_step"),
        optional_priority=("example", "caveat", "next_step"),
        max_optional=2,
        template_name="structured_explainer.j2",
        text_sentence_limits={"answer": 2, "example": 2, "caveat": 2, "next_step": 2},
        text_char_limits={"answer": 360, "example": 260, "caveat": 240, "next_step": 220},
        min_points=2,
        max_points=3,
        point_char_limit=180,
    ),
    "deep_dive": ResponseContractSpec(
        name="deep_dive",
        intent="Fuller treatment with bounded key points and optional caveats or next steps.",
        required_slots=("answer", "points"),
        optional_slots=("example", "caveat", "next_step"),
        optional_priority=("example", "caveat", "next_step"),
        max_optional=2,
        template_name="deep_dive.j2",
        text_sentence_limits={"answer": 3, "example": 3, "caveat": 2, "next_step": 2},
        text_char_limits={"answer": 500, "example": 340, "caveat": 260, "next_step": 240},
        min_points=3,
        max_points=5,
        point_char_limit=220,
    ),
}

_TEMPLATE_DIR = Path(__file__).with_name("templates") / "response_contracts"
_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def parse_response_contract_mode(
    raw: str | None,
    *,
    auto_shadow: bool = False,
) -> ResponseContractMode:
    normalized = str(raw or "").strip().lower()
    if not normalized or normalized == "auto":
        return "shadow" if auto_shadow else "off"
    if normalized in {"0", "off", "false", "no", "n"}:
        return "off"
    if normalized in {"1", "true", "yes", "y", "on", "shadow", "safe", "all"}:
        return "shadow"
    return "off"


def response_contracts_mode() -> ResponseContractMode:
    return parse_response_contract_mode(os.getenv("ISPEC_ASSISTANT_ENABLE_RESPONSE_CONTRACTS"))


def response_contracts_enabled() -> bool:
    return response_contracts_mode() == "shadow"


def response_contract_names() -> tuple[ResponseContractName, ...]:
    return _CONTRACT_ORDER


def response_contract_policy_from_meta(meta: dict[str, Any] | None) -> ResponseContractPolicy:
    raw_policy = meta.get("response_contract") if isinstance(meta, dict) else None
    warnings: list[str] = []
    force_contract: ResponseContractName | None = None
    contract_cap: ResponseContractName | None = None

    if isinstance(raw_policy, dict):
        raw_force = str(raw_policy.get("force") or "").strip()
        raw_cap = str(raw_policy.get("cap") or "").strip()
        if raw_force:
            if raw_force in CONTRACTS:
                force_contract = raw_force  # type: ignore[assignment]
            else:
                warnings.append("invalid_force_contract")
        if raw_cap:
            if raw_cap in CONTRACTS:
                contract_cap = raw_cap  # type: ignore[assignment]
            else:
                warnings.append("invalid_cap_contract")

    allowed = list(_CONTRACT_ORDER)
    if contract_cap is not None:
        cap_index = _CONTRACT_ORDER.index(contract_cap)
        allowed = allowed[: cap_index + 1]

    if force_contract is not None:
        if contract_cap is not None and _CONTRACT_ORDER.index(force_contract) > _CONTRACT_ORDER.index(contract_cap):
            warnings.append("force_overrode_cap")
        allowed = [force_contract]

    return ResponseContractPolicy(
        force_contract=force_contract,
        contract_cap=contract_cap,
        allowed_contracts=tuple(allowed),
        warnings=tuple(warnings),
    )


def run_response_contract_pipeline(
    *,
    generate_reply_fn: Callable[..., AssistantReply],
    context_message: str,
    history_messages: list[dict[str, Any]],
    user_message: str,
    tool_result_messages: list[dict[str, Any]],
    draft_answer: str,
    request_meta: dict[str, Any] | None = None,
) -> ResponseContractResult:
    policy = response_contract_policy_from_meta(request_meta)
    result = ResponseContractResult(ok=False, enabled=True, applied=False, policy=policy)
    result.warnings.extend(policy.warnings)

    draft_text = (draft_answer or "").strip()
    if not draft_text:
        result.skipped_reason = "empty_draft"
        result.errors.append("empty_draft")
        return result
    if not policy.allowed_contracts:
        result.skipped_reason = "no_allowed_contracts"
        result.errors.append("no_allowed_contracts")
        return result

    selected_contract: ResponseContractName | None = None
    if policy.force_contract is not None:
        selected_contract = policy.force_contract
        result.selection = {
            "contract": selected_contract,
            "confidence": 1.0,
            "reason": "Forced by request metadata.",
            "source": "forced",
        }
    else:
        selection_prompt = load_bound_prompt(
            _selection_prompt,
            values={
                "contracts_block": "\n".join(
                    f"- {name}: {CONTRACTS[name].intent}" for name in policy.allowed_contracts
                ),
            },
        )
        selector_messages: list[dict[str, Any]] = [
            {"role": "system", "content": selection_prompt.text},
            {"role": "system", "content": context_message},
            *history_messages,
            {"role": "user", "content": user_message},
            *tool_result_messages,
        ]
        selector_reply = generate_reply_fn(
            messages=selector_messages,
            tools=None,
            vllm_extra_body={
                "structured_outputs": {"json": _selection_schema(policy.allowed_contracts)},
                "temperature": 0,
            },
            observability_context=prompt_observability_context(
                selection_prompt,
                extra={"surface": "support_chat", "stage": "response_contract_selection"},
            ),
        )
        result.selector_reply_meta = _reply_meta_payload(selector_reply)
        if not selector_reply.ok:
            result.errors.append("selection_reply_error")
            result.skipped_reason = "selection_reply_error"
            return result
        selection_payload = _parse_json_object(selector_reply.content or "")
        if not isinstance(selection_payload, dict):
            result.errors.append("selection_json_invalid")
            result.skipped_reason = "selection_json_invalid"
            return result
        contract_name = str(selection_payload.get("contract") or "").strip()
        if contract_name not in policy.allowed_contracts:
            result.errors.append("selection_contract_invalid")
            result.selection = selection_payload
            result.skipped_reason = "selection_contract_invalid"
            return result
        selection_payload.setdefault("source", "selector")
        result.selection = selection_payload
        selected_contract = contract_name  # type: ignore[assignment]

    if selected_contract is None:
        result.errors.append("selection_missing")
        result.skipped_reason = "selection_missing"
        return result

    spec = CONTRACTS[selected_contract]
    result.selected_contract = selected_contract

    fill_prompt = load_bound_prompt(
        _fill_prompt,
        values={
            "contract_name": spec.name,
            "contract_intent": spec.intent,
            "required_slots": ", ".join(spec.required_slots),
            "optional_slots": ", ".join(spec.optional_slots) if spec.optional_slots else "none.",
            "max_optional": spec.max_optional,
            "points_rule": (
                f"points must be a JSON array of {spec.min_points}-{spec.max_points} short strings, not paragraphs."
                if "points" in spec.required_slots + spec.optional_slots
                else ""
            ),
            "allowed_slots": ", ".join(spec.required_slots + spec.optional_slots),
        },
    )
    fill_messages: list[dict[str, Any]] = [
        {"role": "system", "content": fill_prompt.text},
        {"role": "system", "content": context_message},
        *history_messages,
        {"role": "user", "content": user_message},
        *tool_result_messages,
        {
            "role": "system",
            "content": "Draft answer to compress and restructure into slots:\n" + draft_text,
        },
    ]
    fill_reply = generate_reply_fn(
        messages=fill_messages,
        tools=None,
        vllm_extra_body={
            "structured_outputs": {"json": _slots_schema(spec)},
            "temperature": 0,
        },
        observability_context=prompt_observability_context(
            fill_prompt,
            extra={"surface": "support_chat", "stage": "response_contract_fill"},
        ),
    )
    result.fill_reply_meta = _reply_meta_payload(fill_reply)
    if not fill_reply.ok:
        result.errors.append("fill_reply_error")
        result.skipped_reason = "fill_reply_error"
        return result

    raw_slots = _parse_json_object(fill_reply.content or "")
    if not isinstance(raw_slots, dict):
        result.errors.append("fill_json_invalid")
        result.skipped_reason = "fill_json_invalid"
        return result
    result.raw_slots = raw_slots

    normalized_slots, warnings, hard_violations = _normalize_slots(spec, raw_slots)
    result.normalized_slots = normalized_slots
    result.warnings.extend(warnings)
    result.hard_violations = list(hard_violations)

    if hard_violations:
        repair_prompt = load_bound_prompt(
            _repair_prompt,
            values={
                "contract_name": spec.name,
                "violations_block": "\n".join(f"- {item}" for item in hard_violations),
            },
        )
        repair_messages: list[dict[str, Any]] = [
            {"role": "system", "content": repair_prompt.text},
            {"role": "system", "content": context_message},
            *history_messages,
            {"role": "user", "content": user_message},
            *tool_result_messages,
            {
                "role": "system",
                "content": "Original draft answer to preserve meaning from:\n" + draft_text,
            },
            {
                "role": "system",
                "content": "Broken slot payload:\n" + json.dumps(raw_slots, ensure_ascii=False, indent=2),
            },
        ]
        repair_reply = generate_reply_fn(
            messages=repair_messages,
            tools=None,
            vllm_extra_body={
                "structured_outputs": {"json": _slots_schema(spec)},
                "temperature": 0,
            },
            observability_context=prompt_observability_context(
                repair_prompt,
                extra={"surface": "support_chat", "stage": "response_contract_repair"},
            ),
        )
        result.repair_applied = True
        result.repair_reply_meta = _reply_meta_payload(repair_reply)
        if not repair_reply.ok:
            result.errors.append("repair_reply_error")
            result.skipped_reason = "repair_reply_error"
            return result

        repaired_slots = _parse_json_object(repair_reply.content or "")
        if not isinstance(repaired_slots, dict):
            result.errors.append("repair_json_invalid")
            result.skipped_reason = "repair_json_invalid"
            return result
        result.raw_slots = repaired_slots
        normalized_slots, repair_warnings, hard_violations = _normalize_slots(spec, repaired_slots)
        result.normalized_slots = normalized_slots
        result.warnings.extend(repair_warnings)
        result.hard_violations = list(hard_violations)
        if hard_violations:
            result.errors.append("repair_failed_contract_validation")
            result.skipped_reason = "repair_failed_contract_validation"
            return result

    rendered = render_response_contract(selected_contract, normalized_slots)
    if not rendered:
        result.errors.append("rendered_content_empty")
        result.skipped_reason = "rendered_content_empty"
        return result

    result.ok = True
    result.applied = True
    result.rendered_content = rendered
    result.skipped_reason = None
    return result


def render_response_contract(contract_name: ResponseContractName, slots: dict[str, Any]) -> str:
    template = _TEMPLATE_ENV.get_template(CONTRACTS[contract_name].template_name)
    rendered = template.render(**slots)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return rendered


@prompt_binding("assistant.response_contract.selection")
def _selection_prompt(allowed_contracts: tuple[ResponseContractName, ...]) -> str:
    contracts_block = "\n".join(f"- {name}: {CONTRACTS[name].intent}" for name in allowed_contracts)
    return load_bound_prompt(
        _selection_prompt,
        values={"contracts_block": contracts_block},
    ).text


@prompt_binding("assistant.response_contract.fill")
def _fill_prompt(spec: ResponseContractSpec) -> str:
    allowed_slots = list(spec.required_slots + spec.optional_slots)
    return load_bound_prompt(
        _fill_prompt,
        values={
            "contract_name": spec.name,
            "contract_intent": spec.intent,
            "required_slots": ", ".join(spec.required_slots),
            "optional_slots": ", ".join(spec.optional_slots) if spec.optional_slots else "none.",
            "max_optional": spec.max_optional,
            "points_rule": (
                f"points must be a JSON array of {spec.min_points}-{spec.max_points} short strings, not paragraphs."
                if "points" in allowed_slots
                else ""
            ),
            "allowed_slots": ", ".join(allowed_slots),
        },
    ).text


@prompt_binding("assistant.response_contract.repair")
def _repair_prompt(spec: ResponseContractSpec, violations: list[str]) -> str:
    violations_block = "\n".join(f"- {item}" for item in violations)
    return load_bound_prompt(
        _repair_prompt,
        values={
            "contract_name": spec.name,
            "violations_block": violations_block,
        },
    ).text

def _selection_schema(allowed_contracts: tuple[ResponseContractName, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "contract": {"type": "string", "enum": list(allowed_contracts)},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["contract", "confidence", "reason"],
        "additionalProperties": False,
    }


def _slots_schema(spec: ResponseContractSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for slot in spec.required_slots + spec.optional_slots:
        if slot == "points":
            properties[slot] = {
                "type": "array",
                "items": {"type": "string"},
            }
        else:
            properties[slot] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(spec.required_slots),
        "additionalProperties": False,
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    for candidate in (cleaned, _extract_braced_json(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_braced_json(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


def _normalize_slots(spec: ResponseContractSpec, raw_slots: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    hard_violations: list[str] = []
    normalized: dict[str, Any] = {}
    allowed_slots = set(spec.required_slots + spec.optional_slots)

    for key in raw_slots:
        if key not in allowed_slots:
            warnings.append(f"dropped_disallowed_slot:{key}")

    for slot in spec.required_slots:
        if slot == "points":
            points = _coerce_points(raw_slots.get(slot), spec=spec, warnings=warnings)
            if len(points) < spec.min_points:
                hard_violations.append(f"missing_required_points:{slot}")
                normalized[slot] = points
            else:
                normalized[slot] = points[: spec.max_points]
                if len(points) > spec.max_points:
                    warnings.append(f"trimmed_points:{slot}")
        else:
            text_value = _normalize_text_slot(
                raw_slots.get(slot),
                sentence_limit=spec.text_sentence_limits.get(slot),
                char_limit=spec.text_char_limits.get(slot),
                warnings=warnings,
                slot_name=slot,
            )
            if not text_value:
                hard_violations.append(f"missing_required_slot:{slot}")
                normalized[slot] = ""
            else:
                normalized[slot] = text_value

    seen = {
        _dedupe_key(str(value))
        for key, value in normalized.items()
        if key != "points" and isinstance(value, str) and value.strip()
    }
    optional_values: dict[str, str | None] = {}
    for slot in spec.optional_slots:
        text_value = _normalize_text_slot(
            raw_slots.get(slot),
            sentence_limit=spec.text_sentence_limits.get(slot),
            char_limit=spec.text_char_limits.get(slot),
            warnings=warnings,
            slot_name=slot,
        )
        if not text_value:
            optional_values[slot] = None
            continue
        dedupe = _dedupe_key(text_value)
        if dedupe and dedupe in seen:
            warnings.append(f"dropped_duplicate_optional:{slot}")
            optional_values[slot] = None
            continue
        seen.add(dedupe)
        optional_values[slot] = text_value

    kept_optional = 0
    for slot in spec.optional_priority:
        value = optional_values.get(slot)
        if not value:
            normalized[slot] = None
            continue
        if kept_optional >= spec.max_optional:
            warnings.append(f"dropped_optional_over_budget:{slot}")
            normalized[slot] = None
            continue
        normalized[slot] = value
        kept_optional += 1

    for slot in spec.optional_slots:
        normalized.setdefault(slot, None)

    return normalized, warnings, hard_violations


def _coerce_points(value: Any, *, spec: ResponseContractSpec, warnings: list[str]) -> list[str]:
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            candidates = []
        elif "\n" in text:
            candidates = [part for part in text.splitlines() if part.strip()]
            warnings.append("coerced_points_from_multiline_text")
        elif ";" in text:
            candidates = [part for part in text.split(";") if part.strip()]
            warnings.append("coerced_points_from_semicolon_text")
        else:
            candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
            warnings.append("coerced_points_from_sentence_text")
    else:
        candidates = []

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in candidates:
        item = _normalize_text_slot(
            raw_item,
            sentence_limit=1,
            char_limit=spec.point_char_limit,
            warnings=warnings,
            slot_name="points",
        )
        if not item:
            continue
        item = re.sub(r"^[\-\*\d\.\)\s]+", "", item).strip()
        if not item:
            continue
        dedupe = _dedupe_key(item)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        items.append(item)
    return items


def _normalize_text_slot(
    value: Any,
    *,
    sentence_limit: int | None,
    char_limit: int | None,
    warnings: list[str],
    slot_name: str,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = str(value).replace("\r\n", "\n").strip()
    if not text:
        return None
    text = re.sub(r"\n{3,}", "\n\n", text)
    if sentence_limit and sentence_limit > 0:
        sentences = _split_sentences(text)
        if len(sentences) > sentence_limit:
            text = " ".join(sentences[:sentence_limit]).strip()
            warnings.append(f"trimmed_sentences:{slot_name}")
    if char_limit and char_limit > 0 and len(text) > char_limit:
        text = text[:char_limit].rstrip()
        text = text.rstrip(" ,;:-")
        warnings.append(f"trimmed_chars:{slot_name}")
    return text or None


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _reply_meta_payload(reply: AssistantReply) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": reply.ok,
        "provider": reply.provider,
        "model": reply.model,
    }
    if reply.meta:
        payload["provider_meta"] = reply.meta
    if reply.error:
        payload["error"] = reply.error
    return payload
