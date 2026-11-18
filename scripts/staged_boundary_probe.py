"""Read-only, isolated diagnostics of saved staged-comment inference boundaries.

Uses iSPEC's inference adapter, not the controller or a database executor. Every
request and response is retained. Scope decisions are measurements, never tickets.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import requests
from pydantic import BaseModel, ConfigDict, Field

from ispec.assistant.service import generate_reply
from scripts.staged_prompt_trace import WireCapture, WireRequest, WireResponse


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/behavioral/data"
PREPARE = "prepare_project_comment"


class PendingNote(BaseModel):
    draft_id: str
    project_id: int
    comment: str
    status: str
    expires_at: str


class ApprovalContext(BaseModel):
    initial_request: str
    presented_draft: str
    pending_comment: PendingNote
    latest_user_reply: str


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    confidence: float = Field(ge=0, le=1)
    reason: str


class Preparation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: int = Field(gt=0)
    comment: str = Field(min_length=1, max_length=20000)


@dataclass(frozen=True, kw_only=True)
class ApprovalCase:
    name: str
    context: ApprovalContext
    expected_scope: str
    source: str


@dataclass(frozen=True, kw_only=True)
class PreparationCase:
    name: str
    user: str
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Probe:
    case: str
    arm: str
    request: WireRequest
    expected: str
    source: str


def first_tools(*, path: Path) -> WireRequest:
    return next(call.request for call in WireCapture.read(path=path).calls if call.request.tools)


def approval_input(*, path: Path) -> WireRequest:
    for call in WireCapture.read(path=path).calls:
        for message in call.request.messages:
            if message.role != "user":
                continue
            try:
                ApprovalContext.model_validate_json(message.content)
            except ValueError:
                continue
            return call.request
    raise ValueError(f"No staged approval input in {path}")


def approval_cases(*, evidence: Path) -> tuple[WireRequest, list[ApprovalCase]]:
    cases = []
    for suffix, scope in (("wordi", "wording"), ("expli", "saving"), ("natur", "saving")):
        path = evidence / "tulu/pytest" / f"test_staged_conversation_{suffix}0/wire.json"
        request = approval_input(path=path)
        context = ApprovalContext.model_validate_json(request.latest_user_text)
        cases.append(ApprovalCase(name="retained_" + suffix, context=context,
                                  expected_scope=scope, source=str(path)))
    baseline = approval_input(path=evidence / "tulu/pytest/test_staged_conversation_wordi0/wire.json")
    wording = cases[0].context
    variants = (
        ("fresh_wording", "The phrasing is right; keep it as a draft for now.", "wording"),
        ("fresh_override", "The wording is settled. Go ahead and add it to the project history now.", "saving"),
        ("fresh_revision", "That is nearly right; please change eight to nine before we do anything else.", "neither"),
        ("fresh_cancel", "Actually, drop this note entirely.", "neither"),
        ("fresh_condition", "Yes, once the sample count has been checked with the sender.", "neither"),
        ("fresh_ambiguous", "I will check with her.", "unclear"),
    )
    for name, reply, scope in variants:
        context = ApprovalContext(initial_request=wording.initial_request,
                                  presented_draft=wording.presented_draft,
                                  pending_comment=wording.pending_comment, latest_user_reply=reply)
        cases.append(ApprovalCase(name=name, context=context, expected_scope=scope, source="fresh_before_run"))
    saving = cases[1].context
    # Same reply, but an explicit save question: tests the scope of agreement.
    context = ApprovalContext(initial_request=saving.initial_request,
                              presented_draft=saving.presented_draft + "\nShall I save this exact note?",
                              pending_comment=saving.pending_comment, latest_user_reply="Correct, looks good.")
    cases.append(ApprovalCase(name="fresh_save_question", context=context, expected_scope="saving",
                              source="fresh_before_run"))
    return baseline, cases


def approval_probes(*, evidence: Path) -> list[Probe]:
    baseline, cases = approval_cases(evidence=evidence)
    probes = []
    for case in cases:
        for arm in ("existing", "scope"):
            payload = baseline.model_dump(exclude_unset=True)
            schema = payload["structured_outputs"]["json"]
            prompt = baseline.messages[0].content
            if arm == "scope":
                prompt = (DATA / "staged_boundary_scope.txt").read_text()
                schema["properties"]["label"]["enum"] = ["saving", "wording", "neither", "unclear"]
            payload["messages"] = [{"role": "system", "content": prompt},
                                   {"role": "user", "content": case.context.model_dump_json()}]
            probes.append(Probe(case=case.name, arm=arm, request=WireRequest.model_validate(payload),
                                expected=case.expected_scope, source=case.source))
    return probes


def preparation_cases(*, request: WireRequest) -> list[PreparationCase]:
    return [
        PreparationCase(name="retained_explicit", user=request.latest_user_text,
                        required=("eleven", "saliva", "pending")),
        PreparationCase(name="fresh_simple", user="Please draft a note for project 57: four biopsy samples arrived this morning; extraction has not begun. Let me see it first.",
                        required=("four", "biopsy", "not")),
        PreparationCase(name="fresh_mixed", user="For project 57, the phospho report was shared yesterday and the profile report is still pending. Also, our printer is jammed again. Draft a project note for my review.",
                        required=("phospho", "yesterday", "pending"), forbidden=("printer", "jammed")),
    ]


def preparation_probes(*, evidence: Path) -> tuple[list[Probe], list[PreparationCase]]:
    source = evidence / "olmo-instruct/pytest/test_staged_conversation_expli0/wire.json"
    baseline = first_tools(path=source)
    base, context, workflow, user = baseline.messages
    minimal = {"role": "system", "content": (DATA / "staged_boundary_minimal.txt").read_text()}
    one_tool = [tool.model_dump(exclude_unset=True) for tool in baseline.tools if tool.function.name == PREPARE]
    all_tools = [tool.model_dump(exclude_unset=True) for tool in baseline.tools]
    arms = (
        ("minimal", [minimal], one_tool, "auto"),
        ("add_workflow", [minimal, workflow.model_dump()], one_tool, "auto"),
        ("add_context", [minimal, context.model_dump(), workflow.model_dump()], one_tool, "auto"),
        ("add_tools", [minimal, context.model_dump(), workflow.model_dump()], all_tools, "auto"),
        ("full_captured", [base.model_dump(), context.model_dump(), workflow.model_dump()], all_tools, "auto"),
        ("forced_diagnostic", [base.model_dump(), context.model_dump(), workflow.model_dump()], all_tools,
         {"type": "function", "function": {"name": PREPARE}}),
    )
    cases = preparation_cases(request=baseline)
    probes = []
    for case in cases:
        for arm, prefix, tools, choice in arms:
            payload = baseline.model_dump(exclude_unset=True)
            payload.update(messages=[*prefix, {"role": "user", "content": case.user}], tools=tools, tool_choice=choice)
            probes.append(Probe(case=case.name, arm=arm, request=WireRequest.model_validate(payload),
                                expected="prepare", source=str(source)))
    return probes, cases


@contextmanager
def capture_loopback(*, base_url: str, calls: list[dict]):
    original = requests.Session.request

    def request(session, method, url, **kwargs):
        if url != base_url + "/v1/chat/completions":
            raise RuntimeError("Probe network boundary rejects " + url)
        kwargs["allow_redirects"] = False
        started = time.monotonic()
        record = {"path": "/v1/chat/completions", "request": kwargs.get("json")}
        calls.append(record)
        try:
            response = original(session, method, url, **kwargs)
            record.update(status=response.status_code, response=response.json())
            return response
        finally:
            record["elapsed_ms"] = round((time.monotonic() - started) * 1000)

    with patch.object(requests.Session, "request", request):
        yield


def evaluate(*, probe: Probe, response: WireResponse, elapsed_ms: int,
             cases: list[PreparationCase]) -> dict:
    message = response.choices[0].message
    if probe.expected != "prepare":
        decision = Decision.model_validate_json(message.content)
        labels = probe.request.model_extra["structured_outputs"]["json"]["properties"]["label"]["enum"]
        if decision.label not in labels:
            raise ValueError("Classifier label outside requested schema")
        would_allow = decision.label in {"approve_save", "saving"} and decision.confidence >= .75
        return {"decision": decision.model_dump(), "hypothetical_allow": would_allow,
                "expected_allow": probe.expected == "saving", "allow_correct": would_allow == (probe.expected == "saving"),
                "scope_correct": decision.label == probe.expected if probe.arm == "scope" else None,
                "within_live_timeout": elapsed_ms <= 6000}
    names = [call.function.name for call in message.tool_calls]
    prepared = [call for call in message.tool_calls if call.function.name == PREPARE]
    result = {"tools": names, "content": message.content, "selected_prepare": len(prepared) == 1,
              "arguments_valid": False, "grounding_checks": False}
    if len(prepared) == 1:
        arguments = prepared[0].function.arguments
        draft = Preparation.model_validate_json(arguments) if isinstance(arguments, str) else Preparation.model_validate(arguments)
        case = next(case for case in cases if case.name == probe.case)
        result.update(arguments=draft.model_dump(), arguments_valid=draft.project_id == 57,
                      grounding_checks=all(fact in draft.comment.casefold() for fact in case.required)
                      and not any(fact in draft.comment.casefold() for fact in case.forbidden))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tulu", "olmo-instruct"), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    cases = []
    if args.model == "tulu":
        probes = approval_probes(evidence=args.evidence)
    else:
        probes, cases = preparation_probes(evidence=args.evidence)
    summary = []
    for probe in probes:
        calls = []
        folder = args.output / (probe.case + "--" + probe.arm)
        folder.mkdir()
        payload = probe.request.model_dump(exclude_unset=True)
        messages = payload.pop("messages")
        tools = payload.pop("tools", None)
        choice = payload.pop("tool_choice", None)
        payload.pop("model")
        with capture_loopback(base_url=os.environ["ISPEC_VLLM_URL"], calls=calls):
            reply = generate_reply(messages=messages, tools=tools, tool_choice=choice,
                                   vllm_extra_body=payload, timeout_seconds=45)
        result = {"case": probe.case, "arm": probe.arm, "expected": probe.expected, "source": probe.source,
                  "adapter_ok": reply.ok, "adapter_error": reply.error, "calls": len(calls),
                  "elapsed_ms": sum(call["elapsed_ms"] for call in calls), "executed_tools": False}
        try:
            result.update(evaluate(probe=probe, response=WireResponse.model_validate(calls[-1]["response"]),
                                   elapsed_ms=result["elapsed_ms"], cases=cases))
        except (ValueError, KeyError, IndexError) as error:
            result["evaluation_error"] = str(error)
        (folder / "wire.json").write_text(json.dumps({"calls": calls}, indent=2) + "\n")
        from scripts.wire_transcript import render
        render(folder / "wire.json")
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        summary.append(result)
        (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
