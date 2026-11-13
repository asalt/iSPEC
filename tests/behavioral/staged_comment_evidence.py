"""Named snapshots and readable assertions for the staged conversation trial."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class RecordedComment:
    id: int
    project_id: int
    text: str
    person_id: int | None


@dataclass(frozen=True, kw_only=True)
class RecordedDraft:
    id: str
    text: str
    status: str
    comment_id: int | None


@dataclass(frozen=True, kw_only=True)
class RecordedTurn:
    user: str
    assistant: str
    elapsed_ms: int
    metadata: dict[str, Any]
    comments: list[RecordedComment]
    drafts: list[RecordedDraft]

    @property
    def comment_count(self) -> int:
        return len(self.comments)

    @property
    def save_checks(self) -> list[dict[str, Any]]:
        return self.metadata["project_comment_workflow"]["save_checks"]

    @property
    def pending_drafts(self) -> list[RecordedDraft]:
        return [draft for draft in self.drafts if draft.status == "pending"]


@dataclass(frozen=True, kw_only=True)
class TurnExpectation:
    draft_status: str | None = None
    presented: bool = False
    required_facts: tuple[tuple[str, ...], ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    revised: bool = False


@dataclass(frozen=True, kw_only=True)
class SavePathEvidence:
    """Separate model save requests, executor refusals, checks and stored results."""

    turn: int
    prior_draft_presented: bool
    model_save_requests: int
    executor_errors: tuple[str, ...]
    approval_checks: int
    saved_comments_total: int


def save_path_evidence(*, turns: list[RecordedTurn], tool_calls: list[dict[str, Any]]) -> list[SavePathEvidence]:
    """Record the observed stages of each turn's save path for failure diagnosis."""
    evidence = []
    for index, turn in enumerate(turns):
        previous = turns[index - 1] if index else None
        prior_draft_presented = bool(previous and any(draft.text in previous.assistant
                                                      for draft in previous.pending_drafts))
        saves = [call for call in tool_calls if call["turn"] == index and call["name"] == "create_project_comment"]
        errors = tuple(call["result"]["error_code"] for call in saves
                       if "result" in call and "error_code" in call["result"])
        evidence.append(SavePathEvidence(turn=index + 1, prior_draft_presented=prior_draft_presented,
                                        model_save_requests=len(saves),
                                        executor_errors=errors, approval_checks=len(turn.save_checks),
                                        saved_comments_total=turn.comment_count))
    return evidence


def assert_registered_state(*, case_id: str, turn: RecordedTurn, previous: RecordedTurn | None,
                            expected: TurnExpectation) -> None:
    label = f"{case_id}: after {turn.user!r}"
    if expected.draft_status is None:
        return
    matching = [draft for draft in turn.drafts if draft.status == expected.draft_status]
    assert matching, f"{label}: expected a {expected.draft_status} draft, found {turn.drafts}"
    draft = matching[-1]
    if expected.presented:
        assert draft.text in turn.assistant, f"{label}: registered draft was not presented exactly"
    text = draft.text.casefold()
    for options in expected.required_facts:
        assert any(option.casefold() in text for option in options), f"{label}: draft missing {options}: {draft.text}"
    for phrase in expected.forbidden_facts:
        assert phrase.casefold() not in text, f"{label}: draft contains forbidden fact {phrase}: {draft.text}"
    if expected.revised:
        assert previous and previous.pending_drafts, f"{label}: no original pending draft to revise"
        old = previous.pending_drafts[-1]
        assert draft.id != old.id, f"{label}: revision was not registered"
        assert any(row.id == old.id and row.status == "superseded" for row in turn.drafts), label


def response_review(*, turns: list[RecordedTurn]) -> list[dict[str, Any]]:
    """Review leads, not a semantic oracle or a production authorization rule."""
    findings = []
    for index, turn in enumerate(turns):
        presentation = turn.metadata["project_comment_workflow"].get("presentation")
        if presentation:
            findings.append({"turn": index, "kind": "server_presentation", **presentation})
        for name in ("write_claim_guard", "tool_protocol_guard"):
            if name in turn.metadata:
                findings.append({"turn": index, "kind": name, "details": turn.metadata[name]})
    return findings


def assert_comment_counts(*, case_id: str, turns: list[RecordedTurn], expected_counts: list[int],
                          tool_calls: list[dict[str, Any]], evidence_path: str) -> None:
    assert len(turns) == len(expected_counts), f"{case_id}: incomplete conversation; see {evidence_path}"
    mismatches = [f"Turn {number}: expected {expected} saved comments total; observed {turn.comment_count}."
                  for number, (turn, expected) in enumerate(zip(turns, expected_counts, strict=True), start=1)
                  if turn.comment_count != expected]
    if not mismatches:
        return
    lines = [f"Case {case_id}: end-to-end save expectation failed", *mismatches,
             "This is a controller integration result; inspect the stages below before assigning a model failure.",
             "Conversation and registered state (these are distinct):"]
    for index, turn in enumerate(turns):
        lines += [f"Turn {index + 1}", f"  User: {turn.user}", f"  Delivered reply: {turn.assistant}",
                  f"  Registered drafts: {len(turn.drafts)}; saved comments: {turn.comment_count}"]
        for draft in turn.drafts:
            lines += [f"  Draft {draft.id} ({draft.status}): {draft.text}",
                      f"  Exact draft text in this reply: {draft.text in turn.assistant}"]
        calls = [call for call in tool_calls if call["turn"] == index]
        if not calls:
            lines.append("  Executed tools: none")
        for call in calls:
            result = call.get("result") or {}
            lines.append(f"  Tool {call['name']}: ok={result.get('ok')}; error={result.get('error_code')}")
        lines.append(f"  Approval checks: {len(turn.save_checks)}")
    lines.append(f"Full evidence: {evidence_path}")
    raise AssertionError("\n".join(lines))
