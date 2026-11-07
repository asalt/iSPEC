"""Real model-directed staging and saving, exclusively in synthetic databases.

Opt-in loopback inference. No mocked classifier, forced assistant draft, or
injected approval. The restricted project-tool catalog bounds this experiment;
it does not represent the entire production tool menu or Slack transport.
"""

import copy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import pytest

from ispec.api.routes import support
from ispec.api.security import _ApiKeyServiceUser
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.project_comment_workflow import COMMENT_WORKFLOW_TOOLS
from ispec.db.connect import get_session
from ispec.db.models import Project, ProjectComment, ProjectCommentDraft
from ispec.schedule.connect import get_schedule_session
from tests.behavioral.test_model_runtime_followup_live import isolated_runtime  # noqa: F401
from tests.behavioral.staged_comment_evidence import (
    RecordedComment, RecordedDraft, RecordedTurn, assert_comment_counts,
    TurnExpectation, assert_registered_state, response_review,
)


pytestmark = [
    pytest.mark.behavioral,
    pytest.mark.skipif(os.getenv("ISPEC_RUN_LLM_TESTS") != "1", reason="Explicit isolated inference only"),
]


@dataclass(frozen=True, kw_only=True)
class ConversationCase:
    id: str
    messages: list[str]
    comment_counts: list[int]
    required_facts: list[list[str]]
    forbidden_facts: list[str]
    turn_expectations: list[dict] | None = None


CASES = [ConversationCase(**row) for row in json.loads(
    (Path(__file__).parent / "data/staged_comment_cases.json").read_text()
)["cases"]]
READ_TOOLS = {"get_project", "projects", "search_projects", "latest_project_comments"}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_staged_conversation(isolated_runtime, monkeypatch, tmp_path, case):  # noqa: F811
    store = isolated_runtime
    for key, value in {
        "READ_ONLY": "0", "ALLOWED_WRITE_TOOLS": "create_project_comment",
        "PROJECT_COMMENT_APPROVAL_TIMEOUT_MS": "6000",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)

    original_schemas = support.openai_tools_for_user
    monkeypatch.setattr(support, "openai_tools_for_user", lambda user: [
        spec for spec in original_schemas(user)
        if spec["function"]["name"] in READ_TOOLS | {"create_project_comment"}
    ])
    original_run = support.run_tool
    calls = []
    turns: list[RecordedTurn] = []
    current_turn = 0

    def observed_tool(**kwargs):
        assert kwargs["name"] in READ_TOOLS | COMMENT_WORKFLOW_TOOLS
        assert Path(kwargs["core_db"].bind.url.database) == store.core_db_path
        call = {"turn": current_turn, "name": kwargs["name"], "arguments": copy.deepcopy(kwargs["args"])}
        calls.append(call)
        result = original_run(**kwargs)
        call["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(support, "run_tool", observed_tool)
    user = _ApiKeyServiceUser(can_write_project_comments=True)
    report = {"case": case.id, "model": os.environ["ISPEC_VLLM_MODEL"],
              "production": False, "tool_calls": calls,
              "expected_comment_counts": case.comment_counts}
    try:
        with get_session(store.core_db_path) as core:
            core.add(Project(id=57, prj_AddedBy="fixture", prj_ProjectTitle="Synthetic sample follow-up"))
        for current_turn, message in enumerate(case.messages):
            # Match real request boundaries: persist conversation, close each session.
            with (get_session(store.core_db_path) as core,
                  get_assistant_session(store.assistant_db_path) as adb,
                  get_schedule_session(store.schedule_db_path) as schedule):
                started = time.monotonic()
                result = support.chat(
                    support.ChatRequest(sessionId="staged-" + case.id, message=message, history=[]),
                    assistant_db=adb, core_db=core, schedule_db=schedule, user=user,
                )
                row = adb.get(SupportMessage, result.messageId)
                comments = core.query(ProjectComment).order_by(ProjectComment.id).all()
                drafts = core.query(ProjectCommentDraft).order_by(ProjectCommentDraft.created_at).all()
                turns.append(RecordedTurn(
                    user=message, assistant=result.message,
                    elapsed_ms=round((time.monotonic() - started) * 1000), metadata=json.loads(row.meta_json),
                    comments=[RecordedComment(id=c.id, project_id=c.project_id, text=c.com_Comment,
                                              person_id=c.person_id) for c in comments],
                    drafts=[RecordedDraft(id=d.id, text=d.comment, status=d.status,
                                          comment_id=d.comment_id) for d in drafts],
                ))
    finally:
        report["turns"] = [asdict(turn) for turn in turns]
        report["response_review"] = response_review(turns=turns)
        (tmp_path / "conversation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print("\nSTAGED " + json.dumps({"case": case.id, "turns": [
            {"assistant": turn.assistant, "comments": turn.comment_count, "checks": turn.save_checks}
            for turn in turns
        ]}), flush=True)

    assert_comment_counts(case_id=case.id, turns=turns, expected_counts=case.comment_counts,
                          tool_calls=calls, evidence_path=str(tmp_path / "conversation.json"))
    for index, turn in enumerate(turns):
        save_calls = [call for call in calls if call["turn"] == index and call["name"] == "create_project_comment"]
        if turn.save_checks:
            assert index > 0 and turns[index - 1].pending_drafts, "Classifier ran without registered prior state"
            assert save_calls, "Classifier ran without a model-requested save"
        previous_count = turns[index - 1].comment_count if index else 0
        if case.comment_counts[index] == previous_count:
            assert not save_calls, f"{case.id} turn {index + 1}: unexpected save attempt: {save_calls}"
        forced = turn.metadata["tooling"].get("forced_tool_choice")
        assert forced != {"type": "function", "function": {"name": "create_project_comment"}}
        for comment in turn.comments:
            assert any(draft.text == comment.text and draft.comment_id == comment.id
                       for draft in turn.drafts), "Saved content differs from prepared draft"
        if case.turn_expectations:
            assert_registered_state(case_id=case.id, turn=turn, previous=turns[index - 1] if index else None,
                                    expected=TurnExpectation(**case.turn_expectations[index]))
    if case.id == "cancel":
        assert not any(draft.status == "pending" for draft in turns[-1].drafts)
    if case.comment_counts[-1]:
        note = turns[-1].comments[0].text.casefold()
        for options in case.required_facts:
            assert any(option.casefold() in note for option in options), options
        for phrase in case.forbidden_facts:
            assert phrase.casefold() not in note, phrase
