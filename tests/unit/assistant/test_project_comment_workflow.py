"""Synthetic model-tool tests for prepare/revise/cancel and classified saves.

Classifier-authorized save tests from 3be57d2 are superseded by the card action
matrix. These tests retain input, scope, context and model-prose invariants.
"""

from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session

from ispec.assistant.models import SupportMessage
from ispec.assistant.tools import run_tool
from ispec.db.models import ProjectCommentDraft, ProjectAccessMode
from tests.unit.assistant.test_project_comment_cards import (
    conversation,
    NOTE,
)  # noqa: F401


@pytest.mark.parametrize(
    "extra", [{"review_mode": "draft_only"}, {"person_id": 2}, {"confirm": True}]
)
def test_preparation_accepts_only_project_and_note(conversation, extra):
    result = conversation.turn().run(
        name="prepare_project_comment",
        args={"project_id": 57, "comment": NOTE, **extra},
    )
    assert result["error_code"] == "invalid_arguments"
    assert conversation.comments() == []


@pytest.mark.parametrize(
    "project_id,comment", [(True, NOTE), (-1, NOTE), (57, ""), (57, "x" * 20001)]
)
def test_invalid_preparation(conversation, project_id, comment):
    result = conversation.turn().run(
        name="prepare_project_comment",
        args={"project_id": project_id, "comment": comment},
    )
    assert result["error_code"] == "invalid_arguments"


@pytest.mark.parametrize(
    "name",
    [
        "show_project_comment_card",
        "create_project_comment",
        "cancel_project_comment_draft",
    ],
)
def test_unknown_id_never_substitutes_pending_draft(conversation, name):
    _, draft_id = conversation.prepare()
    workflow = conversation.turn()
    assert (
        workflow.run(name=name, args={"draft_id": "invented"})["error_code"]
        == "draft_not_found"
    )
    assert workflow.pending().id == draft_id


@pytest.mark.parametrize("change", ["user", "session"])
def test_draft_scope_is_still_enforced(conversation, change):
    _, draft_id = conversation.prepare()
    workflow = conversation.turn()
    if change == "session":
        workflow.session_id = "another"
    else:
        workflow.user = SimpleNamespace(**{**vars(conversation.user), "id": 9})
    assert (
        workflow.run(name="show_project_comment_card", args={"draft_id": draft_id})[
            "error_code"
        ]
        == "draft_not_found"
    )


def test_pending_draft_is_context_not_permission(conversation):
    _, draft_id = conversation.prepare()
    workflow = conversation.turn(text="Looks good")
    assert workflow.pending().id == draft_id
    assert workflow.evaluations == []
    assert NOTE in "\n".join(
        message["content"] for message in workflow.prompt_messages()
    )
    assert conversation.comments() == []


def test_model_revision_registers_new_text_and_cancellation_updates_card(conversation):
    _, first = conversation.prepare()
    workflow = conversation.turn(text="Change eight to ten")
    revision = workflow.run(
        name="prepare_project_comment",
        args={"project_id": 57, "comment": "Ten samples arrived."},
    )
    new_id = revision["result"]["draft_id"]
    with Session(conversation.core.get_bind()) as db:
        assert db.get(ProjectCommentDraft, first).status == "superseded"
        assert db.get(ProjectCommentDraft, new_id).comment == "Ten samples arrived."
    result = workflow.run(
        name="cancel_project_comment_draft", args={"draft_id": new_id}
    )
    assert result["result"]["cancelled"]
    assert workflow.pending() is None
    assert conversation.comments() == []


def test_prose_revision_never_mutates_registered_note(conversation):
    _, draft_id = conversation.prepare()
    workflow = conversation.turn(text="Make that ten")
    message = SupportMessage(
        session_pk=conversation.session.id,
        role="assistant",
        content="Ten samples arrived.",
    )
    conversation.assistant.add(message)
    conversation.assistant.flush()
    workflow.record_delivery(message=message)
    assert message.content == "Ten samples arrived."
    assert workflow.pending().comment == NOTE
    assert workflow.response_cards()[-1].comment == NOTE


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_off_and_shadow_modes_cannot_authorize(conversation, mode):
    _, draft_id = conversation.prepare()
    workflow = conversation.turn(text="Save now")
    from ispec.assistant.project_comment_approval import ProjectCommentApprovalSettings

    workflow.settings = ProjectCommentApprovalSettings(mode=mode)
    result = workflow.run(
        name="create_project_comment", args={"draft_id": draft_id, "confirm": True}
    )
    assert result["saved"] is False
    assert all(
        not check["write_ticket"]["issued"] or check["write_ticket"]["shadow_only"]
        for check in workflow.evaluations
    )
    assert conversation.comments() == []


def test_classifier_exception_keeps_card_pending(conversation):
    _, draft_id = conversation.prepare()
    workflow = conversation.turn(text="save")

    def timeout(**kwargs):
        raise TimeoutError("synthetic")

    workflow.generate_classifier_reply_fn = timeout
    result = workflow.run(
        name="create_project_comment", args={"draft_id": draft_id, "confirm": True}
    )
    assert result["saved"] is False
    assert result["approval"]["error"] == "TimeoutError"
    assert workflow.pending().id == draft_id


def test_acl_and_deployment_controls_precede_tools(conversation, monkeypatch):
    conversation.user.project_access_mode = ProjectAccessMode.explicit_projects
    workflow = conversation.turn()
    assert (
        workflow.run(
            name="prepare_project_comment", args={"project_id": 57, "comment": NOTE}
        )["error_code"]
        == "project_unavailable"
    )
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", "1")
    result = run_tool(
        name="prepare_project_comment",
        args={"project_id": 57, "comment": NOTE},
        core_db=conversation.core,
        user=conversation.user,
        project_comment_workflow=workflow,
    )
    assert not result["ok"]


def test_low_level_chat_call_without_workflow_cannot_bypass_cards(conversation):
    result = run_tool(
        name="create_project_comment",
        args={"project_id": 57, "comment": NOTE, "confirm": True},
        core_db=conversation.core,
        user=conversation.user,
        support_session_id=conversation.session.session_id,
    )
    assert not result["ok"] and conversation.comments() == []
