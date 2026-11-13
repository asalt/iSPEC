import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import event, select, update

from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.project_comment_approval import ProjectCommentApprovalSettings
from ispec.assistant.project_comment_workflow import ProjectCommentWorkflow, comment_workflow_tools
from ispec.assistant.service import AssistantReply
from ispec.assistant.tools import openai_tools_for_user, run_tool
from ispec.db.models import AuthUser, AuthUserProject, Project, ProjectAccessMode, ProjectComment, ProjectCommentDraft, ProjectCommentSaveOffer, UserRole


NOTE = "Eight tissue samples arrived; analysis has not started."


@dataclass(kw_only=True)
class Conversation:
    core: object
    assistant: object
    user: object
    session: object
    label: str = "approve_save"
    confidence: float = .9
    checks: list = field(default_factory=list)
    mode: str = "own"
    last_workflow: ProjectCommentWorkflow | None = None

    def classifier(self, **kwargs):
        self.checks.append(kwargs)
        return AssistantReply(content=json.dumps({"label": self.label, "confidence": self.confidence,
                                                   "reason": "fixture"}), provider="stub", model="stub")

    def turn(self, *, message):
        row = SupportMessage(session_pk=self.session.id, role="user", content=message)
        self.assistant.add(row)
        self.assistant.commit()
        self.last_workflow = ProjectCommentWorkflow(
            core_db=self.core, assistant_db=self.assistant, user=self.user,
            session_id=self.session.session_id, turn_id=row.id, user_message=message,
            settings=ProjectCommentApprovalSettings(mode=self.mode),
            generate_reply_fn=lambda **kwargs: pytest.fail("Unexpected main model call"),
            generate_classifier_reply_fn=self.classifier,
        )
        return self.last_workflow

    def present(self, *, text=None):
        workflow = self.last_workflow
        model_reply = text if text is not None else f'Here is the note: "{workflow.pending().comment}" Shall I save it?'
        row = SupportMessage(session_pk=self.session.id, role="assistant", content=model_reply)
        self.assistant.add(row)
        self.assistant.flush()
        receipt = workflow.record_delivery(message=row)
        row.meta_json = json.dumps({"project_comment_presentation": receipt})
        self.assistant.commit()
        return row

    def stage(self, *, comment=NOTE, project_id=57):
        workflow = self.turn(message="Please draft a project note.")
        return workflow.run(name="prepare_project_comment", args={
            "project_id": project_id, "comment": comment,
        })["result"]["draft_id"]


@pytest.fixture
def conversation(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", "0")
    monkeypatch.delenv("ISPEC_ASSISTANT_ALLOWED_WRITE_TOOLS", raising=False)
    db_session.add_all([Project(id=57, prj_AddedBy="test", prj_ProjectTitle="Synthetic project"),
                        Project(id=99, prj_AddedBy="test", prj_ProjectTitle="Other project")])
    db_session.commit()
    user = SimpleNamespace(id=1, username="test", role=UserRole.editor,
                           project_access_mode=ProjectAccessMode.all_projects, can_write_project_comments=True)
    with get_assistant_session(tmp_path / "assistant.db") as adb:
        session = SupportSession(session_id="test-conversation", user_id=1)
        adb.add(session)
        adb.commit()
        yield Conversation(core=db_session, assistant=adb, user=user, session=session)


def save(*, workflow, draft_id, **extra):
    return workflow.run(name="create_project_comment", args={"draft_id": draft_id, "confirm": True, **extra})


def test_stage_does_not_classify_or_write_and_survives_new_request(conversation):
    draft_id = conversation.stage()
    assert conversation.assistant.query(SupportMessage).filter_by(role="assistant").count() == 0
    assert conversation.core.query(ProjectCommentSaveOffer).count() == 0
    followup = conversation.turn(message="What else can you tell me?")
    assert followup.pending().id == draft_id
    assert conversation.checks == []
    assert conversation.core.query(ProjectComment).count() == 0
    assert NOTE in followup.prompt_messages()[-1]["content"]


def test_exact_presentation_and_model_requested_save_share_one_check(conversation):
    workflow = conversation.turn(message="Draft a note; show it first.")
    prepared = workflow.prepare(args={"project_id": 57, "comment": NOTE})
    assert "review_mode" not in prepared["result"]
    assert workflow.presented_text(draft=workflow.pending()) is None
    assert not conversation.checks
    model_text = f'Proposed note: "{NOTE}" Does this look right to save?'
    delivered = conversation.present(text=model_text)
    assert delivered.content == model_text
    followup = conversation.turn(message="That is the version I want in the record.")
    approved = followup.pending()
    assert approved.id == prepared["result"]["draft_id"]
    assert save(workflow=followup, draft_id=approved.id)["ok"]
    assert len(conversation.checks) == len(followup.evaluations) == 1
    classifier_context = json.loads(conversation.checks[0]["messages"][-1]["content"])
    assert classifier_context["presented_draft"] == model_text
    assert classifier_context["latest_user_reply"] == "That is the version I want in the record."


@pytest.mark.parametrize("extra", [{"review_mode": "draft_only"}, {"person_id": 2}, {"confirm": True}])
def test_preparation_accepts_only_project_and_note(conversation, extra):
    workflow = conversation.turn(message="Please prepare a note")
    args = {"project_id": 57, "comment": NOTE, **extra}
    result = workflow.run(name="prepare_project_comment", args=args)
    assert result["error_code"] == "invalid_arguments"
    assert workflow.pending() is None
    assert conversation.core.query(ProjectCommentSaveOffer).count() == 0


def test_removed_offer_tool_cannot_trigger_a_save(conversation):
    draft_id = conversation.stage()
    result = conversation.last_workflow.run(name="offer_project_comment_save", args={"draft_id": draft_id})
    assert result["error_code"] == "unknown_tool"
    assert not conversation.checks
    assert conversation.core.query(ProjectComment).count() == 0


def test_ambiguous_registered_state_never_selects_latest(conversation):
    first = conversation.stage()
    second = conversation.stage(comment="A different project note.", project_id=99)
    conversation.core.execute(update(ProjectCommentDraft).where(ProjectCommentDraft.id == first).values(status="pending"))
    conversation.core.commit()
    conversation.present(text="A different project note.")
    workflow = conversation.turn(message="Save that")
    assert first != second and workflow.pending() is None
    assert not conversation.checks


def test_unknown_id_never_substitutes_pending_draft(conversation):
    draft_id = conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="save")
    assert save(workflow=workflow, draft_id="invented")["error_code"] == "draft_not_found"
    assert workflow.pending().id == draft_id
    assert not conversation.checks


def test_shadow_classifier_context_is_not_sent_as_authorization(conversation):
    draft_id = conversation.stage()
    conversation.present()
    conversation.mode = "shadow"
    workflow = conversation.turn(message="yes")
    assert not save(workflow=workflow, draft_id=draft_id)["ok"]
    assert all("classifier returned" not in message["content"] for message in workflow.prompt_messages())


def test_save_checks_only_on_call_then_writes_exact_draft_once(conversation):
    draft_id = conversation.stage()
    conversation.present()
    followup = conversation.turn(message="Correct, looks good.")
    assert not conversation.checks
    result = save(workflow=followup, draft_id=draft_id)
    assert result["ok"]
    assert len(conversation.checks) == 1
    comment = conversation.core.get(ProjectComment, result["result"]["comment_id"])
    assert comment.com_Comment == NOTE
    assert comment.project_id == 57
    repeated = save(workflow=conversation.turn(message="Did that save?"), draft_id=draft_id)
    assert repeated["result"]["already_saved"]
    assert repeated["result"]["comment_id"] == comment.id
    assert len(conversation.checks) == 1
    assert conversation.core.query(ProjectComment).count() == 1
    payload = json.loads(conversation.checks[0]["messages"][-1]["content"])
    assert payload["pending_comment"]["comment"] == NOTE
    assert payload["latest_user_reply"] == "Correct, looks good."
    assert "lexical_features" not in payload


@pytest.mark.parametrize("label,confidence", [
    ("deny_save", .99), ("draft_only", .99), ("revise_draft", .99),
    ("unrelated_or_unclear", .99), ("requires_explicit_confirmation", .99),
    ("approve_save", .749), ("approve_save", float("nan")), ("approve_save", True),
    ("approve_save", 1.1), ("unknown", .99),
])
def test_checker_blocks_without_lexical_fallback_or_retry(conversation, label, confidence):
    draft_id = conversation.stage()
    conversation.present()
    conversation.label, conversation.confidence = label, confidence
    workflow = conversation.turn(message="yes approve save it now")
    for _ in range(2):
        result = save(workflow=workflow, draft_id=draft_id)
        assert not result["ok"]
        assert result["error_code"] == "approval_not_granted"
        assert "approval" in result
    assert len(conversation.checks) == 1
    assert conversation.core.query(ProjectComment).count() == 0


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_disabled_or_shadow_checker_never_authorizes(conversation, mode):
    draft_id = conversation.stage()
    conversation.present()
    conversation.mode = mode
    assert not save(workflow=conversation.turn(message="Save it"), draft_id=draft_id)["ok"]
    assert len(conversation.checks) == (1 if mode == "shadow" else 0)
    assert conversation.core.query(ProjectComment).count() == 0


def test_checker_exception_is_a_not_saved_result(conversation):
    draft_id = conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="yes")
    def timeout(**kwargs):
        raise TimeoutError("synthetic")
    workflow.generate_classifier_reply_fn = timeout
    result = save(workflow=workflow, draft_id=draft_id)
    assert not result["ok"]
    assert result["approval"]["error"] == "TimeoutError"


def test_same_turn_or_unshown_draft_cannot_save(conversation):
    workflow = conversation.turn(message="Make a note and save it")
    draft_id = workflow.prepare(args={"project_id": 57, "comment": NOTE})["result"]["draft_id"]
    assert save(workflow=workflow, draft_id=draft_id)["error_code"] == "draft_not_presented"
    conversation.present(text="Different wording")
    assert save(workflow=conversation.turn(message="yes"), draft_id=draft_id)["error_code"] == "draft_not_presented"
    assert not conversation.checks


def test_revision_supersedes_old_draft_and_requires_new_presentation(conversation):
    old_id = conversation.stage()
    conversation.present()
    new_id = conversation.stage(comment="Revised wording.")
    workflow = conversation.last_workflow
    assert save(workflow=workflow, draft_id=old_id)["error_code"] == "draft_inactive"
    assert save(workflow=workflow, draft_id=new_id)["error_code"] == "draft_not_presented"
    conversation.present()
    assert save(workflow=conversation.turn(message="yes"), draft_id=new_id)["ok"]


def test_cancel_and_expiry_do_not_classify(conversation):
    draft_id = conversation.stage()
    workflow = conversation.turn(message="Cancel")
    assert workflow.run(name="cancel_project_comment_draft", args={"draft_id": draft_id})["ok"]
    assert save(workflow=workflow, draft_id=draft_id)["error_code"] == "draft_inactive"
    other = conversation.stage()
    conversation.core.execute(update(ProjectCommentDraft).where(ProjectCommentDraft.id == other).values(
        expires_at=datetime.now() - timedelta(days=2)))
    conversation.core.commit()
    assert save(workflow=conversation.turn(message="yes"), draft_id=other)["error_code"] == "draft_inactive"
    assert not conversation.checks


@pytest.mark.parametrize("change", ["session", "actor"])
def test_draft_is_scoped_to_conversation_and_actor(conversation, change):
    draft_id = conversation.stage()
    workflow = conversation.turn(message="yes")
    if change == "session":
        workflow.session_id = "different"
    else:
        workflow.source_actor = "different-slack-user"
    assert save(workflow=workflow, draft_id=draft_id)["error_code"] == "draft_not_found"
    assert not conversation.checks


@pytest.mark.parametrize("extra", [{"person_id": 2}, {"project_id": 99}, {"comment": "Changed"}])
def test_save_cannot_supply_new_fields(conversation, extra):
    draft_id = conversation.stage()
    conversation.present()
    assert save(workflow=conversation.turn(message="yes"), draft_id=draft_id, **extra)["error_code"] == "invalid_arguments"
    assert not conversation.checks


def test_revision_during_inference_prevents_old_write(conversation):
    old = conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="yes")
    def classify_and_revise(**kwargs):
        conversation.stage(comment="A newer draft.")
        return conversation.classifier(**kwargs)
    workflow.generate_classifier_reply_fn = classify_and_revise
    assert save(workflow=workflow, draft_id=old)["error_code"] == "draft_changed"
    assert conversation.core.query(ProjectComment).count() == 0


def test_acl_and_deployment_checks_precede_staging(conversation, monkeypatch):
    workflow = conversation.turn(message="make a note")
    conversation.user.can_write_project_comments = False
    assert not workflow.run(name="prepare_project_comment", args={"project_id": 57, "comment": NOTE})["ok"]
    conversation.user.can_write_project_comments = True
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", "1")
    result = run_tool(name="prepare_project_comment", args={"project_id": 57, "comment": NOTE},
                      core_db=conversation.core, user=conversation.user, project_comment_workflow=workflow)
    assert "read-only" in result["error"]
    assert not conversation.core.scalar(select(ProjectCommentDraft))


def test_chat_tool_schemas_do_not_depend_on_reply_text(conversation):
    schemas = comment_workflow_tools(tools=openai_tools_for_user(conversation.user))
    names = {spec["function"]["name"] for spec in schemas}
    assert {"prepare_project_comment", "create_project_comment", "cancel_project_comment_draft"} <= names
    assert "offer_project_comment_save" not in names
    write = next(spec["function"] for spec in schemas if spec["function"]["name"] == "create_project_comment")
    assert set(write["parameters"]["properties"]) == {"draft_id", "confirm"}
    assert comment_workflow_tools(tools=[]) == []


def test_model_instructions_use_one_preparation_and_model_authored_reply(conversation):
    workflow = conversation.turn(message="Please draft a project-history note.")
    instructions = workflow.prompt_messages()[0]
    assert instructions["role"] == "system"
    assert "offer_project_comment_save" not in instructions["content"]
    assert "review_mode" not in instructions["content"]
    assert "save_approval" not in instructions["content"]
    schemas = comment_workflow_tools(tools=openai_tools_for_user(conversation.user))
    preparation = next(spec["function"] for spec in schemas if spec["function"]["name"] == "prepare_project_comment")
    assert "Include the exact returned comment in your reply" in preparation["description"]
    assert preparation["parameters"]["required"] == ["project_id", "comment"]
    assert workflow.pending() is None
    assert not conversation.checks


def test_database_failure_rolls_back_claim_and_does_not_retry_same_turn(conversation):
    draft_id = conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="yes")
    def fail_insert(*args):
        raise RuntimeError("synthetic insertion failure")
    event.listen(ProjectComment, "before_insert", fail_insert)
    try:
        result = run_tool(name="create_project_comment", args={"draft_id": draft_id, "confirm": True},
                          core_db=conversation.core, user=conversation.user, project_comment_workflow=workflow)
        assert not result["ok"]
    finally:
        event.remove(ProjectComment, "before_insert", fail_insert)
    assert conversation.core.query(ProjectComment).count() == 0
    assert conversation.core.get(ProjectCommentDraft, draft_id).status == "pending"
    assert save(workflow=workflow, draft_id=draft_id)["error_code"] == "write_already_attempted"


def test_actual_success_outcome_is_not_lost_after_an_earlier_bad_call():
    from ispec.assistant.project_comment_approval import project_comment_write_outcome
    outcome = project_comment_write_outcome(tool_calls=[
        {"name": "create_project_comment", "ok": False, "error": "Missing draft"},
        {"name": "create_project_comment", "ok": True,
         "result_preview": json.dumps({"project_id": 57, "comment_id": 2})},
    ], focused_project_id=99, safe_int_fn=lambda value: int(value) if value is not None else None)
    assert outcome["status"] == "succeeded"
    assert outcome["project_id"] == 57
    assert outcome["comment_id"] == 2


def test_tool_catalog_describes_staged_signature(conversation):
    workflow = conversation.turn(message="What tools can you use?")
    result = run_tool(name="assistant_list_tools", args={"query": "project_comment"},
                      core_db=conversation.core, user=conversation.user, project_comment_workflow=workflow)
    assert result["ok"]
    encoded = json.dumps(result["result"])
    assert "draft_id" in encoded
    assert "prepare_project_comment" in encoded


def test_client_access_is_checked_on_prepare_context_and_save(conversation):
    client = AuthUser(username="client", password_hash="test", password_salt="test",
                      password_iterations=1, role=UserRole.client, is_active=True,
                      project_access_mode=ProjectAccessMode.explicit_projects)
    conversation.core.add(client)
    conversation.core.commit()
    grant = AuthUserProject(user_id=client.id, project_id=57)
    conversation.core.add(grant)
    conversation.core.commit()
    conversation.user = client
    workflow = conversation.turn(message="Make a note")
    denied = workflow.run(name="prepare_project_comment", args={"project_id": 99, "comment": NOTE})
    assert denied["error_code"] == "project_unavailable"
    draft_id = conversation.stage()
    conversation.present()
    conversation.core.delete(grant)
    conversation.core.commit()
    workflow = conversation.turn(message="yes")
    assert workflow.pending() is None
    assert save(workflow=workflow, draft_id=draft_id)["error_code"] == "project_unavailable"
    assert not conversation.checks


def test_backstop_late_response_cannot_authorize(conversation, monkeypatch):
    from ispec.assistant import project_comment_workflow
    draft_id = conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="yes")
    clock = iter([0, 7])
    monkeypatch.setattr(project_comment_workflow, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    result = save(workflow=workflow, draft_id=draft_id)
    assert not result["ok"]
    assert result["approval"]["error"] == "classifier_timeout_or_invalid_result"


def test_wording_agreement_is_checked_in_actual_context_on_model_save(conversation):
    workflow = conversation.turn(message="Help me word a note; reviewing wording only.")
    draft_id = workflow.prepare(args={"project_id": 57, "comment": NOTE})["result"]["draft_id"]
    model_text = f'{NOTE} Does the wording look right?'
    conversation.present(text=model_text)
    conversation.label = "draft_only"
    workflow = conversation.turn(message="Correct, looks good.")
    assert not conversation.checks
    result = save(workflow=workflow, draft_id=draft_id)
    assert result["error_code"] == "approval_not_granted"
    assert len(conversation.checks) == 1
    context = json.loads(conversation.checks[0]["messages"][-1]["content"])
    assert context["initial_request"] == "Help me word a note; reviewing wording only."
    assert context["presented_draft"] == model_text
    assert context["latest_user_reply"] == "Correct, looks good."
    assert conversation.core.query(ProjectComment).count() == 0


def test_explicit_save_of_presented_draft_needs_no_extra_offer(conversation):
    draft_id = conversation.stage()
    conversation.present(text=f'{NOTE} Does the wording look right?')
    workflow = conversation.turn(message="Save it now.")
    assert not conversation.checks
    assert save(workflow=workflow, draft_id=draft_id)["ok"]
    assert len(conversation.checks) == 1
    assert conversation.core.query(ProjectCommentDraft).count() == 1
    assert conversation.core.query(ProjectComment).one().com_Comment == NOTE


def test_presentation_binds_hash_message_and_scope_and_revision_invalidates(conversation):
    old = conversation.stage()
    message = conversation.present()
    offer = conversation.core.scalar(select(ProjectCommentSaveOffer))
    draft = conversation.core.get(ProjectCommentDraft, old)
    assert offer.content_hash == draft.content_hash
    assert offer.project_id == draft.project_id and offer.assistant_message_id == message.id
    assert offer.session_id == conversation.session.session_id
    assert offer.actor_key == conversation.last_workflow.actor_key
    conversation.stage(comment="Nine tissue samples arrived; analysis has not started.")
    conversation.core.refresh(offer)
    assert offer.status == "invalidated"
    conversation.present(text="The new draft is not shown in this reply.")
    workflow = conversation.turn(message="Yes")
    assert workflow.presentation(draft=workflow.pending()) is None


@pytest.mark.parametrize("change", ["text", "receipt", "intervening", "expiry", "hash", "actor"])
def test_presentation_validation_fails_closed(conversation, change):
    draft_id = conversation.stage()
    row = conversation.present()
    if change == "text":
        row.content += " A model edit."
    elif change == "receipt":
        row.meta_json = "{}"
    elif change == "intervening":
        conversation.assistant.add(SupportMessage(session_pk=conversation.session.id, role="assistant", content="A different question."))
    else:
        offer = conversation.core.scalar(select(ProjectCommentSaveOffer))
        if change == "expiry":
            offer.expires_at = datetime.now() - timedelta(seconds=1)
        elif change == "hash":
            offer.content_hash = "not-the-draft"
        else:
            offer.actor_key = "another-user"
        conversation.core.commit()
    conversation.assistant.commit()
    workflow = conversation.turn(message="Yes, save it.")
    assert save(workflow=workflow, draft_id=draft_id)["error_code"] == "draft_not_presented"
    assert not conversation.checks


def test_unregistered_revision_is_not_replaced_or_made_approvable(conversation):
    conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="Change eight to nine.")
    assert workflow.pending().comment == NOTE
    model_text = "Nine tissue samples arrived; analysis has not started. Shall I save that?"
    row = conversation.present(text=model_text)
    assert row.content == model_text
    assert json.loads(row.meta_json)["project_comment_presentation"]["status"] == "draft_text_missing"
    followup = conversation.turn(message="Yes")
    assert followup.presented_text(draft=followup.pending()) is None
    assert followup.presentation(draft=followup.pending()) is None


def test_presentation_revoked_during_classifier_rolls_back_claim(conversation):
    draft_id = conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="Yes, save it.")

    def revoke_offer(**kwargs):
        with workflow._session() as db, db.begin():
            db.execute(update(ProjectCommentSaveOffer).values(status="invalidated"))
        return conversation.classifier(**kwargs)

    workflow.generate_classifier_reply_fn = revoke_offer
    assert save(workflow=workflow, draft_id=draft_id)["error_code"] == "presentation_changed"
    assert conversation.core.query(ProjectComment).count() == 0
    assert conversation.core.get(ProjectCommentDraft, draft_id).status == "pending"


def test_orphaned_receipt_cannot_authorize_after_assistant_rollback(conversation):
    draft_id = conversation.stage()
    workflow = conversation.last_workflow
    row = SupportMessage(session_pk=conversation.session.id, role="assistant", content=f'{NOTE} Shall I save it?')
    conversation.assistant.add(row)
    conversation.assistant.flush()
    assert workflow.record_delivery(message=row)["receipt_id"]
    conversation.assistant.rollback()
    assert conversation.core.query(ProjectCommentSaveOffer).count() == 1
    followup = conversation.turn(message="Yes, save it.")
    assert save(workflow=followup, draft_id=draft_id)["error_code"] == "draft_not_presented"
    assert not conversation.checks


def test_later_presentation_tracks_existing_draft_without_repreparation(conversation):
    draft_id = conversation.stage()
    conversation.present(text="I prepared the draft but omitted its text.")
    workflow = conversation.turn(message="Show me the note first.")
    assert workflow.pending().id == draft_id
    message = conversation.present(text=f'{NOTE} Shall I save it?')
    assert message.content == f'{NOTE} Shall I save it?'
    assert json.loads(message.meta_json)["project_comment_presentation"]["status"] == "recorded"
    assert not conversation.checks
    assert conversation.core.query(ProjectCommentDraft).count() == 1
    assert save(workflow=conversation.turn(message="Yes, save it."), draft_id=draft_id)["ok"]
    assert len(conversation.checks) == 1
