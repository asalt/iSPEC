"""Synthetic unit cases, hand-authored for execution invariants; not training examples."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import event, update
from sqlalchemy.orm import Session

from ispec.assistant.project_comment_cards import TextSaveContext, save_from_text
from ispec.assistant.project_comment_approval import ProjectCommentAuthorization
from ispec.db.models import ProjectComment, ProjectCommentCard, ProjectCommentDraft
from tests.unit.assistant.test_project_comment_cards import conversation  # noqa: F401


def request_save(conversation, *, draft_id):
    return conversation.turn(text="Yes, save this note.").run(
        name="create_project_comment",
        args={"draft_id": draft_id, "confirm": True},
    )


def test_text_save_then_button_returns_same_comment(conversation):
    card_id, draft_id = conversation.prepare()
    saved = request_save(conversation, draft_id=draft_id)
    clicked = conversation.click(card_id=card_id, draft_id=draft_id)
    assert saved["saved"] and clicked.ok and clicked.already_completed
    assert saved["result"]["comment_id"] == clicked.card.comment_id
    assert clicked.card.can_undo
    assert len(conversation.comments()) == 1


def test_button_then_text_returns_saved_state_without_classifier(conversation):
    card_id, draft_id = conversation.prepare()
    clicked = conversation.click(card_id=card_id, draft_id=draft_id)
    workflow = conversation.turn(text="Please do it.")

    def no_classifier(**kwargs):
        pytest.fail("An already saved draft needs no approval inference")

    workflow.generate_classifier_reply_fn = no_classifier
    saved = workflow.run(
        name="create_project_comment", args={"draft_id": draft_id, "confirm": True}
    )
    assert saved["saved"] and saved["result"]["already_saved"]
    assert saved["result"]["comment_id"] == clicked.card.comment_id
    assert len(conversation.comments()) == 1
    state = json.loads(workflow.state_messages()[0]["content"].split("\n", 1)[1])
    assert state[-1]["status"] == "saved"
    assert state[-1]["comment_id"] == clicked.card.comment_id


def test_undo_text_save_preserves_receipt_and_cannot_resave(conversation):
    card_id, draft_id = conversation.prepare()
    saved = request_save(conversation, draft_id=draft_id)
    undone = conversation.click(card_id=card_id, draft_id=draft_id, action="undo")
    assert undone.ok and undone.card.status == "undone" and not undone.card.can_undo
    assert undone.card.comment_id == saved["result"]["comment_id"]
    assert conversation.comments() == []
    assert conversation.click(
        card_id=card_id, draft_id=draft_id, action="undo"
    ).already_completed
    assert not conversation.click(card_id=card_id, draft_id=draft_id).ok
    assert not request_save(conversation, draft_id=draft_id)["saved"]

    with Session(conversation.core.get_bind()) as db:
        draft = db.get(ProjectCommentDraft, draft_id)
        receipt = json.loads(db.get(ProjectCommentCard, card_id).action_receipt_json)
        assert draft.comment_id is None and draft.status == "undone"
    assert receipt["source"] == "text_approval"
    assert receipt["undo"]["backend"] == "local_sqlite"
    assert receipt["undo"]["actor"] == "user:1"
    assert receipt["undo_target"]["comment_id"] == saved["result"]["comment_id"]


def test_wrong_user_cannot_undo(conversation):
    card_id, draft_id = conversation.prepare()
    request_save(conversation, draft_id=draft_id)
    other = SimpleNamespace(**{**vars(conversation.user), "id": 2})
    denied = conversation.click(
        card_id=card_id, draft_id=draft_id, action="undo", user=other
    )
    assert denied.error_code == "card_unavailable" and denied.card is None
    assert len(conversation.comments()) == 1


def test_undo_retention_is_from_draft_creation(conversation):
    card_id, draft_id = conversation.prepare()
    request_save(conversation, draft_id=draft_id)
    conversation.core.execute(
        update(ProjectCommentDraft)
        .where(ProjectCommentDraft.id == draft_id)
        .values(
            expires_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=1)
        )
    )
    conversation.core.commit()
    denied = conversation.click(card_id=card_id, draft_id=draft_id, action="undo")
    assert denied.error_code == "undo_expired" and not denied.card.can_undo
    assert len(conversation.comments()) == 1


def test_undo_refuses_later_edits(conversation):
    card_id, draft_id = conversation.prepare()
    request_save(conversation, draft_id=draft_id)
    comment = conversation.comments()[0]
    comment.com_Comment = "A person's later correction."
    conversation.core.commit()
    denied = conversation.click(card_id=card_id, draft_id=draft_id, action="undo")
    assert denied.error_code == "comment_changed"
    assert conversation.comments()[0].com_Comment == "A person's later correction."


def test_concurrent_text_and_button_use_one_atomic_claim(conversation):
    from ispec.assistant.project_comment_cards import CardOrigin

    card_id, draft_id = conversation.prepare()
    workflow = conversation.turn(text="Save now.")
    draft = workflow.pending()
    check = workflow.approval_check(
        draft=draft, presented=workflow.presentation(draft=draft).text
    )
    context = TextSaveContext(
        session_id=workflow.session_id,
        actor_key=workflow.actor_key,
        turn_id=workflow.turn_id,
        user_message=workflow.user_message,
        origin=CardOrigin.for_request(user=workflow.user),
        authorization=ProjectCommentAuthorization(gate=check.gate, ticket=check.ticket),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        text = pool.submit(
            save_from_text,
            core_db=conversation.core,
            assistant_db=conversation.assistant,
            draft_id=draft_id,
            content_hash=draft.content_hash,
            user=workflow.user,
            context=context,
        )
        click = pool.submit(conversation.click, card_id=card_id, draft_id=draft_id)
        results = [text.result(timeout=10), click.result(timeout=10)]
    assert all(result.ok for result in results)
    assert sum(result.already_completed for result in results) == 1
    assert results[0].card.comment_id == results[1].card.comment_id
    assert len(conversation.comments()) == 1


def test_undo_does_not_touch_legacy_or_enqueue_work(conversation, monkeypatch):
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "1")
    monkeypatch.setattr(
        "ispec.assistant.tools._enqueue_legacy_project_comment_push",
        lambda **kwargs: {"ok": True, "enqueued": True},
    )
    card_id, draft_id = conversation.prepare()
    saved = request_save(conversation, draft_id=draft_id)
    assert saved["saved"] and not saved["result"]["can_undo"]
    denied = conversation.click(card_id=card_id, draft_id=draft_id, action="undo")
    assert denied.error_code == "legacy_undo_unavailable"
    assert len(conversation.comments()) == 1


@pytest.mark.parametrize(
    "status", ["saved", "cancelled", "undone", "superseded", "pending"]
)
def test_actual_chat_receives_current_state_without_write_tools(
    conversation, tmp_path, monkeypatch, status
):
    from ispec.api.routes import support
    from ispec.assistant.service import AssistantReply
    from ispec.schedule.connect import get_schedule_session

    card_id, draft_id = conversation.prepare()
    if status in {"saved", "undone"}:
        conversation.click(card_id=card_id, draft_id=draft_id)
    if status == "undone":
        conversation.click(card_id=card_id, draft_id=draft_id, action="undo")
    if status == "cancelled":
        conversation.click(card_id=card_id, draft_id=draft_id, action="cancel")
    if status == "superseded":
        conversation.prepare(text="A revised project note.")

    for key, value in {
        "PROVIDER": "stub",
        "READ_ONLY": "1",
        "MAX_TOOL_CALLS": "0",
        "TOOL_PROTOCOL": "openai",
        "HISTORY_LIMIT": "10",
        "MAX_PROMPT_TOKENS": "8000",
        "SUMMARY_MAX_CHARS": "0",
        "TURN_DECISION_MODE": "off",
        "ENABLE_SELF_REVIEW": "0",
        "ENABLE_RESPONSE_CONTRACTS": "off",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)

    observed = []

    def generate(*, messages, tools=None, **kwargs):
        assert not tools
        state_message = next(
            message
            for message in messages
            if message["role"] == "system"
            and message["content"].startswith("Current server confirmation-card state:")
        )
        states = json.loads(state_message["content"].split("\n", 1)[1])
        state = next(item for item in states if item["draft_id"] == draft_id)
        observed.append(state)
        assert state["status"] == status
        reply = (
            f"Already saved as #{state['comment_id']}."
            if status == "saved"
            else f"The draft is {status}."
        )
        return AssistantReply(content=reply, provider="stub", model="stub")

    monkeypatch.setattr(support, "generate_reply", generate)
    with get_schedule_session(tmp_path / "state-schedule.db") as schedule_db:
        response = support.chat(
            support.ChatRequest(
                sessionId=conversation.session.session_id,
                message="Please do it.",
                history=[],
            ),
            assistant_db=conversation.assistant,
            core_db=conversation.core,
            schedule_db=schedule_db,
            user=conversation.user,
        )
    assert len(observed) == 1
    assert response.message.startswith(
        "Already saved as #" if status == "saved" else "The draft is"
    )
    assert len(conversation.comments()) == int(status == "saved")


def test_slack_undo_uses_actual_clicker_allowlist(conversation, monkeypatch):
    from ispec.assistant.project_comment_cards import CardOrigin, SlackClick

    conversation.user.id = 0
    conversation.user.username = "api_key"
    conversation.origin = CardOrigin(
        surface="slack",
        requester_id="UOWNER",
        team_id="T",
        channel_id="C",
        thread_ts="1.0",
    )
    card_id, draft_id = conversation.prepare()
    conversation.core.execute(
        update(ProjectCommentCard)
        .where(ProjectCommentCard.id == card_id)
        .values(message_ts="2.0")
    )
    conversation.core.commit()
    assert request_save(conversation, draft_id=draft_id)["saved"]

    click = SlackClick(user_id="UOTHER", team_id="T", channel_id="C", message_ts="2.0")
    assert (
        conversation.click(
            card_id=card_id, draft_id=draft_id, action="undo", slack=click
        ).error_code
        == "card_unavailable"
    )
    monkeypatch.setenv("ISPEC_SLACK_ALLOWED_USER_IDS", "USTAFF")
    allowed_click = click.model_copy(update={"user_id": "USTAFF"})
    undone = conversation.click(
        card_id=card_id, draft_id=draft_id, action="undo", slack=allowed_click
    )
    assert undone.ok and undone.card.status == "undone"
    assert conversation.comments() == []


def test_failed_text_insert_does_not_reuse_ticket_same_turn(conversation):
    card_id, draft_id = conversation.prepare()
    workflow = conversation.turn(text="Save it.")

    def fail_insert(mapper, connection, target):
        raise RuntimeError("Synthetic insert failure")

    event.listen(ProjectComment, "before_insert", fail_insert)
    try:
        with pytest.raises(RuntimeError, match="Synthetic insert failure"):
            workflow.run(
                name="create_project_comment",
                args={"draft_id": draft_id, "confirm": True},
            )
    finally:
        event.remove(ProjectComment, "before_insert", fail_insert)

    retry = workflow.run(
        name="create_project_comment", args={"draft_id": draft_id, "confirm": True}
    )
    assert not retry["saved"] and "already used" in retry["error"]
    assert conversation.comments() == []
    assert conversation.click(card_id=card_id, draft_id=draft_id).ok
