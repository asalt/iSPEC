"""Shared atomic execution for classified text approvals and confirmation cards."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
import os
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ispec.authz import get_project_for_user, uses_explicit_project_access
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.project_comment_undo import CommentUndoTarget, undo_local_comment
from ispec.assistant.project_comment_approval import (
    ProjectCommentAuthorization,
    ProjectCommentGate,
    ProjectCommentWriteTicket,
    stable_hash,
)
from ispec.db.models import (
    AuthUser,
    Person,
    ProjectComment,
    ProjectCommentCard,
    ProjectCommentDraft,
)


class SlackClick(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: str = Field(min_length=1, max_length=80)
    team_id: str = Field(min_length=1, max_length=80)
    channel_id: str = Field(min_length=1, max_length=80)
    message_ts: str = Field(min_length=1, max_length=80)


class CardAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["save", "cancel", "undo"]
    draft_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    slack: SlackClick | None = None


class ConfirmationCard(BaseModel):
    model_config = ConfigDict(frozen=True)
    card_id: str
    draft_id: str
    content_hash: str
    project_id: int
    session_id: str
    comment: str
    status: str
    expires_at: str
    comment_id: int | None
    message_ts: str | None
    can_undo: bool = False


class CardResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    ok: bool
    card: ConfirmationCard | None = None
    error_code: str | None = None
    error: str | None = None
    already_completed: bool = False
    legacy_push_enqueue: dict[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class CardOrigin:
    surface: Literal["web", "slack"]
    requester_id: str
    team_id: str = ""
    channel_id: str = ""
    thread_ts: str = ""

    @classmethod
    def for_request(
        cls, *, user: Any, slack: dict[str, Any] | None = None
    ) -> "CardOrigin":
        if slack is not None and getattr(user, "id", None) == 0:
            if not all(slack.get(k) for k in ("user_id", "team_id", "channel")):
                raise ValueError(
                    "Slack cards require a verified internal receiver and complete conversation scope."
                )
            return cls(
                surface="slack",
                requester_id=str(slack["user_id"]),
                team_id=str(slack["team_id"]),
                channel_id=str(slack["channel"]),
                thread_ts=str(slack.get("thread_ts") or ""),
            )
        return cls(surface="web", requester_id=str(getattr(user, "id", "")))


@dataclass(frozen=True, kw_only=True)
class TextSaveContext:
    """Server-owned conversation identity and classifier authorization."""

    session_id: str
    actor_key: str
    turn_id: int
    user_message: str
    origin: CardOrigin
    authorization: ProjectCommentAuthorization | None


def new_card(*, draft: ProjectCommentDraft, origin: CardOrigin) -> ProjectCommentCard:
    return ProjectCommentCard(
        id=uuid4().hex,
        draft_id=draft.id,
        surface=origin.surface,
        requester_id=origin.requester_id,
        team_id=origin.team_id,
        channel_id=origin.channel_id,
        thread_ts=origin.thread_ts,
    )


def snapshot(
    *, card: ProjectCommentCard, draft: ProjectCommentDraft
) -> ConfirmationCard:
    status = draft.status
    if status == "pending" and draft.expires_at <= datetime.now(UTC).replace(
        tzinfo=None
    ):
        status = "expired"
    receipt = json.loads(card.action_receipt_json or "{}")
    return ConfirmationCard(
        card_id=card.id,
        draft_id=draft.id,
        content_hash=draft.content_hash,
        project_id=draft.project_id,
        session_id=draft.session_id,
        comment=draft.comment,
        status=status,
        expires_at=draft.expires_at.isoformat() + "Z",
        comment_id=draft.comment_id or receipt.get("comment_id"),
        message_ts=card.message_ts,
        can_undo=(
            draft.status == "saved"
            and draft.expires_at > datetime.now(UTC).replace(tzinfo=None)
            and not receipt.get("legacy_push_possible", False)
        ),
    )


def cards_for_conversation(
    *, db: Session, session_id: str, actor_key: str
) -> list[ConfirmationCard]:
    rows = db.execute(
        select(ProjectCommentCard, ProjectCommentDraft)
        .join(
            ProjectCommentDraft,
            ProjectCommentDraft.id == ProjectCommentCard.draft_id,
        )
        .where(
            ProjectCommentDraft.session_id == session_id,
            ProjectCommentDraft.actor_key == actor_key,
            ProjectCommentCard.assistant_message_id.is_not(None),
        )
        .order_by(ProjectCommentDraft.created_at.desc())
        .limit(20)
    ).all()
    return [snapshot(card=card, draft=draft) for card, draft in reversed(rows)]


def _denied(
    *, code: str, message: str, card: ConfirmationCard | None = None
) -> CardResult:
    return CardResult(ok=False, error_code=code, error=message, card=card)


def _clicker_allowed(
    *, card: ProjectCommentCard, user: Any, slack: SlackClick | None
) -> bool:
    if user is None or not getattr(user, "is_active", True):
        return False
    if card.surface == "web":
        return (
            slack is None
            and getattr(user, "id", 0) > 0
            and str(user.id) == card.requester_id
        )
    if slack is None or getattr(user, "id", None) != 0:
        return False
    if (slack.team_id, slack.channel_id, slack.message_ts) != (
        card.team_id,
        card.channel_id,
        card.message_ts,
    ):
        return False
    allowed = {
        item.strip()
        for item in os.getenv("ISPEC_SLACK_ALLOWED_USER_IDS", "").split(",")
        if item.strip()
    }
    return slack.user_id == card.requester_id or slack.user_id in allowed


def _click_authorization(
    *, draft: ProjectCommentDraft, actor: str, card_id: str
) -> ProjectCommentAuthorization:
    """Issue a request-local ticket after an authenticated, bound button action."""
    now = datetime.now(UTC)
    event_id = uuid4().hex
    source = f"card:{card_id}:save:{actor}"
    gate = ProjectCommentGate(
        eligible=True,
        kind="pending_save_confirmation",
        reason="confirmation_card_click",
        project_id=draft.project_id,
        pending_tool="create_project_comment",
        session_id=draft.session_id,
        thread_key=draft.session_id,
        current_turn_id=event_id,
        prior_assistant_message_id=None,
        pending_draft_hash=draft.content_hash,
        source_message_hash=stable_hash(source),
    )
    ticket = ProjectCommentWriteTicket(
        issued=True,
        ticket_id="pcc_" + event_id,
        tool_name="create_project_comment",
        project_id=draft.project_id,
        session_id=draft.session_id,
        thread_key=draft.session_id,
        current_turn_id=event_id,
        pending_draft_hash=draft.content_hash,
        candidate_comment_hash=draft.content_hash,
        issued_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=120)).isoformat(),
        ttl_seconds=120,
        single_use=True,
        shadow_only=False,
    )
    authorization = ProjectCommentAuthorization(gate=gate, ticket=ticket)
    error = authorization.consume(
        project_id=draft.project_id,
        session_id=draft.session_id,
        turn_id=event_id,
        user_message=source,
        comment=draft.comment,
    )
    if error:
        raise ValueError(error)
    return authorization


def _has_committed_presentation(
    *, assistant_db: Session, card: ProjectCommentCard, draft: ProjectCommentDraft
) -> bool:
    """Verify the persisted reply receipt in the separate conversation database."""
    with Session(bind=assistant_db.get_bind()) as db:
        message = db.scalar(
            select(SupportMessage)
            .join(SupportSession, SupportSession.id == SupportMessage.session_pk)
            .where(
                SupportMessage.id == card.assistant_message_id,
                SupportMessage.role == "assistant",
                SupportSession.session_id == draft.session_id,
            )
        )
        if message is None:
            return False
        try:
            receipt = json.loads(message.meta_json or "{}")[
                "project_comment_presentation"
            ]
        except (ValueError, KeyError, TypeError):
            return False
        return (
            isinstance(receipt, dict)
            and receipt.get("status") == "recorded"
            and receipt.get("draft_id") == draft.id
            and receipt.get("content_hash") == draft.content_hash
            and receipt.get("actor_key") == draft.actor_key
        )


def act_on_card(
    *,
    core_db: Session,
    assistant_db: Session,
    card_id: str,
    action: CardAction,
    user: Any,
) -> CardResult:
    """Execute a transport-authenticated Save, Cancel or Undo button action."""
    return _act_on_draft(
        core_db=core_db,
        assistant_db=assistant_db,
        card_id=card_id,
        action=action,
        user=user,
    )


def save_from_text(
    *,
    core_db: Session,
    assistant_db: Session,
    draft_id: str,
    content_hash: str,
    user: Any,
    context: TextSaveContext,
) -> CardResult:
    """Send a model-requested, classified save through the shared atomic executor."""
    with Session(bind=core_db.get_bind()) as db:
        card_id = db.scalar(
            select(ProjectCommentCard.id).where(ProjectCommentCard.draft_id == draft_id)
        )
    if card_id is None:
        return _denied(code="card_unavailable", message="Prepare a new draft first.")
    return _act_on_draft(
        core_db=core_db,
        assistant_db=assistant_db,
        card_id=card_id,
        action=CardAction(action="save", draft_id=draft_id, content_hash=content_hash),
        user=user,
        text_context=context,
    )


def _text_scope_matches(
    *, card: ProjectCommentCard, draft: ProjectCommentDraft, context: TextSaveContext
) -> bool:
    return (
        context.session_id == draft.session_id
        and context.actor_key == draft.actor_key
        and context.origin.surface == card.surface
        and context.origin.requester_id == card.requester_id
        and context.origin.team_id == card.team_id
        and context.origin.channel_id == card.channel_id
        and context.origin.thread_ts == card.thread_ts
    )


def _act_on_draft(
    *,
    core_db: Session,
    assistant_db: Session,
    card_id: str,
    action: CardAction,
    user: Any,
    text_context: TextSaveContext | None = None,
) -> CardResult:
    """Validate authority and claim/change the draft in one transaction.

    The first statement acquires the SQLite writer lock before any reads. A
    concurrent click or revision is then checked against committed current state.
    """
    from ispec.assistant.tools import (
        assistant_write_error,
        _enqueue_legacy_project_comment_push,
    )

    if error := assistant_write_error("create_project_comment"):
        return _denied(code="write_disabled", message=error)
    if user is None or getattr(user, "can_write_project_comments", True) is not True:
        return _denied(
            code="permission_denied", message="Project-comment actions are unavailable."
        )
    with Session(bind=core_db.get_bind(), expire_on_commit=False) as db, db.begin():
        db.execute(
            update(ProjectCommentCard)
            .where(ProjectCommentCard.id == card_id)
            .values(id=card_id)
        )
        card = db.get(ProjectCommentCard, card_id)
        draft = db.get(ProjectCommentDraft, card.draft_id) if card else None
        actor_allowed = False
        if card is not None and draft is not None:
            if text_context is not None:
                actor_allowed = _text_scope_matches(
                    card=card, draft=draft, context=text_context
                )
            else:
                actor_allowed = _clicker_allowed(
                    card=card, user=user, slack=action.slack
                )
        if not actor_allowed:
            return _denied(
                code="card_unavailable",
                message="This confirmation card is not available to you here.",
            )

        # Reload cookie identities so grant/role revocation is checked in this transaction.
        principal = db.get(AuthUser, user.id) if isinstance(user, AuthUser) else user
        if (
            principal is None
            or not getattr(principal, "is_active", True)
            or get_project_for_user(db, user=principal, project_id=draft.project_id)
            is None
        ):
            return _denied(
                code="project_unavailable", message="Project access is unavailable."
            )

        current = snapshot(card=card, draft=draft)
        if (
            action.draft_id != draft.id
            or action.content_hash != draft.content_hash
            or stable_hash(draft.comment) != draft.content_hash
        ):
            return _denied(
                code="stale_card",
                message="This card no longer matches the stored draft. Use the latest card.",
                card=current,
            )
        if (
            card.assistant_message_id is None
            or card.assistant_message_id <= draft.prepared_turn_id
            or not _has_committed_presentation(
                assistant_db=assistant_db, card=card, draft=draft
            )
        ):
            return _denied(
                code="card_not_presented",
                message="This card has not been presented yet.",
            )

        target_status = {"save": "saved", "cancel": "cancelled", "undo": "undone"}[
            action.action
        ]
        if draft.status == target_status:
            return CardResult(ok=True, card=current, already_completed=True)
        if action.action == "undo":
            return _undo_saved_draft(
                db=db, card=card, draft=draft, action=action, user=user
            )
        if current.status != "pending":
            return _denied(
                code="draft_inactive",
                message=f"This draft is {current.status}; no action was taken.",
                card=current,
            )

        if text_context and card.surface == "slack":
            actor = f"slack:{text_context.origin.requester_id}"
        elif action.slack:
            actor = f"slack:{action.slack.user_id}"
        else:
            actor = f"user:{user.id}"
        receipt = {
            "action": action.action,
            "actor": actor,
            "card_id": card.id,
            "content_hash": draft.content_hash,
            "at": datetime.now(UTC).isoformat(),
            "source": "text_approval" if text_context else "button",
        }
        if action.action == "save":
            if text_context is not None:
                authorization = text_context.authorization
                if (
                    authorization is None
                    or text_context.turn_id <= card.assistant_message_id
                    or authorization.ticket.candidate_comment_hash != draft.content_hash
                    or authorization.ticket.pending_draft_hash != draft.content_hash
                ):
                    return _denied(
                        code="approval_not_granted",
                        message="No matching later-turn approval ticket.",
                        card=current,
                    )
                error = authorization.consume(
                    project_id=draft.project_id,
                    session_id=text_context.session_id,
                    turn_id=str(text_context.turn_id),
                    user_message=text_context.user_message,
                    comment=draft.comment,
                )
                if error:
                    return _denied(
                        code="approval_not_granted", message=error, card=current
                    )
            else:
                authorization = _click_authorization(
                    draft=draft, actor=actor, card_id=card.id
                )
            receipt["ticket"] = authorization.ticket.to_dict()
            draft.status = "saving"
            db.flush()
            person = db.scalar(
                select(Person).where(
                    Person.ppl_Name_First == "iSPEC",
                    Person.ppl_Name_Last == "Assistant",
                )
            )
            if person is None:
                person = Person(
                    ppl_AddedBy=user.username,
                    ppl_Name_First="iSPEC",
                    ppl_Name_Last="Assistant",
                )
                db.add(person)
                db.flush()
            comment = ProjectComment(
                project_id=draft.project_id,
                person_id=person.id,
                com_Comment=draft.comment,
                com_AddedBy=actor,
                com_CommentType=(
                    "client_note"
                    if uses_explicit_project_access(principal)
                    else "assistant_note"
                ),
            )
            db.add(comment)
            db.flush()
            draft.comment_id = comment.id
            receipt["comment_id"] = comment.id
            receipt["undo_target"] = asdict(
                CommentUndoTarget.from_comment(comment=comment)
            )
            receipt["legacy_push_possible"] = os.getenv(
                "ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", ""
            ).strip().lower() in {"1", "true", "yes", "on"}

        draft.status = target_status
        card.action_receipt_json = json.dumps(receipt)
        result = CardResult(ok=True, card=snapshot(card=card, draft=draft))

    if action.action == "save":
        try:
            enqueue = _enqueue_legacy_project_comment_push(
                project_id=result.card.project_id,
                comment_id=result.card.comment_id,
                added_by=actor,
            )
        except Exception as exc:
            enqueue = {"ok": False, "error": type(exc).__name__}
        result = result.model_copy(update={"legacy_push_enqueue": enqueue})
    return result


def _undo_saved_draft(
    *,
    db: Session,
    card: ProjectCommentCard,
    draft: ProjectCommentDraft,
    action: CardAction,
    user: Any,
) -> CardResult:
    """Apply a scoped local reversal and preserve both save and Undo receipts."""
    current = snapshot(card=card, draft=draft)
    if draft.status != "saved":
        return _denied(
            code="draft_inactive",
            message="Only a saved draft can be undone.",
            card=current,
        )
    if draft.expires_at <= datetime.now(UTC).replace(tzinfo=None):
        return _denied(
            code="undo_expired",
            message="This draft's Undo window has expired.",
            card=current,
        )

    receipt = json.loads(card.action_receipt_json or "{}")
    if receipt.get("legacy_push_possible"):
        return _denied(
            code="legacy_undo_unavailable",
            message="Legacy delivery may have occurred; its Undo adapter is not implemented.",
            card=current,
        )
    try:
        target = CommentUndoTarget(**receipt["undo_target"])
    except (KeyError, TypeError):
        return _denied(
            code="undo_receipt_missing",
            message="The save receipt cannot identify an Undo target.",
            card=current,
        )

    outcome = undo_local_comment(db=db, draft=draft, target=target)
    if not outcome.ok:
        return _denied(code=outcome.error_code, message=outcome.error, card=current)

    draft.status = "undone"
    receipt["undo"] = {
        **asdict(outcome),
        "actor": f"slack:{action.slack.user_id}" if action.slack else f"user:{user.id}",
        "at": datetime.now(UTC).isoformat(),
    }
    card.action_receipt_json = json.dumps(receipt)
    return CardResult(ok=True, card=snapshot(card=card, draft=draft))
