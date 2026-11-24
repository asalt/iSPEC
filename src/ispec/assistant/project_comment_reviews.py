"""Request delivery of an existing draft and read the outstanding review list."""

from datetime import UTC, datetime
import os

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from ispec.assistant.project_comment_cards import CardOrigin
from ispec.authz import uses_explicit_project_access
from ispec.db.models import (
    AuthUserProject,
    ProjectCommentCard,
    ProjectCommentDraft,
    UserRole,
)


def delivery_state(card: ProjectCommentCard) -> str:
    if card.surface == "staff_review":
        return "queued"
    if card.surface == "slack":
        return "posted" if card.message_ts else "unconfirmed"
    return "private"


def request_review(
    *, db: Session, user, session_id: str, actor_key: str, draft_id: str
):
    """Move the owner's current pending card to the outbox at most once."""
    tool = "request_staff_review"
    if not getattr(user, "is_comment_agent", False) or not user.is_active:
        return {
            "tool": tool,
            "ok": False,
            "error": "A named comment-agent account is required.",
        }
    try:
        origin = CardOrigin.for_staff_review(user=user)
    except ValueError as exc:
        return {"tool": tool, "ok": False, "error": str(exc)}

    pending = select(ProjectCommentDraft.id).where(
        ProjectCommentDraft.id == draft_id,
        ProjectCommentDraft.session_id == session_id,
        ProjectCommentDraft.actor_key == actor_key,
        ProjectCommentDraft.status == "pending",
        ProjectCommentDraft.expires_at > datetime.now(UTC).replace(tzinfo=None),
    )
    # Write first: concurrent requests serialize with each other and with claim/revision.
    db.execute(
        update(ProjectCommentCard)
        .where(
            ProjectCommentCard.draft_id.in_(pending),
            ProjectCommentCard.surface == "web",
            ProjectCommentCard.requester_id == str(user.id),
        )
        .values(
            surface=origin.surface,
            channel_id=origin.channel_id,
            requester_id=origin.requester_id,
        )
    )
    card = db.scalar(
        select(ProjectCommentCard).where(
            ProjectCommentCard.draft_id.in_(pending),
            ProjectCommentCard.requester_id == origin.requester_id,
            ProjectCommentCard.surface.in_(("slack", "staff_review")),
        )
    )
    if card is None:
        return {
            "tool": tool,
            "ok": False,
            "error": "No shareable pending draft in this conversation.",
        }
    return {
        "tool": tool,
        "ok": True,
        "saved": False,
        "result": {"draft_id": draft_id, "delivery": delivery_state(card)},
    }


def outstanding_reviews(
    *, db: Session, user, project_id: int | None, offset: int, limit: int
):
    """List visible pending drafts, without returning private thread IDs or action tokens."""
    draft, card = ProjectCommentDraft, ProjectCommentCard
    visible = draft.actor_key == f"{user.id}:{user.username}"
    channel = os.getenv("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL", "").strip()
    if user.role in {UserRole.admin, UserRole.editor} and channel.startswith(
        ("C", "G")
    ):
        visible = or_(
            visible,
            (card.channel_id == channel) & card.surface.in_(("slack", "staff_review")),
        )
    query = (
        select(draft, card)
        .join(card, card.draft_id == draft.id)
        .where(
            visible,
            draft.status == "pending",
            draft.expires_at > datetime.now(UTC).replace(tzinfo=None),
            card.assistant_message_id.is_not(None),
        )
    )
    if uses_explicit_project_access(user) and not getattr(
        user, "is_comment_agent", False
    ):
        query = query.where(
            draft.project_id.in_(
                select(AuthUserProject.project_id).where(
                    AuthUserProject.user_id == user.id
                )
            )
        )
    if project_id is not None:
        query = query.where(draft.project_id == project_id)
    rows = db.execute(
        query.order_by(draft.created_at.desc(), draft.id)
        .offset(offset)
        .limit(limit + 1)
    ).all()
    return {
        "drafts": [
            {
                "draft_id": d.id,
                "project_id": d.project_id,
                "comment": d.comment,
                "delivery": delivery_state(c),
                "expires_at": d.expires_at.isoformat() + "Z",
            }
            for d, c in rows[:limit]
        ],
        "nextOffset": offset + limit if len(rows) > limit else None,
    }
