"""Pending assistant notes, kept with comments for atomic save/idempotency."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ProjectCommentDraft(Base):
    __tablename__ = "project_comment_draft"
    __table_args__ = (Index("ix_comment_draft_conversation", "session_id", "actor_key", "status"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(Text)
    actor_key: Mapped[str] = mapped_column(Text)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"))
    comment: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    initial_request: Mapped[str] = mapped_column(Text)
    prepared_turn_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    comment_id: Mapped[int | None] = mapped_column(ForeignKey("project_comment.id"), nullable=True)


class ProjectCommentSaveOffer(Base):
    """A server-rendered offer, not permission or evidence of delivery to Slack."""

    __tablename__ = "project_comment_save_offer"
    __table_args__ = (Index("ix_comment_offer_scope", "session_id", "actor_key", "draft_id", "status"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    draft_id: Mapped[str] = mapped_column(ForeignKey("project_comment_draft.id"))
    content_hash: Mapped[str] = mapped_column(String(64))
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"))
    session_id: Mapped[str] = mapped_column(Text)
    actor_key: Mapped[str] = mapped_column(Text)
    assistant_message_id: Mapped[int] = mapped_column(Integer)
    presentation_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), default="active")
