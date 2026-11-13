"""Authenticated web and internal Slack adapters for the shared card executor."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ispec.api.security import (
    _provided_api_key,
    require_assistant_access,
    require_configured_api_key,
    require_user,
)
from ispec.assistant.project_comment_cards import (
    CardAction,
    CardResult,
    ConfirmationCard,
    act_on_card,
    cards_for_conversation,
)
from ispec.assistant.connect import get_assistant_session_dep
from ispec.db.connect import get_session_dep
from ispec.db.models import ProjectCommentCard


router = APIRouter(prefix="/support/project-comment-cards", tags=["support"])


@router.get("", response_model=list[ConfirmationCard])
def list_cards(
    session_id: str = Query(min_length=1, max_length=256),
    user: Any = Depends(require_user),
    db: Session = Depends(get_session_dep),
):
    """Restore only this browser user's cards for the selected conversation."""
    from ispec.authz import get_project_for_user

    cards = cards_for_conversation(
        db=db, session_id=session_id, actor_key=f"{user.id}:{user.username}"
    )
    return [
        card
        for card in cards
        if get_project_for_user(db, user=user, project_id=card.project_id) is not None
    ]


@router.post("/{card_id}/actions", response_model=CardResult)
def card_action(
    card_id: str,
    payload: CardAction,
    user: Any = Depends(require_assistant_access),
    db: Session = Depends(get_session_dep),
    assistant_db: Session = Depends(get_assistant_session_dep),
    provided_key: str | None = Depends(_provided_api_key),
):
    """Accept cookie-authenticated clicks or verified internal Socket Mode actions."""
    if payload.slack is not None:
        require_configured_api_key(provided_key)
        if getattr(user, "id", None) != 0:
            raise HTTPException(
                status_code=403,
                detail="Slack actions require the internal receiver identity.",
            )
    elif getattr(user, "id", 0) <= 0:
        raise HTTPException(
            status_code=403,
            detail="A browser account is required for web card actions.",
        )
    return act_on_card(
        core_db=db,
        assistant_db=assistant_db,
        card_id=card_id,
        action=payload,
        user=user,
    )


class CardDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team_id: str = Field(min_length=1, max_length=80)
    channel_id: str = Field(min_length=1, max_length=80)
    thread_ts: str = Field(default="", max_length=80)
    message_ts: str = Field(min_length=1, max_length=80)


@router.post("/{card_id}/delivery", dependencies=[Depends(require_configured_api_key)])
def card_delivery(
    card_id: str, payload: CardDelivery, db: Session = Depends(get_session_dep)
):
    """Bind a posted Slack card to its exact message before accepting its buttons."""
    with Session(bind=db.get_bind()) as write_db, write_db.begin():
        changed = write_db.execute(
            update(ProjectCommentCard)
            .where(
                ProjectCommentCard.id == card_id,
                ProjectCommentCard.surface == "slack",
                ProjectCommentCard.team_id == payload.team_id,
                ProjectCommentCard.channel_id == payload.channel_id,
                ProjectCommentCard.thread_ts == payload.thread_ts,
                ProjectCommentCard.message_ts.is_(None),
                ProjectCommentCard.assistant_message_id.is_not(None),
            )
            .values(message_ts=payload.message_ts)
        ).rowcount
        if changed:
            return {"ok": True}
        existing = write_db.scalar(
            select(ProjectCommentCard).where(
                ProjectCommentCard.id == card_id,
                ProjectCommentCard.surface == "slack",
                ProjectCommentCard.team_id == payload.team_id,
                ProjectCommentCard.channel_id == payload.channel_id,
                ProjectCommentCard.thread_ts == payload.thread_ts,
                ProjectCommentCard.message_ts == payload.message_ts,
            )
        )
        if existing:
            return {"ok": True}
    raise HTTPException(
        status_code=409,
        detail="Card delivery does not match its conversation or existing message.",
    )
