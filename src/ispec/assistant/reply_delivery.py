"""User-facing delivery wording based on receipts, never assistant prose."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


UNRETAINED_REPLY = "I'm unable to reply right now. Please try again later."
RETAINED_REPLY = (
    "I've received and saved your reply, but my response is delayed. "
    "Please try again later."
)


class ReplyRetention(BaseModel):
    storage: Literal["support_message", "agent_command"]
    recordId: int = Field(ge=1, strict=True)
    automaticDelivery: Literal[False] = False


def failure_reply(payload: Any = None) -> str:
    """Only the backend's structured unavailable receipt establishes retention."""
    if isinstance(payload, dict) and payload.get("responseStatus") == "unavailable":
        try:
            ReplyRetention.model_validate(payload.get("retention"))
        except (ValidationError, TypeError):
            pass
        else:
            return RETAINED_REPLY
    return UNRETAINED_REPLY


def failed_request_reply(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    try:
        body = response.json() if response is not None else None
    except (ValueError, TypeError):
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    return failure_reply(detail)


def chat_reply_text(payload: dict[str, Any]) -> str:
    if payload.get("responseStatus") == "unavailable":
        return failure_reply(payload)
    return str(payload.get("message") or "").strip()


class SlackEventClaims:
    """Bounded, process-local retry suppression, NOT a durable message inbox."""

    def __init__(self, *, ttl_seconds: float = 3600, limit: int = 4096):
        self.ttl_seconds = ttl_seconds
        self.limit = limit
        self._seen: OrderedDict[tuple[str, ...], float] = OrderedDict()
        self._lock = Lock()

    def claim(self, *, team_id: str | None, event: dict[str, Any], route: str) -> bool:
        channel = str(event.get("channel") or "")
        ts = str(event.get("ts") or "")
        if not channel or not ts:
            return False
        key = (str(team_id or ""), channel, ts, route)
        now = time.monotonic()
        with self._lock:
            while (
                self._seen and next(iter(self._seen.values())) <= now - self.ttl_seconds
            ):
                self._seen.popitem(last=False)
            if key in self._seen:
                return False
            self._seen[key] = now
            while len(self._seen) > self.limit:
                self._seen.popitem(last=False)
            return True
