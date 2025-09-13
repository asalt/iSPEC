"""Chat session data structures used by the AI module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a chat conversation."""

    sender: str
    content: str


@dataclass(frozen=True)
class ChatSession:
    """Immutable chat session holding the conversation history.

    The session stores messages as a tuple so that modifications return
    a new :class:`ChatSession` instance, keeping the dataclass frozen and
    side-effect free.
    """

    messages: Tuple[ChatMessage, ...] = ()

    def add_message(self, sender: str, content: str) -> "ChatSession":
        """Return a new session with an additional message."""

        return ChatSession(self.messages + (ChatMessage(sender, content),))

    def add_user_message(self, content: str) -> "ChatSession":
        """Convenience wrapper to add a user message."""

        return self.add_message("user", content)

    def add_ai_message(self, content: str) -> "ChatSession":
        """Convenience wrapper to add an AI generated message."""

        return self.add_message("ai", content)

    @classmethod
    def from_messages(cls, messages: Iterable[ChatMessage]) -> "ChatSession":
        """Create a session from an iterable of messages."""

        return cls(tuple(messages))
