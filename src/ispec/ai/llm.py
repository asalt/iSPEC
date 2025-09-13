"""Placeholder utilities for interacting with a language model."""

from __future__ import annotations

from .chat import ChatSession


def generate_response(session: ChatSession) -> ChatSession:
    """Generate a response to the latest user message.

    This is a very small stub that simply echoes the last user message.
    In a real implementation this function would call out to an LLM such
    as one served by `ollama` or any other provider.
    """

    for message in reversed(session.messages):
        if message.sender == "user":
            content = f"Echo: {message.content}"
            return session.add_ai_message(content)
    raise ValueError("No user message found to respond to")


def handle_user_message(content: str, *, session: ChatSession | None = None) -> ChatSession:
    """Add a user message and generate a response.

    Parameters
    ----------
    content:
        The user's input.
    session:
        Existing session to append to. If ``None`` a new session is created.

    Returns
    -------
    ChatSession
        A new session including the user message and the generated reply.
    """

    session = session or ChatSession()
    session = session.add_user_message(content)
    return generate_response(session)
