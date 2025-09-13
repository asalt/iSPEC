"""Tests for chat session utilities."""

from ispec.ai import ChatSession, handle_user_message


def test_chat_session_add_message():
    session = ChatSession()
    session2 = session.add_user_message("hello")
    assert session is not session2
    assert len(session.messages) == 0
    assert len(session2.messages) == 1
    assert session2.messages[0].content == "hello"


def test_handle_user_message_generates_response():
    session = handle_user_message("hi there")
    assert len(session.messages) == 2
    assert session.messages[0].sender == "user"
    assert session.messages[1].sender == "ai"
    assert "hi there" in session.messages[1].content
