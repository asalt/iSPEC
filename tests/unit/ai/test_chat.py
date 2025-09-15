"""Tests for chat session utilities."""

import pytest

from ispec.ai.chat import ChatMessage, ChatSession
from ispec.ai.llm import generate_response, handle_user_message


def test_chat_session_add_message():
    session = ChatSession()
    session2 = session.add_user_message("hello")
    assert session is not session2
    assert len(session.messages) == 0
    assert len(session2.messages) == 1
    assert session2.messages[0].content == "hello"


def test_chat_session_immutability_helpers():
    session = ChatSession()

    ai_session = session.add_ai_message("hi")
    assert ai_session is not session
    assert len(session.messages) == 0
    assert len(ai_session.messages) == 1
    assert ai_session.messages[0].sender == "ai"

    generic_session = session.add_message("user", "there")
    assert generic_session is not session
    assert len(generic_session.messages) == 1
    assert generic_session.messages[0].sender == "user"
    assert generic_session.messages[0].content == "there"


def test_chat_session_from_messages_iterable():
    msgs = [ChatMessage("user", "a"), ChatMessage("ai", "b")]
    session_from_list = ChatSession.from_messages(msgs)
    assert session_from_list.messages == tuple(msgs)

    session_from_gen = ChatSession.from_messages(m for m in msgs)
    assert session_from_gen.messages == tuple(msgs)


def test_generate_response_no_user_message():
    session = ChatSession().add_ai_message("hi")
    with pytest.raises(ValueError):
        generate_response(session)


def test_handle_user_message_generates_response():
    session = handle_user_message("hi there")
    assert len(session.messages) == 2
    assert session.messages[0].sender == "user"
    assert session.messages[1].sender == "ai"
    assert "hi there" in session.messages[1].content


def test_handle_user_message_appends_to_session():
    base = ChatSession().add_user_message("init").add_ai_message("Echo: init")
    updated = handle_user_message("next", session=base)
    assert updated is not base
    assert len(updated.messages) == len(base.messages) + 2
    assert updated.messages[-2].sender == "user"
    assert updated.messages[-2].content == "next"
    assert updated.messages[-1].sender == "ai"
    assert "next" in updated.messages[-1].content
    assert len(base.messages) == 2
