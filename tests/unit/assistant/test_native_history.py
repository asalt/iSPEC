"""Exact tool evidence, access boundaries and budget behavior across user turns."""

import json

import pytest

from ispec.assistant.models import SupportMessage
from ispec.assistant.native_history import access_key, complete_exchanges, recent_exchanges, replay_history
from ispec.assistant.prompting import estimate_tokens_for_messages
from ispec.db.models import AuthUser, AuthUserProject, Project, UserRole


def test_replay_key_changes_when_a_client_loses_a_grant(db_session):
    user = AuthUser(username="reader", role=UserRole.client, password_hash="test", password_salt="test")
    project = Project(prj_ProjectTitle="Synthetic", prj_AddedBy="test")
    db_session.add_all([user, project])
    db_session.flush()
    grant = AuthUserProject(user_id=user.id, project_id=project.id)
    db_session.add(grant)
    db_session.flush()
    options = dict(db=db_session, user=user, actor="reader", tool_names={"get_project"})
    original_key = access_key(**options)
    db_session.delete(grant)
    db_session.flush()
    assert access_key(**options) != original_key


def lookup(call_id):
    return {
        "id": call_id, "type": "function",
        "function": {"name": "get_project", "arguments": '{"id":57}'},
    }


def tool_result(call_id, *, ok):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"ok": ok})}


def test_parallel_calls_keep_each_original_result_together():
    first, second = lookup("a"), lookup("b")
    call = {"role": "assistant", "content": "", "tool_calls": [first, second]}
    results = [tool_result("b", ok=False), tool_result("a", ok=True)]
    group = [call, *results]
    assert complete_exchanges([call, {"role": "system", "content": "controller"}, *results]) == [group]
    assert complete_exchanges(group[:-1]) == []
    assert complete_exchanges([call, results[0], results[0]]) == []
    exact_cost = estimate_tokens_for_messages(group)
    assert recent_exchanges([group], budget=exact_cost) == group
    assert recent_exchanges([group], budget=exact_cost - 1) == []


@pytest.mark.parametrize("reason", ["changed_access", "origin_missing", "unrelated_user_turn", "budget", "old_row"])
def test_missing_or_ineligible_evidence_retains_visible_reply(reason):
    calls = [{"role": "assistant", "tool_calls": [lookup("1")]}, tool_result("1", ok=False)]
    receipt = {"messages": calls, "user_message_id": 12, "scope": "original"}
    question = SupportMessage(id=12, role="user", content="What is the project?")
    reply = SupportMessage(id=13, role="assistant", content="The lookup failed.",
                           meta_json=json.dumps({"native_history": receipt}))
    assert replay_history([question, reply], scope="original", remaining_tokens=500)[1:-1] == calls
    rows = [question, reply]
    if reason == "origin_missing":
        rows = [reply]
    if reason == "unrelated_user_turn":
        question.id = 11
    if reason == "old_row":
        reply.meta_json = json.dumps({"tool_calls": [{"name": "get_project", "result_preview": "failed"}]})
    actual = replay_history(
        rows, scope="other" if reason == "changed_access" else "original",
        remaining_tokens=0 if reason == "budget" else 500,
    )
    assert actual == [{"role": row.role, "content": row.content} for row in rows]
