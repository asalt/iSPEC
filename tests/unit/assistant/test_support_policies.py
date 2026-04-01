from __future__ import annotations

from ispec.assistant.support_policies import (
    comment_intent_messages_for_write_mode,
    hinted_support_tool_groups,
    select_support_tool_policy,
)


def test_select_support_tool_policy_for_project_existence_lookup() -> None:
    selection = select_support_tool_policy(message="does project 1598 exist?")

    assert selection is not None
    assert selection.rule_name == "project_existence_lookup"
    assert selection.tool_name == "get_project"
    assert selection.args == {"id": 1598}
    assert selection.messages
    assert "answer directly whether it exists" in selection.messages[0]["content"]


def test_select_support_tool_policy_for_tmux_request_extracts_session_name() -> None:
    selection = select_support_tool_policy(message="hello what is going on on the ispec tmux session?")

    assert selection is not None
    assert selection.rule_name == "tmux_inspection"
    assert selection.tool_name == "assistant_list_tmux_panes"
    assert selection.args == {"session_name": "ispec"}


def test_hinted_support_tool_groups_accumulates_project_and_file_hints() -> None:
    hinted = hinted_support_tool_groups(
        message="can you check the results files for project 1598?",
        focused_project_id=1598,
    )

    assert hinted == {"files", "projects"}


def test_comment_intent_messages_for_save_now_missing_text() -> None:
    messages = comment_intent_messages_for_write_mode(write_mode="save_now", missing_comment_text=True)

    assert len(messages) == 1
    assert "comment text is still missing" in messages[0]["content"]

