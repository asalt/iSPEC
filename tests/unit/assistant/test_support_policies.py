from __future__ import annotations

from ispec.assistant.service import AssistantReply
from ispec.assistant.support_policies import (
    comment_intent_messages_for_write_mode,
    hinted_support_tool_groups,
    select_support_tool_policy,
)


def _pane(*, session: str, session_group: str | None, target: str, capture_target: str, title: str, command: str) -> dict:
    return {
        "session": session,
        "session_group": session_group,
        "session_names": [session],
        "target": target,
        "preferred_alias": target if session_group is None else target.replace(session + ":", session_group + ":"),
        "capture_target": capture_target,
        "pane_id": capture_target,
        "pane_number": int(str(capture_target).lstrip("%") or 0),
        "target_aliases": [target],
        "window_aliases": [target.rsplit(".", 1)[0]],
        "window_name": target.split(":", 1)[1].split(".", 1)[0],
        "pane_title": title,
        "current_command": command,
    }


def test_select_support_tool_policy_for_project_existence_lookup() -> None:
    selection = select_support_tool_policy(message="does project 1598 exist?")

    assert selection is not None
    assert selection.rule_name == "project_existence_lookup"
    assert selection.tool_name == "get_project"
    assert selection.args == {"id": 1598}
    assert selection.messages
    assert "answer directly whether it exists" in selection.messages[0]["content"]


def test_select_support_tool_policy_for_tmux_request_resolves_unique_pane(monkeypatch) -> None:
    monkeypatch.setattr(
        "ispec.assistant.support_policies._tmux_policy_panes",
        lambda: [
            _pane(
                session="ispec-0",
                session_group="ispec",
                target="ispec-0:node.1",
                capture_target="%0",
                title="codex resume --all /home/alex/tools/ispec-full",
                command="node",
            )
        ],
    )

    selection = select_support_tool_policy(message="hello what is going on on the ispec tmux pane?")

    assert selection is not None
    assert selection.rule_name == "tmux_capture_unique_pane"
    assert selection.tool_name == "assistant_capture_tmux_pane"
    assert selection.args == {"target": "%0", "lines": 40}
    assert selection.force_tool_choice is True
    assert selection.override_tool_args is True
    assert selection.meta is not None
    assert selection.meta["selected_target"] == "%0"
    assert selection.meta["strategy"] == "unique_candidate"
    assert "summarize the pane's current state" in selection.messages[0]["content"]


def test_select_support_tool_policy_for_tmux_raw_request_prefers_larger_capture(monkeypatch) -> None:
    monkeypatch.setattr(
        "ispec.assistant.support_policies._tmux_policy_panes",
        lambda: [
            _pane(
                session="ispec-0",
                session_group="ispec",
                target="ispec-0:node.1",
                capture_target="%0",
                title="codex resume --all /home/alex/tools/ispec-full",
                command="node",
            )
        ],
    )

    selection = select_support_tool_policy(message="show me the raw output from the ispec tmux pane")

    assert selection is not None
    assert selection.rule_name == "tmux_capture_unique_pane"
    assert selection.args == {"target": "%0", "lines": 120}


def test_select_support_tool_policy_for_tmux_request_does_not_invent_session_name(monkeypatch) -> None:
    monkeypatch.setattr(
        "ispec.assistant.support_policies._tmux_policy_panes",
        lambda: [
            _pane(
                session="ispecfull",
                session_group=None,
                target="ispecfull:backend.1",
                capture_target="%2",
                title="backend logs",
                command="make",
            ),
            _pane(
                session="ispecfull",
                session_group=None,
                target="ispecfull:supervisor.1",
                capture_target="%4",
                title="supervisor logs",
                command="python",
            ),
        ],
    )

    selection = select_support_tool_policy(message="what is going on in the default tmux pane?")

    assert selection is not None
    assert selection.rule_name == "tmux_list_choices"
    assert selection.tool_name == "assistant_list_tmux_panes"
    assert selection.args == {"session_name": "ispecfull"}
    assert selection.meta is not None
    assert selection.meta["selected_target"] is None
    assert selection.meta["strategy"] == "list_session"


def test_select_support_tool_policy_for_tmux_request_can_use_classifier(monkeypatch) -> None:
    monkeypatch.setattr(
        "ispec.assistant.support_policies._tmux_policy_panes",
        lambda: [
            _pane(
                session="ispec-0",
                session_group="ispec",
                target="ispec-0:node.1",
                capture_target="%0",
                title="codex task alpha",
                command="node",
            ),
            _pane(
                session="ispec-1",
                session_group="ispec",
                target="ispec-1:node.1",
                capture_target="%7",
                title="codex task beta",
                command="node",
            ),
        ],
    )
    monkeypatch.setattr(
        "ispec.assistant.support_policies.generate_classifier_reply",
        lambda **_: AssistantReply(
            content='{"candidate_key":"c2","confidence":0.84,"reason":"The second codex pane is the better match."}',
            provider="test",
            model="classifier-test",
        ),
    )

    selection = select_support_tool_policy(
        message="what is going on in the codex tmux pane?",
        generate_reply_fn=lambda **_: AssistantReply(content="", provider="base", model="base"),
    )

    assert selection is not None
    assert selection.rule_name == "tmux_capture_classifier_choice"
    assert selection.tool_name == "assistant_capture_tmux_pane"
    assert selection.args == {"target": "%7", "lines": 40}
    assert selection.meta is not None
    assert selection.meta["classifier_used"] is True
    assert selection.meta["classifier_confidence"] == 0.84


def test_select_support_tool_policy_for_tmux_list_request_prefers_listing(monkeypatch) -> None:
    monkeypatch.setattr(
        "ispec.assistant.support_policies._tmux_policy_panes",
        lambda: [
            _pane(
                session="ispec-0",
                session_group="ispec",
                target="ispec-0:node.1",
                capture_target="%0",
                title="codex task alpha",
                command="node",
            ),
            _pane(
                session="ispec-1",
                session_group="ispec",
                target="ispec-1:node.1",
                capture_target="%7",
                title="codex task beta",
                command="node",
            ),
        ],
    )

    selection = select_support_tool_policy(message="show me the panes in the ispec tmux session")

    assert selection is not None
    assert selection.rule_name == "tmux_list_choices"
    assert selection.tool_name == "assistant_list_tmux_panes"
    assert selection.args == {"session_name": "ispec"}
    assert selection.meta is not None
    assert selection.meta["strategy"] == "list_session"


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
