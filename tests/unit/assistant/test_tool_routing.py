from __future__ import annotations

from ispec.assistant.tool_routing import tool_groups_for_available_tools


def test_bridge_group_keeps_tmux_read_and_slack_relay_tools_together() -> None:
    available = {
        "assistant_list_tmux_panes",
        "assistant_capture_tmux_pane",
        "assistant_compare_tmux_pane",
        "assistant_list_slack_artifact_replies",
        "assistant_relay_slack_reply_to_tmux",
    }

    groups = tool_groups_for_available_tools(available)
    by_name = {group.name: group for group in groups}

    assert set(by_name["tmux"].tool_names) == {
        "assistant_list_tmux_panes",
        "assistant_capture_tmux_pane",
        "assistant_compare_tmux_pane",
    }
    assert set(by_name["bridge"].tool_names) == available
    assert "read-only" in by_name["tmux"].description.lower()
    assert "relay" in by_name["bridge"].description.lower()
