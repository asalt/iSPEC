from __future__ import annotations

from ispec.assistant.tool_routing import router_fallback_tool_names, tool_groups_for_available_tools


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


def test_router_failure_keeps_only_project_workflow_and_discovery():
    groups = tool_groups_for_available_tools({
        "get_project", "latest_project_comments", "search_people",
        "assistant_stats", "assistant_enqueue_dev_restart_services",
    })
    selected = router_fallback_tool_names(
        groups=groups,
        workflow_tool_names={"prepare_project_comment", "cancel_project_comment_draft"},
    )
    assert selected == {
        "get_project", "latest_project_comments", "prepare_project_comment",
        "cancel_project_comment_draft",
    }


def test_router_failure_without_authorized_projects_does_not_add_them():
    selected = router_fallback_tool_names(
        groups=tool_groups_for_available_tools({"search_people"}),
        workflow_tool_names=set(),
    )
    assert selected == set()
