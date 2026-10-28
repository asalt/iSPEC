from types import SimpleNamespace

import pytest

from ispec.assistant import tools
from ispec.db.models import UserRole, ProjectAccessMode


@pytest.mark.parametrize("name", sorted(tools._WRITE_TOOL_NAMES))
@pytest.mark.parametrize("setting", ["1", "true", "invalid"])
def test_read_only_denies_before_touching_any_storage(monkeypatch, name, setting):
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", setting)
    user = SimpleNamespace(id=1, username="admin", role=UserRole.admin,
                           project_access_mode=ProjectAccessMode.all_projects)
    reply = tools.run_tool(name=name, args={"confirm": True}, core_db=None,
                           user=user, project_comment_save_authorized=True,
                           project_comment_authorization=object())
    assert reply["ok"] is False
    assert "read-only" in reply["error"]


def test_read_only_hides_writes_but_retains_queries(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", "1")
    user = SimpleNamespace(id=1, username="admin", role=UserRole.admin,
                           project_access_mode=ProjectAccessMode.all_projects)
    names = {tool["function"]["name"] for tool in tools.openai_tools_for_user(user)}
    assert "get_project" in names
    assert not names.intersection(tools._WRITE_TOOL_NAMES)
    prompt = tools.tool_prompt(tool_names={"get_project", "create_project_comment"})
    assert "- get_project(" in prompt
    assert "create_project_comment" not in prompt
    assert "drafts are not saved" in prompt


def test_read_only_is_opt_in(monkeypatch):
    monkeypatch.delenv("ISPEC_ASSISTANT_READ_ONLY", raising=False)
    assert not tools.assistant_read_only()
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", "0")
    assert not tools.assistant_read_only()


def test_background_prepare_can_be_disabled_without_touching_queue(monkeypatch):
    from ispec.assistant.controller import enqueue_post_send_prepare_command

    monkeypatch.setenv("ISPEC_ASSISTANT_POST_SEND_PREPARE_ENABLED", "0")
    result = enqueue_post_send_prepare_command(agent_db=None, source="support_chat",
                                               thread_key="support:synthetic", payload={})
    assert not result.enqueued
    assert result.reason == "disabled"
