"""Opt-in real inference through iSPEC, with temporary DBs and bounded tools.

These test the adapter and controller, not deployment, Slack transport, or HTTP
authentication. Every non-inference network request and non-fixture tool fails.
"""

import json
import copy
import os
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests

from ispec.api.routes import support
from ispec.assistant import service
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession, SupportMessage
from ispec.db.connect import get_session
from ispec.db.models import Project, ProjectComment, UserRole, ProjectAccessMode
from ispec.schedule.connect import get_schedule_session

pytestmark = [pytest.mark.behavioral, pytest.mark.skipif(
    os.getenv("ISPEC_RUN_LLM_TESTS") != "1", reason="Explicit isolated inference only")]

TOOL = {"type":"function","function":{"name":"get_project","description":"Read a synthetic project's sample count.",
        "parameters":{"type":"object","properties":{"project_id":{"type":"integer"}},"required":["project_id"]}}}


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path, behavioral_datastore):
    endpoint = urlsplit(os.environ["ISPEC_VLLM_URL"])
    assert endpoint.scheme == "http" and endpoint.hostname == "127.0.0.1"
    original = requests.sessions.Session.request
    def guarded(self, method, url, **kw):
        target = urlsplit(url)
        assert (target.scheme, target.hostname, target.port) == (endpoint.scheme, endpoint.hostname, endpoint.port)
        assert target.path in {"/v1/models", "/v1/chat/completions"}
        kw["allow_redirects"] = False
        return original(self, method, url, **kw)
    monkeypatch.setattr(requests.sessions.Session, "request", guarded)
    for key, value in behavioral_datastore.env_overrides().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ISPEC_INFERENCE_USAGE_LOG_ENABLED", "0")
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "0")
    monkeypatch.setenv("ISPEC_VLLM_STRICT_PROTOCOL", "1")
    for key, value in {
        "PROVIDER":"vllm", "CLASSIFIER_PROVIDER":"inherit", "TOOL_PROTOCOL":"openai",
        "TEMPERATURE":"0", "MAX_OUTPUT_TOKENS":os.getenv("ISPEC_EVAL_MAX_OUTPUT_TOKENS", "512"), "MAX_TOOL_CALLS":"3",
        "MAX_PROMPT_TOKENS":"4000", "HISTORY_LIMIT":"10", "SUMMARY_MAX_CHARS":"0",
        "SELF_REVIEW":"0", "SELF_REVIEW_DECIDER":"0", "ENABLE_SELF_REVIEW":"0",
        "COMPACTION_ENABLED":"0", "CHAT_QUEUE_ENABLED":"0", "TURN_DECISION_MODE":"off",
        "PROJECT_COMMENT_INTENT_HINTS":"0", "ENABLE_RESPONSE_CONTRACTS":"off",
        "PROJECT_COMMENT_APPROVAL_EVAL_MODE":"own", "PROJECT_COMMENT_APPROVAL_APPROVE_THRESHOLD":"0.75",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    return behavioral_datastore


@pytest.mark.parametrize("choice", ["auto", "required", {"type":"function","function":{"name":"get_project"}}])
def test_native_tool_roundtrip(isolated_runtime, choice):
    native_roundtrip(choice)


def test_native_default_system_roundtrip(isolated_runtime):
    native_roundtrip("auto", default_system=True)


def native_roundtrip(choice, *, default_system=False):
    messages = [] if default_system else [{"role":"system","content":"Use the supplied tool to answer; do not guess."}]
    messages.append({"role":"user","content":"How many samples were received for project 42?"})
    t = time.monotonic()
    first = service.generate_reply(messages=messages, tools=[TOOL], tool_choice=choice, timeout_seconds=120)
    print("\nNATIVE_FIRST " + json.dumps({"choice":choice,"default_system":default_system,"reply":first.__dict__}))
    assert first.ok, first.error
    assert len(first.tool_calls or []) == 1
    call = first.tool_calls[0]
    assert json.loads(call["function"]["arguments"]) == {"project_id":42}
    messages.extend([{"role":"assistant","content":first.content,"tool_calls":first.tool_calls},
                     {"role":"tool","tool_call_id":call["id"],"content":'{"project_id":42,"sample_count":11}'}])
    second = service.generate_reply(messages=messages, tools=[TOOL], tool_choice="none", timeout_seconds=120)
    print("\nNATIVE_RESULT " + json.dumps({"choice":choice,"default_system":default_system,"reply":second.__dict__,"total_ms":round((time.monotonic()-t)*1000)}))
    assert second.ok and not second.tool_calls
    assert "11" in second.content or "eleven" in second.content.lower()


@pytest.mark.parametrize("kind", ["count", "approve", "deny", "social_close", "social_close_short", "approve_required_only"])
def test_controller_with_real_fixture_tools(isolated_runtime, monkeypatch, kind):
    store = isolated_runtime
    allowed = {"count_all_projects", "get_project", "create_project_comment"}
    original_schemas = support.openai_tools_for_user
    def schemas(user):
        tools = copy.deepcopy([tool for tool in original_schemas(user) if tool["function"]["name"] in allowed])
        if kind == "approve_required_only":
            # Diagnostic only: isolate hallucinated optional arguments. Never
            # change the production schema or silently repair returned calls.
            parameters = next(t for t in tools if t["function"]["name"] == "create_project_comment")["function"]["parameters"]
            parameters["properties"] = {key: parameters["properties"][key] for key in parameters["required"]}
            parameters["additionalProperties"] = False
        return tools
    monkeypatch.setattr(support, "openai_tools_for_user", schemas)
    original_run = support.run_tool
    executed = []
    def fixture_tool(**kw):
        assert kw["name"] in allowed
        assert str(kw["core_db"].bind.url.database) == str(store.core_db_path)
        executed.append(kw["name"])
        return original_run(**kw)
    monkeypatch.setattr(support, "run_tool", fixture_tool)
    user = SimpleNamespace(id=1, username="fixture", role=UserRole.viewer,
                           project_access_mode=ProjectAccessMode.all_projects, can_write_project_comments=True)
    with get_session(store.core_db_path) as core, get_assistant_session(store.assistant_db_path) as adb, get_schedule_session(store.schedule_db_path) as sdb:
        project_id = store.project_ids[0]
        if kind == "social_close_short":
            project_id = 42
            core.add(Project(id=project_id, prj_AddedBy="behavioral", prj_ProjectTitle="Synthetic saliva review"))
            core.commit()
        session = SupportSession(session_id="runtime-fixture", user_id=1, state_json=json.dumps({"current_project_id":project_id}))
        adb.add(session); adb.flush()
        if kind != "count":
            draft = f'Draft for project {project_id}: "Eleven saliva samples arrived; peptide coverage review is pending." Shall I save this exact note to project history?'
            if kind == "social_close_short":
                draft = f"Draft for project {project_id}: Eleven saliva samples arrived; peptide coverage review is pending. Shall I save this exact note?"
            adb.add(SupportMessage(session_pk=session.id, role="assistant", content=draft))
        adb.commit()
        text = {"count":"Use the database tool: how many total projects are there?",
                "approve":"Yep, that captures it. Put that in the project history please.",
                "approve_required_only":"Yep, that captures it. Put that in the project history please.",
                "deny":"Yes that's the right project. No, don't save a comment.",
                "social_close":"Much appreciated, catch you tomorrow!",
                "social_close_short":"Much appreciated, catch you tomorrow!"}[kind]
        t = time.monotonic()
        result = support.chat(support.ChatRequest(sessionId=session.session_id, message=text, history=[]),
                              assistant_db=adb, core_db=core, schedule_db=sdb, user=user)
        metadata = json.loads(adb.get(SupportMessage, result.messageId).meta_json)
        comments = core.query(ProjectComment).all()
        print("\nCONTROLLER " + json.dumps({"case":kind,"response":result.message,"elapsed_ms":round((time.monotonic()-t)*1000),
             "executed":executed,"comment_count":len(comments),"metadata":metadata}, default=str))
        assert len(comments) == int(kind.startswith("approve"))
        if kind == "count":
            assert "count_all_projects" in executed
            assert "3" in result.message or "three" in result.message.lower()
        elif kind.startswith("approve"):
            assert metadata["project_comment_approval_eval"]["write_outcome"]["status"] == "succeeded"
        else:
            assert "create_project_comment" not in executed
            assert not metadata["project_comment_approval_eval"]["write_ticket"]["issued"]
