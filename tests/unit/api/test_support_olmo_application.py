"""Synthetic controller-to-wire integration; only the HTTP model transport is stubbed.

These are hand-authored unit fixtures, not model benchmarks or training examples.
"""

from dataclasses import dataclass, field
from json import dumps, loads
from types import SimpleNamespace

import pytest

from ispec.api.routes import support
from ispec.api.security import _ApiKeyServiceUser
from ispec.assistant import service
from ispec.assistant.models import SupportMessage
from ispec.prompt.loader import load_prompt_source
from ispec.schedule.connect import get_schedule_session
from scripts.staged_prompt_trace import WireRequest
from tests.unit.assistant.test_project_comment_cards import (
    NOTE,
    conversation,  # noqa: F401 (pytest fixture)
)  # noqa: F401


@dataclass(kw_only=True)
class ScriptedModel:
    approval_label: str = "approve_save"
    confidence: float = 0.75
    requests: list[WireRequest] = field(default_factory=list)
    tool_round: int = 0
    draft_id: str | None = None
    classifier_calls: int = 0

    def post(self, url, *, json: dict, **kwargs):
        if url.endswith("/tokenize"):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"count": 1000, "max_model_len": 8192},
            )
        assert url == "http://127.0.0.1:18099/v1/chat/completions"
        request = WireRequest.model_validate(json)
        self.requests.append(request)
        if request.tools:
            message = self.tool_reply(request=request)
        else:
            schema = json["structured_outputs"]["json"]
            if "primary" in schema["properties"]:
                decision = {
                    "primary": "projects",
                    "secondary": [],
                    "confidence": 0.99,
                    "clarify": False,
                }
            else:
                assert "label" in schema["properties"]
                self.classifier_calls += 1
                decision = {
                    "label": self.approval_label,
                    "confidence": self.confidence,
                    "reason": "Synthetic classifier result.",
                }
            message = {"content": dumps(decision)}

        body = {
            "choices": [
                {
                    "finish_reason": (
                        "tool_calls" if message.get("tool_calls") else "stop"
                    ),
                    "message": message,
                }
            ]
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    def tool_reply(self, *, request: WireRequest) -> dict:
        self.tool_round += 1
        names = {tool.function.name for tool in request.tools}
        assert {"get_project", "prepare_project_comment"} <= names
        # Save joins the offered tools once a draft is pending.
        if self.tool_round >= 3:
            save_tool = next(
                tool.function
                for tool in request.tools
                if tool.function.name == "create_project_comment"
            )
            assert set(save_tool.parameters["properties"]) == {"draft_id", "confirm"}
            assert "person_id" not in save_tool.parameters["properties"]

        if self.tool_round == 1:
            return self.call(
                name="prepare_project_comment",
                arguments={"project_id": 57, "comment": NOTE},
            )
        if self.tool_round == 2:
            result_message = next(
                message for message in request.messages if message.role == "tool"
            )
            result = loads(result_message.content)["result"]
            self.draft_id = result["draft_id"]
            return {"content": "The note is ready for review."}
        if self.tool_round == 3:
            roles = [message.role for message in request.messages]
            assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
            _, _, preparation, receipt, delivered, approval = request.messages
            call, = preparation.tool_calls
            assert call.function.name == "prepare_project_comment"
            assert receipt.tool_call_id == call.id
            assert delivered.content == "The note is ready for review."
            assert approval.content == "Yes, save that exact note."
            stored_id = loads(receipt.content)["result"]["draft_id"]
            assert stored_id == self.draft_id
            return self.call(
                name="create_project_comment",
                arguments={"draft_id": stored_id, "confirm": True},
            )

        assert self.tool_round == 4
        result_message = next(
            message for message in reversed(request.messages) if message.role == "tool"
        )
        outcome = loads(result_message.content)
        text = (
            f"Saved as comment #{outcome['result']['comment_id']}."
            if outcome["saved"]
            else "The note remains pending."
        )
        return {"content": text}

    def call(self, *, name: str, arguments: dict) -> dict:
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{self.tool_round}",
                    "type": "function",
                    "function": {"name": name, "arguments": dumps(arguments)},
                }
            ],
        }


@pytest.fixture
def application_settings(monkeypatch, tmp_path):
    values = {
        "PROVIDER": "vllm",
        "CLASSIFIER_PROVIDER": "inherit",
        "TOOL_PROTOCOL": "openai",
        "RUNTIME_PROFILE": "olmo7b-instruct-vllm",
        "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "own",
        "PROJECT_COMMENT_APPROVAL_APPROVE_THRESHOLD": "0.75",
        "MAX_TOOL_CALLS": "3",
        "HISTORY_LIMIT": "10",
        "MAX_PROMPT_TOKENS": "8000",
        "SUMMARY_MAX_CHARS": "0",
        "COMPACTION_ENABLED": "0",
        "TURN_DECISION_MODE": "off",
        "ENABLE_SELF_REVIEW": "0",
        "SELF_REVIEW": "0",
        "SELF_REVIEW_DECIDER": "0",
        "ENABLE_RESPONSE_CONTRACTS": "off",
        "CHAT_QUEUE_ENABLED": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    for key, value in {
        "ISPEC_VLLM_MODEL": "synthetic-olmo",
        "ISPEC_VLLM_URL": "http://127.0.0.1:18099",
        "ISPEC_VLLM_TOOL_CALL_PARSER": "olmo3",
        "ISPEC_VLLM_STRICT_PROTOCOL": "1",
        "ISPEC_VLLM_TOOL_PREAMBLE_FAMILY": "assistant.tools.olmo3_preamble",
        "ISPEC_PROMPTS_DB_PATH": str(tmp_path / "prompts.db"),
        "ISPEC_INFERENCE_USAGE_LOG_ENABLED": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA", raising=False)
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA_PATH", raising=False)


def test_named_agent_can_prepare_through_controller(conversation, application_settings, monkeypatch, tmp_path):  # noqa: F811
    from ispec.db.models import AuthUser, UserRole

    class AgentModel(ScriptedModel):
        def tool_reply(self, *, request):
            names = {tool.function.name for tool in request.tools}
            assert "prepare_project_comment" in names and "create_project_comment" not in names
            self.tool_round += 1
            if self.tool_round == 1:
                return self.call(name="prepare_project_comment", arguments={"project_id": 57, "comment": NOTE})
            if self.tool_round == 2:
                return {"content": "The draft is ready."}
            assert '"requested_by": "dot"' in request.messages[0].content
            return {"content": "Dot requested it."}

    model = AgentModel()
    monkeypatch.setattr(service.requests, "post", model.post)
    agent = AuthUser(id=23, username="dot", role=UserRole.viewer, is_comment_agent=True, is_active=True)
    with get_schedule_session(tmp_path / "schedule.db") as schedule:
        def send(message, **fields):
            return support.chat(support.ChatRequest(message=message, **fields), assistant_db=conversation.assistant,
                                core_db=conversation.core, schedule_db=schedule, user=agent)

        prepared = send("I'm Dot. Draft a note for project 57.")
        assert prepared.sessionId.startswith("agent:23:")
        assert prepared.commentCards[-1].comment == NOTE
        answer = send("Who requested that draft?", sessionId=prepared.sessionId)
        assert answer.message == "Dot requested it."
    assert model.classifier_calls == 0 and conversation.comments() == []


@pytest.mark.parametrize("surface", ["web", "slack"])
@pytest.mark.parametrize(
    "label,confidence,saved",
    [
        ("approve_save", 0.75, True),
        ("approve_save", 0.749, False),
        ("deny_save", 0.99, False),
    ],
)
def test_real_controller_uses_application_olmo_prompts(
    conversation,  # noqa: F811 (pytest fixture)
    application_settings,
    monkeypatch,
    tmp_path,
    surface,
    label,
    confidence,
    saved,
):
    model = ScriptedModel(approval_label=label, confidence=confidence)
    monkeypatch.setattr(service.requests, "post", model.post)
    assert support.generate_reply is service.generate_reply
    user = conversation.user
    meta = None
    if surface == "slack":
        user = _ApiKeyServiceUser(can_write_project_comments=True)
        meta = {
            "source": "slack",
            "slack": {
                "user_id": "USYNTH",
                "team_id": "TSYNTH",
                "channel": "CSYNTH",
                "thread_ts": "1.0",
                "message_ts": "2.0",
            },
        }

    with get_schedule_session(tmp_path / "schedule.db") as schedule:
        for message in (
            "Draft a note for project 57: " + NOTE,
            "Yes, save that exact note.",
        ):
            response = support.chat(
                support.ChatRequest(
                    sessionId="app-" + surface, message=message, history=[], meta=meta
                ),
                assistant_db=conversation.assistant,
                core_db=conversation.core,
                schedule_db=schedule,
                user=user,
            )
            conversation.assistant.commit()

    # A valid approve_save is confirmed by the save follow-up question.
    assert model.classifier_calls == (2 if label == "approve_save" else 1)
    assert len(conversation.comments()) == int(saved)
    assert response.commentCards[-1].status == ("saved" if saved else "pending")
    assert response.commentCards[-1].comment == NOTE
    assert response.commentCards[-1].can_undo == saved
    if saved:
        assert conversation.comments()[0].com_Comment == NOTE

    preamble = load_prompt_source("assistant.tools.olmo3_preamble").body.strip()
    native_calls = [request for request in model.requests if request.tools]
    assert len(native_calls) == 4
    for request in native_calls:
        tool_names = {tool.function.name for tool in request.tools}
        application_prompt = service._render_system_prompt_planner(
            tools_available=True,
            tool_names=tool_names,
        ).text
        # OLMo's template attaches tool schemas to every system message, so
        # all system instructions arrive as one message.
        assert [message.role for message in request.messages].count("system") == 1
        assert request.messages[0].role == "system"
        assert request.messages[0].content.startswith(preamble + "\n" + application_prompt)
        assert "\n\nCONTEXT" in request.messages[0].content
        assert "staff and project-scoped clients" in application_prompt
        assert "Available tools:" not in application_prompt
        for message in request.messages:
            if message.role == "tool":
                result = loads(message.content)
                compatibility_copy = support.format_tool_result_message(result["tool"], result)
                assert compatibility_copy not in request.messages[0].content
    assert any(
        message.role == "assistant"
        and message.content == "The note is ready for review."
        for message in native_calls[2].messages
    )
    for request in model.requests:
        if not request.tools:
            assert (
                "built-in support assistant for the iSPEC web app"
                not in request.messages[0].content
            )
            assert preamble not in request.messages[0].content

    stored = conversation.assistant.get(SupportMessage, response.messageId)
    trace = loads(stored.meta_json)["llm_trace"]
    assert stored.content == response.message
    for step in trace:
        assert step["prompt_family"] == "assistant.planner.system"
        assert (
            step["provider_meta"]["tool_preamble"]["prompt_family"]
            == "assistant.tools.olmo3_preamble"
        )


@pytest.mark.parametrize("parser", ["olmo3", "llama3_json"])
def test_clarification_bypasses_policy_lookup_only_for_olmo(
    conversation, application_settings, monkeypatch, tmp_path, parser,  # noqa: F811
):
    monkeypatch.setenv("ISPEC_VLLM_TOOL_CALL_PARSER", parser)
    answer = "Which project do you mean? Please provide its ID or name."
    executed = []

    def post(url, *, json, **kwargs):
        if url.endswith("/tokenize"):
            data = {"count": 1000, "max_model_len": 8192}
        else:
            content = answer if json.get("tools") else dumps({
                "primary": "projects", "secondary": [], "confidence": 0.8, "clarify": True,
            })
            data = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    def record_tool(**kwargs):
        executed.append(kwargs["name"])
        return {"ok": True, "tool": kwargs["name"], "result": []}

    monkeypatch.setattr(service.requests, "post", post)
    monkeypatch.setattr(support, "run_tool", record_tool)
    with get_schedule_session(tmp_path / "schedule.db") as schedule:
        response = support.chat(
            support.ChatRequest(sessionId="missing-id", message="What is the status of my project?"),
            assistant_db=conversation.assistant, core_db=conversation.core,
            schedule_db=schedule, user=conversation.user,
        )
    assert response.message == answer
    assert executed == ([] if parser == "olmo3" else ["my_projects"])
