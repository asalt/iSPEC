"""Opt-in local inference only: never dispatch returned tools or contact Slack."""

import json
import os
from pathlib import Path
import time

import pytest
import requests

from ispec.assistant.service import AssistantReply
from ispec.assistant.slack_outbound import ConversationScope
from ispec.assistant.slack_thread_context import SlackThreadContext, ThreadMessage, render_thread_context
from ispec.assistant.tool_routing import route_tool_groups_vllm, tool_groups_for_available_tools
from ispec.assistant.tools import openai_tool_names_all

pytestmark = pytest.mark.skipif(os.getenv("ISPEC_RUN_VLLM_TESTS") != "1", reason="Opt-in local vLLM evaluation")


@pytest.mark.parametrize("user_message,expected", [
    ("make a note these were reported", "projects"),
    ("What was reported for this project?", "projects"),
    ("Search my previous assistant chats about BioID", "assistant"),
])
def test_same_thread_routing_with_loaded_local_model(user_message, expected, record_property):
    base = "http://127.0.0.1:8000/v1"
    models = requests.get(base + "/models", timeout=5)
    models.raise_for_status()
    model = models.json()["data"][0]["id"]
    fixture = json.loads((Path(__file__).parents[1] / "behavioral/data/slack_reported_thread.json").read_text())
    context = SlackThreadContext(
        scope=ConversationScope(team_id="TDEMO", channel="CDEMO", thread_ts="1.0", message_ts="10.0"),
        messages=tuple(ThreadMessage(inbox_id=i, user_id=row["user"], message_ts=f"{i}.0", text=row["text"])
                       for i, row in enumerate(fixture["messages"], 1)),
    )
    def generate(**kwargs):
        start = time.monotonic()
        response = requests.post(base + "/chat/completions", json={
            "model": model, "messages": kwargs["messages"], **kwargs["vllm_extra_body"],
        }, timeout=60)
        response.raise_for_status()
        data = response.json()
        record_property("elapsed_ms", round(1000 * (time.monotonic() - start)))
        record_property("usage", json.dumps(data.get("usage")))
        record_property("model", model)
        return AssistantReply(content=data["choices"][0]["message"].get("content") or "",
                              provider="vllm", model=model, meta=None)
    decision, _ = route_tool_groups_vllm(
        user_message=user_message, groups=tool_groups_for_available_tools(openai_tool_names_all()),
        generate_reply_fn=generate, context_messages=render_thread_context(context).messages,
    )
    assert decision is not None and decision["primary"] == expected
