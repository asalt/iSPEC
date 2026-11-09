"""Real inference, synthetic threads, temporary databases; no external tools."""

import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import requests

from ispec.api.routes import support
from ispec.api.security import _ApiKeyServiceUser
from ispec.assistant import service
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.slack_inbox import SlackInbox
from ispec.db.connect import get_session
from ispec.db.models import Project, ProjectComment
from ispec.prompt import render_prompt
from ispec.schedule.connect import get_schedule_session
from scripts.prompt_composition_trial import (
    layout_summary,
    merge_adjacent_system_messages,
)
from tests.behavioral.test_model_runtime_followup_live import isolated_runtime

pytestmark = [
    pytest.mark.behavioral,
    pytest.mark.skipif(
        os.getenv("ISPEC_RUN_LLM_TESTS") != "1",
        reason="Explicit isolated inference only",
    ),
]
CASES = json.loads(
    (Path(__file__).parent / "data/prompt_composition_cases.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_composed_thread_continuity(isolated_runtime, monkeypatch, case):
    store = isolated_runtime
    trial = os.environ["ISPEC_INSTRUCT_TRIAL"]
    assert trial in {"baseline", "concise", "compact"}
    if trial != "baseline":
        monkeypatch.setattr(
            service,
            "_render_base_system_prompt",
            lambda: render_prompt("experimental.assistant.base.concise"),
        )
    base_text = service._render_base_system_prompt().text
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_ALLOWED_WRITE_TOOLS", "create_project_comment")
    monkeypatch.setenv("ISPEC_VLLM_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("ISPEC_INFERENCE_USAGE_LOG_ENABLED", "1")
    original_generate = support.generate_reply
    layouts, wire, executed = [], [], []
    current_request = ""

    def generate(**kwargs):
        messages = kwargs["messages"]
        if trial == "compact":
            messages = merge_adjacent_system_messages(messages)
        layouts.append(
            layout_summary(
                messages, base_text=base_text, current_request=current_request
            )
        )
        return original_generate(**{**kwargs, "messages": messages})

    monkeypatch.setattr(support, "generate_reply", generate)
    guarded_request = requests.sessions.Session.request

    def record_request(self, method, url, **kwargs):
        # Only the synthetic JSON body is retained, never headers/environment.
        body = copy.deepcopy(kwargs.get("json", {}))
        record = {
            "request": body,
            "base_occurrences": sum(
                str(message.get("content") or "").count(base_text)
                for message in body.get("messages", [])
            ),
        }
        start = time.monotonic()
        try:
            response = guarded_request(self, method, url, **kwargs)
            if url.endswith("/v1/chat/completions"):
                record.update(status=response.status_code, response=response.json())
            return response
        finally:
            if url.endswith("/v1/chat/completions"):
                record["elapsed_ms"] = round((time.monotonic() - start) * 1000)
                wire.append(record)

    monkeypatch.setattr(requests.sessions.Session, "request", record_request)
    original_tool = support.run_tool

    def fixture_tool(**kwargs):
        assert kwargs["name"] in {
            "get_project",
            "projects",
            "search_projects",
            "latest_project_comments",
            "create_project_comment",
        }
        assert str(kwargs["core_db"].bind.url.database) == str(store.core_db_path)
        if case["lookup_fails"] and kwargs["name"] in {"get_project", "projects"}:
            result = {
                "ok": False,
                "tool": kwargs["name"],
                "error": "synthetic_database_unavailable",
            }
        else:
            result = original_tool(**kwargs)
        executed.append(
            {
                "name": kwargs["name"],
                "args": copy.deepcopy(kwargs["args"]),
                "result": result,
            }
        )
        return result

    monkeypatch.setattr(support, "run_tool", fixture_tool)
    inbox = SlackInbox(file=store.assistant_db_path)
    rows = [
        {"ts": f"{10 + index}.0", "thread_ts": "10.0", "user": "UAUTHOR", "text": text}
        for index, text in enumerate(case["messages"])
    ]
    errors = []
    with (
        get_session(store.core_db_path) as core,
        get_assistant_session(store.assistant_db_path) as adb,
        get_schedule_session(store.schedule_db_path) as sdb,
    ):
        if case["project_id"] is not None:
            core.add(
                Project(
                    id=case["project_id"],
                    prj_AddedBy="fixture",
                    prj_ProjectTitle=case["title"],
                )
            )
            core.commit()
        for turn_number, turn in enumerate(case["turns"]):
            current_request = turn["request"]
            ts = f"{20 + turn_number * 2}.0"
            event = {
                "type": "app_mention",
                "channel": "CSYNTH",
                "user": "USTAFF",
                "ts": ts,
                "thread_ts": "10.0",
                "text": current_request,
            }
            receipt = inbox.receive(
                {"team_id": "TSYNTH", "event": event}, {"bot_user_id": "UBOT"}
            )
            meta = {
                "source": "slack",
                "slack_inbox_id": receipt,
                "slack": {
                    "team_id": "TSYNTH",
                    "channel": "CSYNTH",
                    "thread_ts": "10.0",
                    "message_ts": ts,
                },
            }
            meta["slack_thread_context"] = inbox.thread_context(
                meta,
                client=SimpleNamespace(
                    conversations_replies=lambda **kwargs: {
                        "ok": True,
                        "messages": list(rows),
                    }
                ),
            )
            wire_start, layout_start, tool_start = (
                len(wire),
                len(layouts),
                len(executed),
            )
            start = time.monotonic()
            response = support.chat(
                support.ChatRequest(
                    sessionId="composition-" + case["id"],
                    message=current_request,
                    history=[],
                    meta=meta,
                ),
                assistant_db=adb,
                core_db=core,
                schedule_db=sdb,
                user=_ApiKeyServiceUser(can_write_project_comments=True),
            )
            metadata = json.loads(adb.get(SupportMessage, response.messageId).meta_json)
            comments = core.query(ProjectComment).count()
            calls = executed[tool_start:]
            turn_errors = []
            if any(call.get("status") != 200 for call in wire[wire_start:]):
                turn_errors.append("transport_failure")
            if (metadata.get("provider_meta") or {}).get("error"):
                turn_errors.append("provider_contract_failure")
            if comments or (
                (metadata.get("project_comment_approval_eval") or {}).get(
                    "write_ticket"
                )
                or {}
            ).get("issued"):
                turn_errors.append("unauthorized_write_or_ticket")
            if any(call["name"] == "create_project_comment" for call in calls):
                turn_errors.append("unrequested_write_tool")
            if turn["tool_expectation"] == "none" and calls:
                turn_errors.append("unrequested_tool")
            if turn["tool_expectation"] == "lookup" and not any(
                call["name"] in {"get_project", "projects"} for call in calls
            ):
                turn_errors.append("missing_lookup")
            text = response.message.casefold()
            for group in turn["required_any"]:
                if not any(word.casefold() in text for word in group):
                    turn_errors.append("missing:" + "/".join(group))
            for phrase in turn["forbidden"]:
                if phrase.casefold() in text:
                    turn_errors.append("unwanted:" + phrase)
            record = {
                "case": case["id"],
                "trial": trial,
                "turn": turn_number,
                "request": current_request,
                "response": response.message,
                "elapsed_ms": round((time.monotonic() - start) * 1000),
                "executed": calls,
                "comment_count": comments,
                "metadata": metadata,
                "layouts": layouts[layout_start:],
                "wire": wire[wire_start:],
                "errors": turn_errors,
            }
            print("\nCOMPOSITION_EVAL " + json.dumps(record, default=str), flush=True)
            errors.extend(f"turn {turn_number}: {error}" for error in turn_errors)
            # The real generated answer becomes next-turn evidence, not an expected answer.
            rows.extend(
                [
                    {
                        "ts": ts,
                        "thread_ts": "10.0",
                        "user": "USTAFF",
                        "text": current_request,
                    },
                    {
                        "ts": f"{21 + turn_number * 2}.0",
                        "thread_ts": "10.0",
                        "user": "UBOT",
                        "text": response.message,
                    },
                ]
            )
            # Match the request-scoped lifecycle before the independent inbox
            # writer receives another turn. No transaction spans turns.
            adb.commit()
            core.commit()
            sdb.commit()
            if comments:
                break  # Preserve the failure and stop this synthetic conversation.
    assert not errors, errors
