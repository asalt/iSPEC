from __future__ import annotations

import json

from ispec.assistant.work_bag import (
    append_work_bag_entries,
    build_work_bag_entries_from_tool_calls,
    recent_work_bag_payload,
    work_bag_context_summary,
)


def test_work_bag_records_project_comment_success_without_raw_payloads():
    entries = build_work_bag_entries_from_tool_calls(
        tool_calls=[
            {
                "name": "create_project_comment",
                "arguments": {"comment": "full raw text should not be copied"},
                "ok": True,
                "error": None,
                "result_preview": json.dumps(
                    {
                        "project_id": 1499,
                        "comment_id": 555,
                        "person_id": 7,
                        "snippet": "Reanalysis completed.",
                        "legacy_push_enqueue": {"queued": True, "command_id": 12},
                    }
                ),
            }
        ],
        user_message_id=10,
        assistant_message_id=11,
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["kind"] == "write"
    assert entry["status"] == "succeeded"
    assert entry["tool_name"] == "create_project_comment"
    assert {"kind": "project", "id": 1499} in entry["refs"]
    assert {"kind": "project_comment", "id": 555} in entry["refs"]
    assert {"kind": "agent_command", "id": 12} in entry["refs"]
    assert "raw_arguments" in entry["omitted"]
    assert "full raw text" not in json.dumps(entry)


def test_work_bag_records_blocked_project_comment_without_comment_id():
    entries = build_work_bag_entries_from_tool_calls(
        tool_calls=[
            {
                "name": "create_project_comment",
                "arguments": {"project_id": 1499, "comment": "not copied"},
                "ok": False,
                "error": "User did not explicitly request saving to project history.",
                "result_preview": None,
            }
        ],
        user_message_id=20,
        assistant_message_id=21,
    )

    entry = entries[0]
    assert entry["status"] == "blocked"
    assert entry["refs"] == []
    assert not any(ref.get("kind") == "project_comment" for ref in entry["refs"])


def test_work_bag_append_caps_recent_entries_and_context_summarizes():
    state: dict[str, object] = {}
    entries = [
        {
            "entry_id": f"wb:a{i}:t1",
            "created_at": "2026-06-15T00:00:00+00:00",
            "source": "support_chat",
            "kind": "tool_call",
            "tool_name": "tool",
            "status": "succeeded" if i % 2 else "failed",
            "refs": [],
            "summary": f"entry {i}",
            "omitted": ["raw_arguments", "raw_result", "prompt_text"],
        }
        for i in range(15)
    ]

    assert append_work_bag_entries(state, entries, cap=12) is True
    payload = recent_work_bag_payload(state, session_id="s1", limit=5)
    assert payload["total_entries"] == 12
    assert [entry["entry_id"] for entry in payload["entries"]] == [
        "wb:a14:t1",
        "wb:a13:t1",
        "wb:a12:t1",
        "wb:a11:t1",
        "wb:a10:t1",
    ]

    summary = work_bag_context_summary(state)
    assert summary is not None
    assert summary["entry_count"] == 12
    assert len(summary["recent_entries"]) == 3
    assert summary["full_available_via_tool"] == "assistant_recent_session_work_bag"
