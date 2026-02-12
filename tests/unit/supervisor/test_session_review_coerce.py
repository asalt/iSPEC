from __future__ import annotations

from ispec.supervisor.loop import _SESSION_REVIEW_VERSION, _coerce_session_review_output


def test_session_review_coerce_accepts_review_notes_variant():
    parsed = {
        "review_notes": {
            "summary": "The assistant missed a tool call.",
            "issues": [
                {
                    "severity": "warning",
                    "category": "tool_use",
                    "description": "Did not call count_all_projects.",
                    "evidence_message_ids": [101, "102", "bad"],
                    "suggested_fix": "Use the count tool.",
                }
            ],
            "followups": "Double-check project counts.",
            "repo_search_queries": ["count_all_projects", ""],
        }
    }

    coerced = _coerce_session_review_output(parsed, session_id="s1", target_message_id=12)
    assert isinstance(coerced, dict)
    assert coerced["schema_version"] == _SESSION_REVIEW_VERSION
    assert coerced["session_id"] == "s1"
    assert coerced["target_message_id"] == 12
    assert coerced["summary"]
    assert coerced["followups"] == ["Double-check project counts."]

    issues = coerced["issues"]
    assert isinstance(issues, list) and issues
    assert issues[0]["severity"] == "warning"
    assert issues[0]["category"] == "tool_use"
    assert issues[0]["evidence_message_ids"] == [101, 102]
    assert issues[0]["suggested_fix"] == "Use the count tool."


def test_session_review_coerce_accepts_legacy_review_keys():
    parsed = {
        "review": {
            "missed_tool_opportunities": ["Should have used get_project."],
            "incorrect_claims": ["Said there was 1 project."],
            "followups": ["Confirm project totals."],
        }
    }

    coerced = _coerce_session_review_output(parsed, session_id="s2", target_message_id=55)
    assert isinstance(coerced, dict)
    assert coerced["schema_version"] == _SESSION_REVIEW_VERSION
    assert coerced["session_id"] == "s2"
    assert coerced["target_message_id"] == 55
    assert any(issue["category"] == "tool_use" for issue in coerced["issues"])
    assert any(issue["category"] == "accuracy" for issue in coerced["issues"])
    assert coerced["followups"] == ["Confirm project totals."]

