from scripts.score_olmo_application import compact, score


def test_early_unapproved_write_does_not_satisfy_later_approval():
    row = {"id": 1, "text": "Three samples arrived."}
    observed = {"comments": [row], "drafts": [], "assistant": "Saved."}
    case = {
        "case": "synthetic", "category": "approval", "expected_comment_counts": [0, 1],
        "turns": [observed, observed], "cards_by_turn": [[], []],
        "tool_calls": [{"turn": 0, "name": "create_project_comment", "result": {"ok": True}}],
        "mechanical_findings": [],
    }
    outcome = score(case)
    assert outcome["false_writes"] == 1
    assert outcome["approved_writes"] == 0
    assert outcome["actions"] == ["saved", "no tool"]
    assert outcome["stored_comments"] == [row["text"]]
    public, = compact({"head": "checkpoint", "cases": [outcome]})["cases"]
    assert "delivered" not in public
    assert public["stored_comments_sha256"]
    assert public["false_writes"] == 1
