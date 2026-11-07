import json
from xml.etree import ElementTree

import pytest

from scripts.review_staged_conversations import load_reviews, main, render_review, review_signals


@pytest.fixture
def evidence(tmp_path):
    folder = tmp_path / "case0"
    folder.mkdir()
    case = {
        "model": "synthetic", "case": "denial", "expected_comment_counts": [0],
        "turns": [{"user": "Do not save", "assistant": "<script>alert('x')</script>",
                   "metadata": {"project_comment_workflow": {"save_checks": [{"label": "deny_save"}]}},
                   "comments": [], "drafts": [], "elapsed_ms": 1}],
        "tool_calls": [{"turn": 0, "name": "create_project_comment", "result": {"ok": False}}],
    }
    (folder / "conversation.json").write_text(json.dumps(case))
    (folder / "wire.json").write_text(json.dumps({"calls": [
        {"path": "/v1/chat/completions", "request": {"messages": [{"role": "system", "content": "exact prompt"}]},
         "response": {"content": "raw saved claim"}, "elapsed_ms": 1},
    ]}))
    (tmp_path / "casecurrent").symlink_to(folder, target_is_directory=True)
    return tmp_path


def write_junit(*, root, outcome=None, duplicate=False):
    suite = ElementTree.Element("testsuite")
    for _ in range(2 if duplicate else 1):
        test = ElementTree.SubElement(suite, "testcase", name="test_staged_conversation[denial]")
        if outcome:
            ElementTree.SubElement(test, outcome, message="Grounding assertion").text = "Detailed assertion evidence"
    ElementTree.ElementTree(suite).write(root / "results.xml")


def test_review_preserves_evidence_and_escapes_model_text(evidence):
    html = render_review(roots=[evidence])
    assert html.count('<details class="case-panel" id="case-') == 1
    assert "<script>alert(" not in html and "&lt;script&gt;" in html
    assert "exact prompt" in html and "raw saved claim" in html and "deny_save" in html
    assert 'data-result="unknown"' in html
    assert 'data-attention="true"' in html
    assert "Save attempted on a non-save turn" in html
    assert "No manual semantic verdict recorded" in html
    assert 'id="case-index"' in html and 'id="footnotes"' in html
    assert html.index('id="footnotes"') > html.index('id="evidence"')


@pytest.mark.parametrize("outcome,status", [(None, "pass"), ("failure", "fail"), ("error", "error"), ("skipped", "skipped")])
def test_recorded_test_outcome_is_not_inferred_from_counts(evidence, outcome, status):
    write_junit(root=evidence, outcome=outcome)
    review, = load_reviews(roots=[evidence], notes=None)
    assert review.test.status == status
    assert review.actual == (0,)
    html = render_review(roots=[evidence])
    assert f'data-result="{status}"' in html and "Match</td>" in html
    if outcome:
        assert "Detailed assertion evidence" in html


@pytest.mark.parametrize("problem", ["absent", "invalid", "duplicate", "wrong-case"])
def test_missing_or_ambiguous_junit_never_becomes_pass(evidence, problem):
    if problem == "invalid":
        (evidence / "results.xml").write_text("not XML")
    elif problem == "duplicate":
        write_junit(root=evidence, duplicate=True)
    elif problem == "wrong-case":
        (evidence / "results.xml").write_text('<testsuite><testcase name="something_else"/></testsuite>')
    review, = load_reviews(roots=[evidence], notes=None)
    assert review.test.status == "unknown"


def test_search_does_not_use_junit_outside_requested_root(evidence):
    write_junit(root=evidence.parent)
    review, = load_reviews(roots=[evidence], notes=None)
    assert review.test.status == "unknown"


def test_manual_findings_do_not_change_pytest_result(evidence):
    write_junit(root=evidence)
    notes = evidence / "notes.json"
    notes.write_text(json.dumps([{"batch": evidence.name, "model": "synthetic", "case": "denial",
                                 "findings": ["<b>Misleading reply</b>"], "source": "reviewed fixture"}]))
    html = render_review(roots=[evidence], notes=notes)
    assert 'data-result="pass"' in html and 'data-attention="true"' in html
    assert "&lt;b&gt;Misleading reply&lt;/b&gt;" in html and "reviewed fixture" in html
    assert "Passed with flags (1)" in html


def test_blocked_attempt_is_distinct_from_write_and_guard_is_only_a_signal(evidence):
    case = json.loads((evidence / "case0/conversation.json").read_text())
    case["turns"][0]["metadata"]["write_claim_guard"] = {"triggered": True}
    signals = review_signals(case=case)
    assert "Save attempted on a non-save turn" in signals
    assert "Write on a non-save turn" not in signals
    assert "Response guard intervened" in signals
    case["turns"][0]["comments"] = [{"id": 1}]
    assert "Write on a non-save turn" in review_signals(case=case)


def test_cli_overwrites_only_with_explicit_option(evidence):
    output = evidence / "review.html"
    args = [str(evidence), "--output", str(output)]
    main(args)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == before
    main([*args, "--overwrite"])
    assert output.read_bytes() == before
    with pytest.raises(SystemExit):
        main([str(evidence), "--output", str(evidence / "conversation.json"), "--overwrite"])


def test_empty_review_is_not_a_successful_report(tmp_path):
    with pytest.raises(ValueError, match="No conversation"):
        render_review(roots=[tmp_path])


def test_promotion_junit_and_diagnostics_are_distinct(evidence):
    (evidence / "results.xml").write_text(
        '<testsuite><testcase name="test_promotion_conversation[denial]"/></testsuite>')
    path = evidence / "case0/conversation.json"
    case = json.loads(path.read_text())
    case["negative_checks"] = [{"source": "diagnostic_no_execution", "write_ticket": {"issued": True}}]
    path.write_text(json.dumps(case))
    review, = load_reviews(roots=[evidence], notes=None)
    assert review.test.status == "pass"
    html = render_review(roots=[evidence])
    assert "Negative approval coverage (actual vs diagnostic)" in html
    assert "diagnostic_no_execution" in html and "never execute their tickets" in html
