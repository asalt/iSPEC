import pytest

from scripts.run_model_regression import batches


def test_batches_cover_saved_suites_without_repeating_lookup_controls():
    rows = list(batches())
    names = [r[0] for r in rows]
    assert len(names) == len(set(names)) == 10
    assert {r[2] for r in rows if r[0].startswith("refinement-")} == {"dev", "held"}
    held = next(r for r in rows if r[0] == "refinement-held")
    assert held[4] == ["-k", "not test_project_lookup_controller"]
    assert {r[1] for r in rows if r[0].startswith("composition-")} == {"baseline", "concise", "compact"}
    assert names[-1] == "wording-jeff"


def test_save_contract_suite_only_selects_targeted_controller_cases():
    rows = list(batches("save-contract"))
    assert [r[0] for r in rows] == ["runtime-dev", "refinement-dev", "refinement-held", "wording-jeff"]
    assert all(r[1] == "baseline" for r in rows)
    assert "not approve_required_only" in rows[0][4][1]
    assert rows[1][4] == ["-k", "test_comment_controller and dev_approval"]
    assert rows[2][4] == ["-k", "test_comment_controller and held_explicit_person"]
    assert rows[3][4] == []


def test_unknown_suite_rejected():
    with pytest.raises(ValueError, match="Unknown regression suite"):
        list(batches("typo"))
