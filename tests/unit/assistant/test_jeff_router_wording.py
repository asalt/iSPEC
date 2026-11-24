"""Offline selection and frozen-input tests; fake predictions only."""

import json
from pathlib import Path

import pytest

from ispec.assistant.project_comment_workflow import COMMENT_WORKFLOW_TOOLS
from scripts.eval_jeff_router_wording import (
    BUDGET, EvaluationCase, NONE_VARIANTS, PROJECTS, build_catalogs,
    digest, pending_tools, prepare_schema_followup, score_one, variant_question,
)
from scripts.eval_jeff_tool_routing import MISC_DESCRIPTION
from scripts.jeff_router_catalog import select_categories


def test_feature_complete_staff_and_general_catalogs_filter_before_scoring():
    catalogs = build_catalogs()
    assert len(catalogs["staff"].groups) == 14
    assert set(catalogs["general_user"].groups) == {"projects", "notes", "files", "schedule", "misc"}
    for catalog in catalogs.values():
        question = variant_question(variant="N0", state="Save the draft", catalog=catalog)["question"]
        assert set(question["criteria"]) == catalog.groups.keys() | {"none"}
        assert set(catalog.directory_schema()["function"]["parameters"]["properties"]["name"]["enum"]) == catalog.groups.keys()


def test_variants_change_only_the_declared_choice_descriptions():
    catalog = build_catalogs()["staff"]
    baseline = variant_question(variant="baseline", state="Same context", catalog=catalog)
    revised = variant_question(variant="N0", state="Same context", catalog=catalog)
    expected = json.loads(json.dumps(baseline))
    expected["question"]["criteria"]["projects"] = PROJECTS
    assert revised == expected
    for name, description in NONE_VARIANTS.items():
        expected["question"]["criteria"]["none"] = description
        assert variant_question(variant=name, state="Same context", catalog=catalog) == expected


def test_historical_save_probability_now_keeps_projects():
    catalog = build_catalogs()["staff"]
    selected = select_categories(
        scores={"none": 0.815, "projects": 0.133, "files": 0.052}, catalog=catalog, budget=BUDGET,
    )
    assert selected.categories == ("projects",)
    assert selected.probability_mass == pytest.approx(0.948)


def test_pending_tools_always_present_even_when_none_exceeds_target():
    catalog = build_catalogs()["general_user"]
    case = EvaluationCase(id="test", category="none", state="Leave pending", pending_draft=True, genuine_none=True, provenance="synthetic")
    selected = select_categories(
        scores={"none": 0.95, "projects": 0.05}, catalog=catalog, budget=BUDGET,
        always_tool_names=pending_tools(case=case, catalog=catalog),
    )
    assert selected.categories == ()
    assert set(selected.tool_names) == COMMENT_WORKFLOW_TOOLS
    assert selected.tool_count == 5


def test_pending_tools_are_deduplicated_against_project_selection():
    catalog = build_catalogs()["staff"]
    selected = select_categories(
        scores={"none": 0.1, "projects": 0.9}, catalog=catalog, budget=BUDGET,
        always_tool_names=COMMENT_WORKFLOW_TOOLS,
    )
    assert set(selected.tool_names) == catalog.groups["projects"]
    assert selected.tool_count == len(catalog.groups["projects"]) + 1


def test_unavailable_always_tools_are_rejected():
    catalog = build_catalogs()["general_user"]
    with pytest.raises(ValueError, match="authorized"):
        select_categories(scores={"none": 1}, catalog=catalog, budget=BUDGET, always_tool_names=frozenset({"get_person"}))


def test_fixture_is_balanced_hand_authored_and_has_ten_genuine_none():
    path = Path(__file__).resolve().parents[2] / "behavioral/data/jeff_router_wording_20261113.json"
    fixture = json.loads(path.read_text())
    cases = fixture["cases"]
    assert fixture["provenance"] == "synthetic/hand-authored"
    assert fixture["purpose"] == "evaluation_only_not_training"
    assert len(cases) == len({case["id"] for case in cases}) == 36
    assert sum(case.get("genuine_none", False) for case in cases) == 10
    for category in ("schedule", "files", "omics", "people", "experiments", "misc"):
        assert sum(case["category"] == category for case in cases) == 3
    assert sum(case["category"] == "projects" for case in cases) == 5


def test_exact_input_cache_reuses_only_identical_questions_and_context(tmp_path):
    class Model:
        calls = 0

        def predict(self, *, rows, batch_size):
            self.calls += 1
            return [[0.8, 0.2]]

    model = Model()
    row = {"state": "Prior draft + save it", "question": {"criteria": {"projects": "Drafts", "none": "Direct reply"}}}
    results = {}
    assert score_one(model=model, row=row, checkpoint="test", results=results, output=tmp_path)
    assert not score_one(model=model, row=row, checkpoint="test", results=results, output=tmp_path)
    assert model.calls == 1
    row["state"] = "Different context"
    assert score_one(model=model, row=row, checkpoint="test", results=results, output=tmp_path)
    row["question"]["criteria"]["none"] = "No action"
    assert score_one(model=model, row=row, checkpoint="test", results=results, output=tmp_path)
    assert model.calls == 3
    assert len(json.loads((tmp_path / "results.json").read_text())) == 3


def test_schema_followup_changes_only_misc_and_reuses_exact_baselines(tmp_path):
    from dataclasses import asdict

    source = tmp_path / "prior"
    source.mkdir()
    catalog = build_catalogs()["staff"]
    original = variant_question(variant="N3", state="", catalog=catalog)["question"]
    original["criteria"]["misc"] = "Other utilities (API schema search, DB file stats, recent activity)."
    cases = [asdict(EvaluationCase(
        id=identifier, category=category, state=f"Synthetic request {identifier}",
        pending_draft=False, genuine_none=category == "none", provenance="synthetic/hand-authored",
    )) for identifier, category in (
        ("M01", "misc"), ("M02", "misc"), ("M03", "misc"),
        ("E01", "experiments"), ("E02", "experiments"), ("E03", "experiments"),
        ("P02", "projects"), ("Z03", "none"), ("UNRELATED", "files"),
    )]
    manifest = {
        "checkpoint": "test", "cases": cases,
        "questions": {"staff": {"N3": original}}, "catalogs": {"staff": {}},
        "selection_costs": {"staff": {}},
    }
    source_text = json.dumps(manifest)
    (source / "manifest.json").write_text(source_text)
    scores = {
        digest({"checkpoint": "test", "row": {"state": case["state"], "question": original}}):
            {"scores": {"none": 1}, "seconds": 1}
        for case in cases
    }
    (source / "results.json").write_text(json.dumps(scores))

    output = tmp_path / "focused"
    prepare_schema_followup(source=source, output=output)

    focused = json.loads((output / "manifest.json").read_text())
    assert set(focused["questions"]) == {"staff"}
    assert len(focused["cases"]) == 10
    assert "UNRELATED" not in {case["id"] for case in focused["cases"]}
    assert {case["id"] for case in focused["cases"][-2:]} == {"API_FRESH", "RUN_FRESH"}
    assert focused["questions"]["staff"]["N3_original"] == original
    expected = json.loads(json.dumps(original))
    expected["criteria"]["misc"] = MISC_DESCRIPTION
    assert focused["questions"]["staff"]["N3_schema"] == expected
    assert len(json.loads((output / "results.json").read_text())) == 8
    assert (source / "manifest.json").read_text() == source_text
