"""Synthetic bookkeeping checks; no model loading or inference."""

import json
from types import SimpleNamespace

import pytest

from scripts.eval_jeff_tool_routing import (
    DraftContext, RecordedTurn, RouterInput, collect_inputs, conversation_context,
    expected_category, recorded_turns, record_prediction, resume_results, routing_question, score_categories,
    threshold_curve,
)
from scripts.jeff_router_catalog import SelectionBudget, ToolCatalog, select_categories


def test_one_question_contains_available_categories_and_none(catalog):
    row = routing_question(state="User: Show project 57", catalog=catalog)
    assert row["state"] == "User: Show project 57"
    question = row["question"]
    assert set(question["criteria"]) == catalog.groups.keys() | {"none"}
    assert question["criteria"]["projects"] == "Find projects by number, title, topic, PI or lab contact; list and count them; read project details and history."
    assert "none" in question["criteria"]
    assert "in this conversation, or none?" in question["instructions"]


def test_general_user_question_keeps_none_and_hides_staff_categories(catalog):
    client = ToolCatalog.from_authorized_schemas(schemas=list(catalog.schemas.values()), staff=False)
    choices = routing_question(state="Save that one", catalog=client)["question"]["criteria"]

    assert set(choices) == {"projects", "notes", "files", "none"}
    assert "none" in choices
    assert {"assistant", "devops", "tmux", "bridge", "repo", "automation"}.isdisjoint(choices)


class RecordingModel:
    def __init__(self):
        self.calls = []

    def predict(self, *, rows, batch_size):
        self.calls.append(rows)
        assert len(rows) == batch_size == 1
        choices = rows[0]["question"]["criteria"]
        return [[1.0 if name == "projects" else 0.0 for name in choices]]


def test_scoring_uses_one_model_input_for_all_probabilities(catalog):
    model = RecordingModel()

    scores = score_categories(model=model, state="Previous draft: ...\nUser: Save that one", catalog=catalog)

    assert len(model.calls) == 1
    assert set(scores) == catalog.groups.keys() | {"none"}
    assert sum(scores.values()) == 1
    assert scores["projects"] == 1


def test_completed_scores_are_saved_before_the_next_request(tmp_path, catalog):
    case = RouterInput(key="case", case="P01_explicit", state="Save that one", expected="projects", messages=[], sources=[])

    record_prediction(model=RecordingModel(), case=case, threads=1, results={}, output=tmp_path, catalog=catalog)

    saved = json.loads((tmp_path / "results.json").read_text())[case.key]
    assert saved["scores"]["projects"] == 1
    assert saved["threads"] == 1


def test_labels_distinguish_control_and_topic_change():
    assert expected_category(case="C04_greeting", latest="Hi") is None
    assert expected_category(case="N17_unrelated_question", latest="which day is the staff meeting?") == "schedule"
    assert expected_category(case="N17_unrelated_question", latest="Draft a project note") == "projects"


def test_dedup_keeps_provenance_and_incomplete_baseline(tmp_path):
    messages = [
        {"role": "system", "content": "JSON router instructions"},
        {"role": "user", "content": 'User request: "Look up project 57"'},
    ]
    call = {
        "request": {"messages": messages, "structured_outputs": {"json": {"properties": {"primary": {}}}}},
        "response": {"choices": [{"finish_reason": "length", "message": {"content": '{"primary":'}}]},
        "elapsed_ms": 1000,
    }
    for name in ("case-a", "case-b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "wire.json").write_text(json.dumps({"calls": [call]}))
        (folder / "conversation.json").write_text(json.dumps({
            "case": "C01_project", "turns": [{"user": "Look up project 57", "assistant": "Future result", "drafts": []}],
        }))

    cases = collect_inputs(roots=[tmp_path])

    assert len(cases) == 1
    case = cases[0]
    assert len(case.sources) == 2
    assert case.sources[0]["decision"] is None
    assert case.messages == messages
    assert case.state == "Latest user request:\nLook up project 57"
    assert case.expected == "projects"


@pytest.fixture
def exchanges():
    pending = DraftContext(
        draft_id="draft-one", project_id=57, status="pending", comment_id=None,
        text="Seven samples arrived; analysis is pending.",
    )
    saved = DraftContext(
        draft_id="draft-one", project_id=57, status="saved", comment_id=999,
        text="Seven samples arrived; analysis is pending.",
    )
    return (
        RecordedTurn(user="Draft a project 57 note about the seven samples.", assistant="Here is the draft. Save it?", drafts=(pending,)),
        RecordedTurn(user="Save that one", assistant="Future answer: saved 999.", drafts=(saved,)),
    )


def test_followup_contains_preceding_exchange_and_prior_draft_but_no_future(exchanges):
    context = conversation_context(messages=[], turns=exchanges, turn_index=1)

    assert exchanges[0].user in context
    assert exchanges[0].assistant in context
    assert exchanges[0].drafts[0].text in context
    assert "status=pending; project=57" in context
    assert context.endswith("Latest user request:\nSave that one")
    assert context.count("Save that one") == 1
    assert "Future answer" not in context
    assert "saved" not in context
    assert "999" not in context


def test_first_request_does_not_see_its_future_draft_or_reply(exchanges):
    context = conversation_context(messages=[], turns=exchanges, turn_index=0)

    assert context == f"Latest user request:\n{exchanges[0].user}"


def test_aborted_current_turn_uses_request_without_inventing_a_reply():
    turns = recorded_turns(conversation={
        "turns": [], "case_definition": {"messages": ["Which project?", "Unreached future request"]},
    })

    assert len(turns) == 1
    assert turns[0].assistant is None
    assert conversation_context(messages=[], turns=turns, turn_index=0) == "Latest user request:\nWhich project?"


def test_incomplete_previous_exchange_is_rejected():
    turns = (
        RecordedTurn(user="First request", assistant=None, drafts=()),
        RecordedTurn(user="Save that one", assistant=None, drafts=()),
    )
    with pytest.raises(ValueError, match="preceding exchange"):
        conversation_context(messages=[], turns=turns, turn_index=1)


def test_context_retains_original_request_and_two_recent_exchanges():
    turns = tuple(
        RecordedTurn(user=f"request-{index}", assistant=f"answer-{index}", drafts=())
        for index in range(5)
    )
    messages = [{"role": "system", "content": "Router instructions"},
                {"role": "system", "content": "Quoted Slack source"},
                {"role": "user", "content": 'User request: "request-4"'}]

    context = conversation_context(messages=messages, turns=turns, turn_index=4)

    assert "request-0" in context
    assert "request-1" not in context
    assert "answer-2" in context and "answer-3" in context
    assert "answer-4" not in context
    assert "Quoted Slack source" in context
    assert "Router instructions" not in context


def schema(name):
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object"}}}


@pytest.fixture
def catalog():
    return ToolCatalog.from_authorized_schemas(
        schemas=[schema(name) for name in (
            "get_project", "prepare_project_comment", "create_project_comment", "get_person",
            "project_files_for_project", "assistant_list_tools", "assistant_stats", "assistant_capture_tmux_pane",
        )],
        staff=True,
    )


def test_cumulative_target_not_individual_probability_cutoff(catalog):
    scores = {"projects": 0.65, "people": 0.25, "files": 0.06, "none": 0.04}
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=3, max_tools=10)

    selection = select_categories(scores=scores, catalog=catalog, budget=budget)

    assert selection.categories == ("projects", "people")
    assert selection.probability_mass == pytest.approx(0.9)
    assert selection.stop_reason == "probability_target"
    assert selection.tool_count == 5


def test_cumulative_target_tolerates_floating_point_roundoff(catalog):
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=3, max_tools=10)

    selection = select_categories(scores={"projects": 0.7, "people": 0.2, "files": 0.1, "none": 0}, catalog=catalog, budget=budget)

    assert selection.categories == ("projects", "people")
    assert selection.stop_reason == "probability_target"


@pytest.mark.parametrize("max_categories,max_tools,expected,reason", [
    (1, 10, ("projects",), "category_cap"),
    (5, 4, ("projects",), "tool_cap"),
    (5, 3, ("projects",), "tool_cap"),
])
def test_budget_caps_additions_but_keeps_top_category(catalog, max_categories, max_tools, expected, reason):
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=max_categories, max_tools=max_tools)
    scores = {"projects": 0.65, "people": 0.25, "files": 0.06, "none": 0.04}

    selection = select_categories(scores=scores, catalog=catalog, budget=budget)

    assert selection.categories == expected
    assert selection.stop_reason == reason
    assert selection.tool_count <= max_tools or selection.categories == ("projects",)


def test_none_is_not_a_veto_below_cumulative_target(catalog):
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=3, max_tools=10)

    none = select_categories(scores={"projects": 0.4, "none": 0.6}, catalog=catalog, budget=budget)
    partial = select_categories(scores={"projects": 0.6, "none": 0.4}, catalog=catalog, budget=budget)

    assert none.categories == ("projects",)
    assert none.tool_count == 4
    assert none.stop_reason == "probability_target"
    assert partial.categories == ("projects",)
    assert partial.probability_mass == 1
    assert partial.stop_reason == "probability_target"


def test_none_point_three_stops_selection_without_unnecessary_categories(catalog):
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=5, max_tools=24)

    selection = select_categories(
        scores={"projects": 0.6, "none": 0.3, "people": 0.07, "files": 0.03}, catalog=catalog, budget=budget,
    )

    assert selection.categories == ("projects",)
    assert selection.probability_mass == pytest.approx(0.9)
    assert selection.stop_reason == "probability_target"


def test_hidden_mass_is_rejected_because_filtering_precedes_scoring(catalog):
    client = ToolCatalog.from_authorized_schemas(schemas=list(catalog.schemas.values()), staff=False)
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=5, max_tools=24)

    with pytest.raises(ValueError, match="before scoring"):
        select_categories(
            scores={"assistant": 0.4, "projects": 0.35, "none": 0.19, "files": 0.06}, catalog=client, budget=budget,
        )


def test_single_dominant_category_survives_its_own_tool_cap(catalog):
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=3, max_tools=1)

    selection = select_categories(scores={"projects": 0.95, "none": 0.05}, catalog=catalog, budget=budget)

    assert selection.categories == ("projects",)
    assert selection.tool_count == 4
    assert selection.stop_reason == "probability_target"


def test_none_at_target_returns_only_directory(catalog):
    client = ToolCatalog.from_authorized_schemas(schemas=list(catalog.schemas.values()), staff=False)
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=3, max_tools=10)

    selection = select_categories(scores={"projects": 0.1, "none": 0.9}, catalog=client, budget=budget)

    assert selection.categories == ()
    assert selection.stop_reason == "probability_target"


def test_missing_none_is_rejected(catalog):
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=3, max_tools=10)
    with pytest.raises(ValueError, match="none choice"):
        select_categories(scores={"projects": 1}, catalog=catalog, budget=budget)


def test_directory_replaces_old_discovery_and_reveals_only_authorized_tools(catalog):
    directory = catalog.directory_schema()["function"]

    assert directory["name"] == "open_tool_category"
    assert "assistant_list_tools" not in catalog.schemas
    assert {spec["function"]["name"] for spec in catalog.open_category(name="projects")} == {
        "get_project", "prepare_project_comment", "create_project_comment",
    }
    with pytest.raises(ValueError, match="Unavailable"):
        catalog.open_category(name="devops")


def test_client_directory_hides_staff_categories_even_if_supplied(catalog):
    client = ToolCatalog.from_authorized_schemas(schemas=list(catalog.schemas.values()), staff=False)
    names = client.directory_schema()["function"]["parameters"]["properties"]["name"]["enum"]

    assert set(names) == {"projects", "notes", "files"}
    assert "get_person" not in client.schemas
    assert "get_project" not in client.schemas
    for name in ("assistant", "devops", "tmux", "bridge", "repo", "automation"):
        with pytest.raises(ValueError, match="Unavailable"):
            client.open_category(name=name)


def test_overlapping_groups_count_each_tool_once(catalog):
    budget = SelectionBudget(cumulative_probability=0.9, max_categories=3, max_tools=2)

    selection = select_categories(scores={"tmux": 0.5, "bridge": 0.4, "none": 0.1}, catalog=catalog, budget=budget)

    assert selection.categories == ("tmux", "bridge")
    assert selection.tool_count == 2


def test_curve_reports_directory_recovery_and_costs(catalog):
    case = RouterInput(key="case", case="C02_person", state="Look up person 73", expected="people", messages=[], sources=[])
    scores = {"projects": 0.7, "people": 0.2, "none": 0.1}
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: SimpleNamespace(ids=list(text)))

    curve = threshold_curve(cases=[case], results={case.key: {"scores": scores}}, catalog=catalog, tokenizer=tokenizer)

    tight = next(row for row in curve if row["cumulative_probability"] == 0.7 and row["max_categories"] == 1)
    broad = next(row for row in curve if row["cumulative_probability"] == 0.95 and row["max_categories"] == 3)
    assert tight["directory_needed"] == tight["directory_recoverable"] == 1
    assert broad["expected_included"] == 1
    assert broad["directory_needed"] == 0
    assert broad["mean_tool_count"] > tight["mean_tool_count"]
    assert broad["mean_schema_json_tokens"] > tight["mean_schema_json_tokens"]


def test_resume_reuses_only_matching_context_and_role(tmp_path, catalog):
    case = RouterInput(key="saved", case="P01_explicit", state="User: Save that one", expected="projects", messages=[], sources=[])
    manifest = {
        "checkpoint": str(tmp_path.resolve()),
        "cases": [{"key": case.key, "state": case.state}],
        "passes": {"staff": {
            "question": routing_question(state="", catalog=catalog),
            "directory_groups": {name: sorted(tools) for name, tools in catalog.groups.items()},
        }},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    staff = tmp_path / "staff"
    staff.mkdir()
    (staff / "results.json").write_text(json.dumps({case.key: {"scores": {"projects": 0.9, "none": 0.1}, "seconds": 2.0}}))

    recovered = resume_results(directory=tmp_path, pass_name="staff", staff=True, cases=[case], checkpoint=tmp_path, catalog=catalog)

    assert recovered[case.key]["seconds"] == 2.0
    assert recovered[case.key]["resumed_from"] == str(tmp_path.resolve())
    changed = RouterInput(key="saved", case=case.case, state="A different preceding exchange", expected="projects", messages=[], sources=[])
    assert resume_results(directory=tmp_path, pass_name="staff", staff=True, cases=[changed], checkpoint=tmp_path, catalog=catalog) == {}
    assert resume_results(directory=tmp_path, pass_name="general_user", staff=False, cases=[case], checkpoint=tmp_path, catalog=catalog) == {}
    client = ToolCatalog.from_authorized_schemas(schemas=list(catalog.schemas.values()), staff=False)
    with pytest.raises(ValueError, match="question differs"):
        resume_results(directory=tmp_path, pass_name="staff", staff=False, cases=[case], checkpoint=tmp_path, catalog=client)
