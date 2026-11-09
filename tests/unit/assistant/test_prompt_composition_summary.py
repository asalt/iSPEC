import json

from scripts.summarize_prompt_composition_trials import read_records, summarize


def test_summary_separates_router_from_main_request_and_preserves_failures(tmp_path):
    record = {
        "case": "lookup",
        "turn": 0,
        "response": "The database is unavailable.",
        "errors": ["missing_lookup"],
        "elapsed_ms": 1200,
        "comment_count": 0,
        "executed": [],
        "metadata": {"context": {"history_messages_used": 2}},
        "layouts": [
            {"base_occurrences": 0, "roles": ["system", "user"]},
            {"base_occurrences": 1, "roles": ["system", "system", "user"]},
        ],
        "wire": [
            {
                "base_occurrences": 0,
                "response": {"usage": {"prompt_tokens": 50, "completion_tokens": 10}},
            },
            {
                "base_occurrences": 1,
                "response": {"usage": {"prompt_tokens": 200, "completion_tokens": 20}},
            },
            {"base_occurrences": 1, "elapsed_ms": 900},
        ],
    }
    path = tmp_path / "pytest.log"
    path.write_text("pytest noise\nCOMPOSITION_EVAL " + json.dumps(record) + "\n")
    assert read_records(path) == [record]
    report = summarize([path])[0]
    assert report["recorded_turns"] == 1
    assert report["passing_turns"] == 0
    turn = report["turns"][0]
    assert turn["first_input_tokens"] == 50
    assert turn["first_main_input_tokens"] == 200
    assert turn["first_main_system_messages"] == 2
    assert turn["input_tokens"] == 250
    assert turn["output_tokens"] == 30
    assert turn["model_calls"] == 3
    assert turn["errors"] == ["missing_lookup"]
    assert turn["history_messages"] == 2


def test_empty_log_is_not_reported_as_a_success(tmp_path):
    path = tmp_path / "pytest.log"
    path.write_text("startup failed before any recorded turn\n")
    report = summarize([path])[0]
    assert report["recorded_turns"] == report["passing_turns"] == 0
    assert report["median_ms"] is None
    assert report["turns"] == []
