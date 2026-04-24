from __future__ import annotations

import json
from pathlib import Path

from ispec.assistant.support_benchmark import (
    assert_case_expectations,
    format_benchmark_summary,
    load_local_benchmark_scenarios,
    load_synthetic_benchmark_scenarios,
    summarize_benchmark_results,
)


def test_load_synthetic_benchmark_scenarios_renders_placeholders_and_filters(tmp_path: Path) -> None:
    path = tmp_path / "scenarios.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenarios": [
                    {
                        "label": "lookup-one",
                        "family": "lookup",
                        "tags": ["read_only"],
                        "turns": [
                            {
                                "message": "does project ${lookup_project_id} exist?",
                                "expect": {"final_assistant_contains": ["project ${lookup_project_id}"]},
                            }
                        ],
                    },
                    {
                        "label": "tmux-one",
                        "family": "tmux",
                        "turns": [{"message": "show me ${tmux_alias}"}],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_synthetic_benchmark_scenarios(
        path=path,
        variables={"lookup_project_id": 1596, "tmux_alias": "ispec"},
        labels={"lookup-one"},
    )

    assert len(scenarios) == 1
    assert scenarios[0]["label"] == "lookup-one"
    assert scenarios[0]["turns"][0]["message"] == "does project 1596 exist?"
    assert scenarios[0]["turns"][0]["expect"]["final_assistant_contains"] == ["project 1596"]


def test_load_local_benchmark_scenarios_converts_user_turns_and_final_expect(tmp_path: Path) -> None:
    case_path = tmp_path / "0244-example.json"
    case_path.write_text(
        json.dumps(
            {
                "label": "project-note-review-flow",
                "tags": ["project-note", "slack"],
                "messages": [
                    {"id": 1, "role": "user", "content": "draft a note"},
                    {"id": 2, "role": "assistant", "content": "Would you like me to save it?"},
                    {"id": 3, "role": "user", "content": "no don't save it"},
                ],
                "expect": {"final_assistant_not_contains": ["Comment ID"]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_local_benchmark_scenarios(case_selectors=[str(case_path)], root=tmp_path)

    assert len(scenarios) == 1
    assert scenarios[0]["family"] == "project-note"
    assert [turn["message"] for turn in scenarios[0]["turns"]] == ["draft a note", "no don't save it"]
    assert scenarios[0]["final_expect"] == {"final_assistant_not_contains": ["Comment ID"]}


def test_assert_case_expectations_uses_final_assistant_meta() -> None:
    case = {
        "messages": [
            {"id": 1, "role": "user", "content": "question"},
            {
                "id": 2,
                "role": "assistant",
                "content": "I did not save it.",
                "assistant_meta": {
                    "reply_interpretation": {"runtime_action": "deny_save"},
                    "tool_calls": [{"name": "assistant_list_tmux_panes"}],
                },
            },
        ]
    }

    assert_case_expectations(
        case,
        {
            "final_assistant_contains": ["did not save"],
            "reply_interpretation_runtime_action": "deny_save",
            "tool_call_names_include": ["assistant_list_tmux_panes"],
        },
    )


def test_summarize_benchmark_results_groups_by_family() -> None:
    summary = summarize_benchmark_results(
        [
            {
                "label": "s1",
                "family": "lookup",
                "turns": [
                    {"ok": True, "wall_clock_ms": 800, "model_elapsed_ms_total": 500, "model_call_count": 1, "tool_call_count": 1},
                    {"ok": False, "wall_clock_ms": 1200, "model_elapsed_ms_total": 700, "model_call_count": 2, "tool_call_count": 1},
                ],
            },
            {
                "label": "s2",
                "family": "tmux",
                "turns": [
                    {"ok": True, "wall_clock_ms": 1500, "model_elapsed_ms_total": 900, "model_call_count": 2, "tool_call_count": 1},
                ],
            },
        ]
    )

    assert summary["scenario_count"] == 2
    assert summary["turn_count"] == 3
    assert summary["successful_turns"] == 2
    assert summary["family_summary"]["lookup"]["turn_count"] == 2
    assert summary["family_summary"]["lookup"]["wall_clock_ms_p50"] == 800
    assert "Family	Scenarios" in format_benchmark_summary(summary)
