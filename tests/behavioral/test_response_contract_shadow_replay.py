from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from ispec.assistant.service import AssistantReply


pytestmark = pytest.mark.behavioral


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts" / "eval_response_contracts_shadow.py"
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT_PATH))
evaluate_cases = SCRIPT_GLOBALS["evaluate_cases"]
load_cases = SCRIPT_GLOBALS["load_cases"]


def test_behavioral_response_contract_shadow_replay_fixture_produces_expected_report():
    fixture_path = Path(__file__).resolve().parent / "data" / "response_contract_shadow_cases.json"
    cases = load_cases(fixture_path)

    calls: list[dict[str, object]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_):  # type: ignore[no-untyped-def]
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})
        selector_turn = len(calls) % 2 == 1
        case_index = (len(calls) - 1) // 2
        label = str(cases[case_index].get("label") or "")
        if selector_turn:
            contract = "direct" if label == "gratitude_close" else "brief_explainer"
            return AssistantReply(
                content=json.dumps(
                    {
                        "contract": contract,
                        "confidence": 0.9,
                        "reason": f"Selected for {label}.",
                    }
                ),
                provider="test",
                model="selector",
            )
        if label == "gratitude_close":
            return AssistantReply(
                content=json.dumps({"answer": "Thanks for the note."}),
                provider="test",
                model="fill",
            )
        return AssistantReply(
            content=json.dumps(
                {
                    "answer": "The model over-answers because the reply shape is not constrained.",
                    "reason": "Without a bounded answer skeleton, it keeps adding support.",
                }
            ),
            provider="test",
            model="fill",
        )

    report = evaluate_cases(cases, generate_reply_fn=fake_generate_reply)

    assert len(report) == 2
    gratitude = report[0]
    explainer = report[1]

    assert gratitude["label"] == "gratitude_close"
    assert gratitude["selected_contract"] == "direct"
    assert gratitude["shadow_candidate"] == "Thanks for the note."
    assert gratitude["would_apply_if_live"] is False
    assert gratitude["protection_reason"] == "social_close_experiment"

    assert explainer["label"] == "brief_explainer"
    assert explainer["selected_contract"] == "brief_explainer"
    assert explainer["shadow_candidate"] == (
        "The model over-answers because the reply shape is not constrained.\n\n"
        "Why: Without a bounded answer skeleton, it keeps adding support."
    )
    assert explainer["would_apply_if_live"] is True
    assert explainer["protection_reason"] is None
