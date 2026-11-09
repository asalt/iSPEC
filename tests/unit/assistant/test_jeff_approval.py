from dataclasses import replace
import json

import pytest
import requests

from ispec.assistant.jeff_approval import (
    ApprovalPromptVariant,
    ApprovalTrialSettings,
    JeffContractError,
    JeffUnavailable,
    LocalJeffEndpoint,
    PendingNoteExchange,
    evaluate_pending_approval,
    lexical_pending_approval,
    prepare_approval_request,
    parse_approval_response,
    BINARY_LABELS,
)
from scripts.eval_jeff_approval import (
    evaluate_case,
    load_cases,
    load_short_cases,
    main,
    short_prompt_variants,
    load_choice_cases,
    choice_variants,
    summarize,
)


EXCHANGE = PendingNoteExchange(
    project_id=42,
    note="Samples arrived; analysis is pending.",
    initial_request="Make a note for project 42: samples arrived; analysis is pending.",
    assistant_draft="Draft: Samples arrived; analysis is pending. Shall I save this exact note?",
    latest_reply="confirm",
)


def response(label="yes", probabilities=None):
    return {
        "answers": {
            "save_approval": {
                "type": "choice",
                "choice": label,
                "probabilities": probabilities
                or {
                    key: 0.9 if key == label else 0.05
                    for key in ("yes", "no", "unsure")
                },
                "confidence": 0.85,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def never(*args, **kwargs):
    pytest.fail("No fallback, network request or tool execution is allowed here")


def test_context_and_question_are_separate_and_only_three_messages_are_sent():
    wire = prepare_approval_request(EXCHANGE)
    assert wire["state"]["messages"] == [
        {"role": "user", "content": EXCHANGE.initial_request},
        {"role": "assistant", "content": EXCHANGE.assistant_draft},
        {"role": "user", "content": EXCHANGE.latest_reply},
    ]
    instructions = wire["questions"]["save_approval"]["instructions"]
    assert "without revisions or unmet conditions" in instructions
    assert EXCHANGE.initial_request not in instructions
    assert EXCHANGE.assistant_draft not in instructions
    assert set(wire["questions"]["save_approval"]["criteria"]) == {
        "yes",
        "no",
        "unsure",
    }
    assert "tools" not in wire


@pytest.mark.parametrize("label", ["yes", "no", "unsure"])
def test_valid_result_never_falls_back_and_cannot_claim_a_write(label):
    result = evaluate_pending_approval(
        EXCHANGE, query=lambda _: response(label), lexical_fallback=never
    )
    assert result.decision == label
    assert result.would_save == (label == "yes")
    assert result.to_dict()["write_outcome"] == {"status": "not_attempted"}
    assert result.to_dict()["shadow_only"] is True


@pytest.mark.parametrize(
    "probabilities",
    [
        {"yes": 0.54, "no": 0.43, "unsure": 0.03},
        {"yes": 0.43, "no": 0.54, "unsure": 0.03},
        {"yes": 0.50, "no": 0.50, "unsure": 0.00},
    ],
)
def test_low_confidence_or_tie_abstains_without_fallback(probabilities):
    label = max(probabilities, key=probabilities.get)
    result = evaluate_pending_approval(
        EXCHANGE,
        query=lambda _: response(label, probabilities),
        lexical_fallback=never,
    )
    assert result.decision == "unsure"
    assert not result.would_save


def test_only_unavailability_can_invoke_lexical_fallback():
    calls = []

    def query(_):
        raise JeffUnavailable("offline")

    def fallback(reply):
        calls.append(reply)
        return "yes"

    result = evaluate_pending_approval(EXCHANGE, query=query, lexical_fallback=fallback)
    assert calls == [EXCHANGE.latest_reply]
    assert result.source == "lexical" and result.would_save
    assert result.probabilities is None  # Do not fabricate classifier confidence.
    disabled = evaluate_pending_approval(
        EXCHANGE,
        query=query,
        lexical_fallback=never,
        settings=ApprovalTrialSettings(allow_lexical_fallback=False),
    )
    assert not disabled.would_save


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("confirm", "yes"),
        ("YES!", "yes"),
        ("save it", "yes"),
        ("no", "no"),
        ("yes but shorten it", "unsure"),
        ("yes after checking", "unsure"),
        ("confirm tomorrow", "unsure"),
        ("don't save", "unsure"),
    ],
)
def test_lexical_fallback_is_a_narrow_whole_reply_protocol(reply, expected):
    assert lexical_pending_approval(reply) == expected


@pytest.mark.parametrize(
    "change",
    [
        {"pending": False},
        {"pending": "true"},
        {"project_id": 0},
        {"project_id": True},
        {"note": "A different note"},
        {"latest_reply": ""},
        {"initial_request": ""},
        {"initial_request": None},
    ],
)
def test_invalid_pending_exchange_never_queries_or_falls_back(change):
    result = evaluate_pending_approval(
        replace(EXCHANGE, **change), query=never, lexical_fallback=never
    )
    assert not result.would_save and result.source == "none"


def test_over_limit_is_rejected_not_truncated_or_fallback_authorized():
    result = evaluate_pending_approval(
        EXCHANGE,
        query=never,
        lexical_fallback=never,
        settings=ApprovalTrialSettings(max_context_chars=10),
    )
    assert result.reason == "context_too_large" and not result.would_save


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"answers": []},
        {"answers": {"save_approval": None}},
        response("yes", {"yes": float("nan"), "no": 0.1, "unsure": 0.1}),
        response("yes", {"yes": True, "no": 0.0, "unsure": 0.0}),
        response("yes", {"yes": 0.1, "no": 0.8, "unsure": 0.1}),
        response("yes", {"yes": 0.7, "no": 0.7, "unsure": 0.1}),
        response("yes", {"yes": 0.9, "no": 0.1}),
    ],
)
def test_invalid_contract_never_uses_fallback(bad):
    result = evaluate_pending_approval(
        EXCHANGE, query=lambda _: bad, lexical_fallback=never
    )
    assert result.reason == "jeff_contract_error" and not result.would_save


@pytest.mark.parametrize(
    "url",
    [
        "https://slack.com",
        "http://example.com:8765",
        "http://127.0.0.1",
        "http://user:pass@127.0.0.1:8765",
        "http://127.0.0.1:8765/other",
        "http://127.0.0.1:8765?target=external",
    ],
)
def test_endpoint_is_loopback_only(url):
    with pytest.raises(ValueError):
        LocalJeffEndpoint(url=url)


@pytest.mark.parametrize(
    "status,error_type",
    [
        (503, JeffUnavailable),
        (529, JeffUnavailable),
        (401, JeffContractError),
        (403, JeffContractError),
        (422, JeffContractError),
        (302, JeffContractError),
        (500, JeffContractError),
    ],
)
def test_transport_errors_are_distinct_and_redirects_are_not_followed(
    monkeypatch, status, error_type
):
    calls = []

    def request(session, method, url, **kwargs):
        assert session.trust_env is False
        assert kwargs["allow_redirects"] is False
        assert not kwargs.get("headers") and not kwargs.get("auth")
        calls.append((method, url))
        reply = requests.Response()
        reply.status_code = status
        return reply

    monkeypatch.setattr(requests.Session, "request", request)
    with pytest.raises(error_type):
        LocalJeffEndpoint(url="http://127.0.0.1:18765")({})
    assert calls == [("POST", "http://127.0.0.1:18765/v1/systemone")]


@pytest.mark.parametrize("error", [requests.ConnectionError, requests.Timeout])
def test_transport_unavailability_has_no_retry(monkeypatch, error):
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise error("private details must not be copied to the receipt")

    monkeypatch.setattr(requests.Session, "request", request)
    with pytest.raises(JeffUnavailable, match=error.__name__):
        LocalJeffEndpoint(url="http://127.0.0.1:18765")({})
    assert len(calls) == 1


def test_transport_has_no_executor_paths():
    with pytest.raises(ValueError):
        LocalJeffEndpoint(url="http://127.0.0.1:18765").request(
            method="POST", path="/api/comments"
        )


def test_frozen_corpus_contains_length_controls_and_keeps_expectations_out_of_requests():
    cases = load_cases()
    assert len({case["id"] for case in cases}) == len(cases) == 35
    assert {case["split"] for case in cases} == {
        "prior",
        "development",
        "held",
        "length",
    }
    for case in cases:
        if case["exchange"].pending:
            assert case["exchange"].gate_error() is None
        request = prepare_approval_request(case["exchange"])
        assert "expected_label" not in json.dumps(request)
        assert len(request["state"]["messages"]) == 3


def test_report_does_not_hide_false_approvals():
    case = next(case for case in load_cases() if case["id"] == "held_cancel")
    result = evaluate_case(case, lambda _: response("yes"), ApprovalTrialSettings())
    assert result["false_approval"]
    summary = summarize([result])
    assert summary["false_approvals"] == ["held_cancel"]
    assert summary["write_count"] == 0


def test_request_matches_installed_jeff_client_without_model_dependencies():
    client = pytest.importorskip("jeff.client")
    request = prepare_approval_request(EXCHANGE)
    question = request["questions"]["save_approval"]
    official = client.build_request(
        request["state"],
        {
            "save_approval": client.choice_question(
                question["criteria"],
                question["instructions"],
            )
        },
        "jeff-latest",
        None,
        None,
    )
    assert official == request


def test_layout_probe_changes_only_the_intended_factors():
    request = prepare_approval_request(EXCHANGE)
    original = json.loads(json.dumps(request))
    assert (
        prepare_approval_request(EXCHANGE, variant=ApprovalPromptVariant()) == original
    )
    changed = prepare_approval_request(
        EXCHANGE,
        variant=ApprovalPromptVariant(context="transcript", choices="descriptive"),
    )
    assert request == original
    for message in request["state"]["messages"]:
        assert message["content"] in changed["state"]
    assert (
        changed["questions"]["save_approval"]["instructions"]
        == request["questions"]["save_approval"]["instructions"]
    )


@pytest.mark.parametrize("variant", short_prompt_variants())
def test_short_variants_preserve_choices_and_full_message_content(variant):
    request = prepare_approval_request(EXCHANGE, variant=variant)
    baseline = prepare_approval_request(EXCHANGE)
    assert (
        request["questions"]["save_approval"]["criteria"]
        == baseline["questions"]["save_approval"]["criteria"]
    )
    serialized = json.dumps(request["state"])
    for value in (
        EXCHANGE.initial_request,
        EXCHANGE.assistant_draft,
        EXCHANGE.latest_reply,
    ):
        assert value in serialized
    question = request["questions"]["save_approval"]["instructions"]
    if variant.question != "baseline":
        assert len(question.split()) <= 18
        assert "Use the earlier" not in question
    assert "tools" not in request


@pytest.mark.parametrize("context", ["object", "messages", "live_last", "transcript"])
def test_templates_preserve_quotes_newlines_and_literal_dollar_signs(context):
    text = 'A "quoted" note\nwith $latest_reply and backslash \\ intact.'
    exchange = replace(
        EXCHANGE,
        note=text,
        assistant_draft=text,
        latest_reply='Save $note, "exactly".\nThanks.',
    )
    state = prepare_approval_request(
        exchange, variant=ApprovalPromptVariant(context=context)
    )["state"]
    if context == "object":
        messages = state["messages"]
        assert messages[1]["content"] == text
        assert messages[2]["content"] == exchange.latest_reply
    elif context == "messages":
        assert state[1]["content"] == text
        assert state[2]["content"] == exchange.latest_reply
    elif context == "live_last":
        assert state["assistant_draft"] == text
        assert list(state)[-1] == "latest_user_reply"
        assert state["latest_user_reply"] == exchange.latest_reply
    else:
        assert text in state
        assert state.endswith(exchange.latest_reply)


def test_jeff_native_layout_is_verified_without_loading_weights():
    model = pytest.importorskip("jeff.model")
    for context in ("object", "messages", "live_last", "transcript"):
        request = prepare_approval_request(
            EXCHANGE, variant=ApprovalPromptVariant(context=context)
        )
        rendered = model.decision_messages(
            {
                "state": request["state"],
                "question": request["questions"]["save_approval"],
            },
            ["A", "B", "C"],
            layout="live-last",
        )
        assert [message["role"] for message in rendered] == ["system", "user"]
        text = rendered[-1]["content"][0]["text"]
        if context == "live_last":
            latest = text.split("Latest:\n", 1)[1]
            assert EXCHANGE.latest_reply in latest
            assert EXCHANGE.assistant_draft not in latest
            assert EXCHANGE.initial_request not in latest
        elif context == "object":
            assert EXCHANGE.assistant_draft in text.split("Latest:\n", 1)[1]
        else:
            assert "Latest:\n" not in text
            assert (
                text.index("State:\n")
                < text.index("Question:\n")
                < text.index("Options:\n")
            )


def test_short_sweep_is_fixed_and_separates_seen_and_fresh_cases():
    cases = load_short_cases()
    assert len({case["id"] for case in cases}) == len(cases) == 18
    assert sum(case["split"] == "fresh" for case in cases) == 12
    assert len(short_prompt_variants()) == 16
    assert all(variant.choices == "original" for variant in short_prompt_variants())
    for case in cases:
        assert case["exchange"].gate_error() is None
        assert "expected_label" not in json.dumps(
            prepare_approval_request(case["exchange"])
        )
        assert not case["id"].startswith("length_")


def test_preview_needs_no_endpoint_and_renders_selected_files(monkeypatch, capsys):
    monkeypatch.setattr(LocalJeffEndpoint, "request", never)
    assert (
        main(
            [
                "--preview",
                "example_confirm",
                "--question",
                "authorize",
                "--context",
                "live_last",
            ]
        )
        == 0
    )
    request = json.loads(capsys.readouterr().out)
    assert (
        request["questions"]["save_approval"]["instructions"]
        == "Does the user's last reply authorize saving the project comment?"
    )
    assert request["state"]["latest_user_reply"] == "confirm"


@pytest.mark.parametrize(
    "change", [{"question": "unknown"}, {"context": "unknown"}, {"choices": "unknown"}]
)
def test_variant_names_are_validated(change):
    with pytest.raises(ValueError):
        ApprovalPromptVariant(**change)


def test_binary_variant_requires_exact_binary_response_not_relaxed_schema():
    variant = ApprovalPromptVariant(
        question="authorize", context="reply", choices="binary"
    )
    wire = prepare_approval_request(EXCHANGE, variant=variant)
    assert wire["state"] == EXCHANGE.latest_reply
    baseline = prepare_approval_request(EXCHANGE)["questions"]["save_approval"][
        "criteria"
    ]
    assert wire["questions"]["save_approval"]["criteria"] == {
        key: baseline[key] for key in BINARY_LABELS
    }
    good = response("yes", {"yes": 0.8, "no": 0.2})
    assert evaluate_pending_approval(
        EXCHANGE, query=lambda _: good, variant=variant
    ).would_save
    assert (
        evaluate_pending_approval(
            EXCHANGE, query=lambda _: response(), variant=variant
        ).reason
        == "jeff_contract_error"
    )
    assert (
        evaluate_pending_approval(EXCHANGE, query=lambda _: good).reason
        == "jeff_contract_error"
    )
    with pytest.raises(ValueError):
        parse_approval_response(good, labels=("yes",))


def test_choice_corpus_has_matched_context_controls_and_separate_fresh_split():
    cases = load_choice_cases()
    assert len(cases) == len({case["id"] for case in cases}) == 36
    assert sum(case["split"] == "fresh" for case in cases) == 18
    assert len(choice_variants()) == 8
    by_id = {case["id"]: case for case in cases}
    save = by_id["choice_looks_good_save"]
    wording = by_id["choice_looks_good_wording"]
    assert save["exchange"].latest_reply == wording["exchange"].latest_reply
    assert save["expected_label"] != wording["expected_label"]
    assert {
        case["id"]
        for case in cases
        if case.get("evaluation_role") == "forced_gate_boundary"
    } == {"choice_looks_good_wording", "choice_short_yes_keep_draft"}
    for case in cases:
        assert case["exchange"].gate_error() is None
    for variant in choice_variants():
        result = evaluate_case(
            wording,
            lambda _: response("no", {"yes": 0.1, "no": 0.9})
            if variant.choices == "binary"
            else response("unsure"),
            ApprovalTrialSettings(),
            variant=variant,
        )
        assert result["raw_label_correct"] and not result["false_approval"]


@pytest.mark.parametrize(
    "case_id,eligible",
    [
        ("choice_looks_good_wording", False),
        ("choice_short_yes_keep_draft", False),
        ("choice_agreement_thanks", True),
    ],
)
def test_real_gate_is_distinct_from_forced_classifier_context(case_id, eligible):
    from ispec.assistant.project_comment_approval import (
        ProjectCommentApprovalSettings,
        build_project_comment_gate,
        detect_project_comment_trigger,
    )
    from ispec.assistant.reply_interpretation import (
        interpret_reply_for_project_comment_save,
    )

    exchange = next(
        case["exchange"] for case in load_choice_cases() if case["id"] == case_id
    )
    reply = interpret_reply_for_project_comment_save(
        tool_protocol="openai",
        available_tool_names={"create_project_comment"},
        focused_project_id=exchange.project_id,
        last_assistant_message=exchange.assistant_draft,
        user_message=exchange.latest_reply,
    )
    gate = build_project_comment_gate(
        settings=ProjectCommentApprovalSettings(mode="own"),
        trigger=detect_project_comment_trigger(
            message=exchange.latest_reply,
            legacy_confirmation_kind=reply.kind,
            legacy_save_requested=False,
        ),
        tool_protocol="openai",
        available_tool_names={"create_project_comment"},
        focused_project_id=exchange.project_id,
        session_id="fixture",
        thread_key="fixture",
        current_turn_id="1",
        prior_assistant_message_id=1,
        prior_assistant_message=exchange.assistant_draft,
        user_message=exchange.latest_reply,
        awaiting_reply_state=reply.awaiting_state_dict(),
    )
    assert gate.eligible is eligible
    assert exchange.pending is True  # Deliberate classifier-only stress condition.


def test_runner_stops_on_first_transport_error(monkeypatch, tmp_path):
    calls = []

    def request(self, *, method, path, payload=None):
        if path == "/health":
            return {"status": "ready"}
        calls.append(payload)
        raise JeffUnavailable("simulated_disconnect")

    monkeypatch.setattr(LocalJeffEndpoint, "request", request)
    output = tmp_path / "run"
    assert main(["--choice-sweep", "--output-dir", str(output)]) == 1
    assert len(calls) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["complete"]
    assert summary["cases"] == 1 and summary["planned_cases"] == 288


def test_runner_can_finish_only_selected_cases_without_repeating_the_corpus(
    monkeypatch, tmp_path
):
    calls = []

    def request(self, *, method, path, payload=None):
        if path == "/health":
            return {"status": "ready"}
        calls.append(payload)
        labels = payload["questions"]["save_approval"]["criteria"]
        values = {label: 0.1 / (len(labels) - 1) for label in labels}
        values["yes"] = 0.9
        return response("yes", values)

    monkeypatch.setattr(LocalJeffEndpoint, "request", request)
    output = tmp_path / "selected"
    assert (
        main(
            ["--choice-sweep", "--case", "example_confirm", "--output-dir", str(output)]
        )
        == 0
    )
    rows = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert len(calls) == len(rows) == 8
    assert {row["id"] for row in rows} == {"example_confirm"}
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] and summary["planned_cases"] == 8
