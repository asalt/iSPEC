from copy import deepcopy
import json

import pytest

from ispec.assistant.jeff_approval import (
    ApprovalPromptVariant,
    ApprovalTrialSettings,
    LocalJeffEndpoint,
)
from scripts.eval_jeff_approval import evaluate_case, load_choice_cases
from scripts.jeff_approval_review import (
    checkpoint,
    export_checkpoints,
    followup_request,
    load_checkpoint,
    main,
    messages_digest,
    replay_checkpoint,
    threshold_sweep,
    would_approve,
    write_json,
)


def reply(probabilities):
    return {
        "model": "synthetic-jeff",
        "answers": {
            "save_approval": {
                "type": "choice",
                "choice": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }
        },
    }


def example():
    case = load_choice_cases()[0]
    return evaluate_case(
        case,
        lambda _: reply({"yes": 0.64, "no": 0.21, "unsure": 0.15}),
        ApprovalTrialSettings(),
        variant=ApprovalPromptVariant(question="authorize", context="live_last"),
    )


def test_strict_half_and_tie_rejection():
    assert not would_approve(
        {"yes": 0.5, "no": 0.3, "unsure": 0.2}, threshold=0.5, strict=True
    )
    assert would_approve(
        {"yes": 0.501, "no": 0.3, "unsure": 0.199}, threshold=0.5, strict=True
    )
    assert not would_approve({"yes": 0.5, "no": 0.5}, threshold=0)


def test_margin_is_redundant_at_point_seven_five_not_at_lower_threshold():
    for yes in (0.75, 0.8, 0.99):
        for share in (0, 0.5, 1):
            p = {"yes": yes, "no": (1 - yes) * share, "unsure": (1 - yes) * (1 - share)}
            assert would_approve(p, threshold=0.75, margin=0.2) == would_approve(
                p, threshold=0.75
            )
    p = {"yes": 0.51, "no": 0.48, "unsure": 0.01}
    assert would_approve(p, threshold=0.5, strict=True)
    assert not would_approve(p, threshold=0.5, strict=True, margin=0.2)


def test_threshold_report_counts_false_approvals_and_keeps_raw_scores():
    row = example()
    original = deepcopy(row)
    report = threshold_sweep([row])
    low = next(
        x
        for x in report["results"]
        if x["split"] == "all" and x["threshold"] == 0.5 and x["margin"] == 0
    )
    high = next(
        x
        for x in report["results"]
        if x["split"] == "all" and x["threshold"] == 0.75 and x["margin"] == 0
    )
    assert low["false_approvals"] == [row["id"]]
    assert not high["false_approvals"]
    assert row == original
    assert report["write_count"] == 0


def test_before_and_after_checkpoints_preserve_exact_request_without_label_leakage(
    tmp_path,
):
    row = example()
    before = checkpoint(row, boundary="before", source="synthetic.jsonl", run={})
    after = checkpoint(row, boundary="after", source="synthetic.jsonl", run={})
    assert before["response"] is None and before["policy_result"] is None
    assert after["request"] == row["wire"][0]["request"] == before["request"]
    assert after["response"] == row["wire"][0]["response"]
    assert len(after["messages"]) == 3
    assert "expected_label" not in json.dumps(after)
    path = tmp_path / "checkpoint.json"
    write_json(path, after)
    assert load_checkpoint(path) == after
    modified = deepcopy(after)
    modified["request"]["state"]["latest_user_reply"] = "tampered"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="Checkpoint changed"):
        load_checkpoint(path)


def test_export_has_readable_transcripts_and_refuses_overwriting(tmp_path):
    row = example()
    source = tmp_path / "results.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = tmp_path / "checkpoints"
    export_checkpoints(source, arm=row["arm"], case_ids=[row["id"]], output=output)
    text = (output / row["id"] / "conversation.md").read_text()
    assert row["exchange"]["initial_request"] in text
    assert (output / row["id"] / "before.json").exists()
    with pytest.raises(FileExistsError):
        export_checkpoints(source, arm=row["arm"], case_ids=[], output=output)


def test_replay_only_rescores_frozen_request():
    row = example()
    packet = checkpoint(
        row,
        boundary="after",
        source="synthetic",
        run={"health": {"model": "synthetic-jeff"}},
    )
    calls = []

    class Endpoint:
        def request(self, **kwargs):
            assert kwargs == {"method": "GET", "path": "/health"}
            return {"status": "ready", "model": "synthetic-jeff"}

        def __call__(self, request):
            calls.append(request)
            return reply({"yes": 0.6, "no": 0.3, "unsure": 0.1})

    result = replay_checkpoint(packet, Endpoint())
    assert calls == [packet["request"]]
    assert result["write_outcome"] == {"status": "not_attempted"}
    assert (
        packet["response"]["answers"]["save_approval"]["probabilities"]["yes"] == 0.64
    )


def test_followup_is_no_tools_review_and_continues_matched_response():
    packet = checkpoint(example(), boundary="after", source="synthetic", run={})
    request = followup_request(
        packet, ask="Could the preceding save question explain this?"
    )
    assert set(request[0]) == {"id", "messages", "max_tokens"}
    assert "expected_label" not in json.dumps(request)
    response = {
        "results": [
            {
                "id": request[0]["id"],
                "messages_sha256": messages_digest(request[0]["messages"]),
                "ok": True,
                "content": "That is a hypothesis we can test.",
                "model": "review-model",
            }
        ]
    }
    new = followup_request(
        packet,
        ask="Suggest a paired test.",
        previous_requests=request,
        previous_response=response,
    )
    assert new[0]["messages"][-2:] == [
        {"role": "assistant", "content": "That is a hypothesis we can test."},
        {"role": "user", "content": "Suggest a paired test."},
    ]
    response["results"][0]["messages_sha256"] = "wrong"
    with pytest.raises(ValueError):
        followup_request(
            packet, ask="more", previous_requests=request, previous_response=response
        )


def test_offline_commands_do_not_need_a_server(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        pytest.fail("Offline review contacted an endpoint")

    monkeypatch.setattr(LocalJeffEndpoint, "request", no_network)
    row = example()
    source = tmp_path / "results.jsonl"
    source.write_text(json.dumps(row) + "\n")
    assert (
        main(["thresholds", str(source), "--output", str(tmp_path / "thresholds.json")])
        == 0
    )
    assert (
        main(
            [
                "export",
                str(source),
                "--arm",
                row["arm"],
                "--output-dir",
                str(tmp_path / "cp"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "followup",
                str(tmp_path / "cp" / row["id"] / "after.json"),
                "--ask",
                "Why?",
                "--output",
                str(tmp_path / "requests.json"),
            ]
        )
        == 0
    )
