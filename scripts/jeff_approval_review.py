"""Offline policy comparisons and conversational checkpoints; never an executor."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ispec.assistant.jeff_approval import (
    BINARY_LABELS,
    LABELS,
    JeffContractError,
    LocalJeffEndpoint,
    PendingNoteExchange,
    parse_approval_response,
)
from ispec.prompt import render_prompt
from scripts.eval_model_matrix import digest as messages_digest


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def distribution(row):
    if row["result"].get("error") or not row["wire"]:
        return None
    wire = row["wire"][-1]
    labels = tuple(wire["request"]["questions"]["save_approval"]["criteria"])
    return parse_approval_response(wire["response"], labels=labels)[1]


def would_approve(probabilities, *, threshold, margin=0.0, strict=False):
    """Offline policy only. Reject ties, including at the argmax-only control."""
    yes = probabilities["yes"]
    other = max(value for key, value in probabilities.items() if key != "yes")
    passed = yes > threshold if strict else yes >= threshold
    return yes > other and passed and yes - other >= margin


def threshold_sweep(rows):
    policies = [(0.0, 0.0, False), (0.5, 0.0, True), (0.5, 0.2, True)]
    policies += [
        (threshold, margin, False)
        for threshold in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)
        for margin in (0.0, 0.2)
    ]
    results = []
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        for split in ("all", *sorted({row["split"] for row in arm_rows})):
            selected = [
                row for row in arm_rows if split == "all" or row["split"] == split
            ]
            for threshold, margin, strict in policies:
                false, missed, accepted, errors = [], [], [], []
                for row in selected:
                    try:
                        probabilities = distribution(row)
                        approved = probabilities is not None and would_approve(
                            probabilities,
                            threshold=threshold,
                            margin=margin,
                            strict=strict,
                        )
                    except JeffContractError:
                        probabilities, approved = None, False
                    if probabilities is None:
                        errors.append(row["id"])
                    expected = row["expected_label"] == "yes"
                    if approved and not expected:
                        false.append(row["id"])
                    if expected and not approved:
                        missed.append(row["id"])
                    if approved and expected:
                        accepted.append(row["id"])
                results.append(
                    {
                        "arm": arm,
                        "split": split,
                        "threshold": threshold,
                        "comparison": ">" if strict else ">=",
                        "margin": margin,
                        "cases": len(selected),
                        "positive_cases": sum(
                            row["expected_label"] == "yes" for row in selected
                        ),
                        "accepted_approvals": accepted,
                        "missed_approvals": missed,
                        "false_approvals": false,
                        "errors": errors,
                    }
                )
    return {"shadow_only": True, "write_count": 0, "results": results}


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def checkpoint(row, *, boundary, source, run):
    if boundary not in {"before", "after"} or not row["wire"]:
        raise ValueError(
            "A before/after checkpoint requires a captured inference request"
        )
    exchange = PendingNoteExchange(**row["exchange"])
    if exchange.gate_error():
        raise ValueError("Cannot replay an invalid pending exchange")
    payload = {
        "schema": "ispec.approval-checkpoint.v1",
        "boundary": boundary,
        "case_id": row["id"],
        "arm": row["arm"],
        "source": str(source),
        "source_record_sha256": digest(row),
        "model_provenance": run,
        "exchange": deepcopy(row["exchange"]),
        "messages": exchange.context()["messages"],
        "request": deepcopy(row["wire"][-1]["request"]),
        "response": deepcopy(row["wire"][-1]["response"])
        if boundary == "after"
        else None,
        "policy_result": deepcopy(row["result"]) if boundary == "after" else None,
        "shadow_only": True,
        "write_outcome": {"status": "not_attempted"},
    }
    return {**payload, "checkpoint_sha256": digest(payload)}


def load_checkpoint(path):
    payload = json.loads(Path(path).read_text())
    expected = payload.pop("checkpoint_sha256")
    if (
        payload.get("schema") != "ispec.approval-checkpoint.v1"
        or digest(payload) != expected
    ):
        raise ValueError(
            "Checkpoint changed; preserve the original and create a new branch"
        )
    if payload.get("shadow_only") is not True or payload.get("write_outcome") != {
        "status": "not_attempted"
    }:
        raise ValueError("This runner only accepts non-writing evaluation checkpoints")
    payload["checkpoint_sha256"] = expected
    return payload


def export_checkpoints(records, *, arm, case_ids, output):
    rows = [
        row
        for row in read_rows(records)
        if row["arm"] == arm and (not case_ids or row["id"] in case_ids)
    ]
    if not rows or (case_ids and set(case_ids) != {row["id"] for row in rows}):
        raise ValueError("Requested case/arm not found")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case/arm records")
    for row in rows:
        if not row["id"].isascii() or not all(
            char.isalnum() or char in "_-" for char in row["id"]
        ):
            raise ValueError("Unsafe case filename")
    run_path = Path(records).parent / "run.json"
    run = (
        json.loads(run_path.read_text()) if run_path.exists() else {"unavailable": True}
    )
    Path(output).mkdir(parents=True, exist_ok=False)
    index = [
        "# Approval checkpoints",
        "",
        "Saved conversation-level state, not model tensors. No note was saved.",
        "",
    ]
    for row in rows:
        folder = Path(output) / row["id"]
        folder.mkdir()
        for boundary in ("before", "after"):
            write_json(
                folder / f"{boundary}.json",
                checkpoint(
                    row, boundary=boundary, source=Path(records).resolve(), run=run
                ),
            )
        text = [f"# {row['id']}", "", f"Variant: `{row['arm']}`", ""]
        for message in PendingNoteExchange(**row["exchange"]).context()["messages"]:
            text += [f"## {message['role'].title()}", "", message["content"], ""]
        text += [
            "## Classifier request and result",
            "",
            "```json",
            json.dumps(row["wire"][-1], indent=2),
            "```",
            "",
            "No write attempted. Any later explanation is a hypothesis, not a recorded rationale.",
            "",
        ]
        (folder / "conversation.md").write_text("\n".join(text))
        index.append(
            f"- [{row['id']}]({row['id']}/conversation.md): [before]({row['id']}/before.json), [after]({row['id']}/after.json)"
        )
    (Path(output) / "README.md").write_text("\n".join(index) + "\n")


def replay_checkpoint(packet, endpoint):
    exchange = PendingNoteExchange(**packet["exchange"])
    if exchange.gate_error():
        raise ValueError("Invalid pending exchange")
    health = endpoint.request(method="GET", path="/health")
    if health.get("status") != "ready":
        raise ValueError("Selected Jeff endpoint is not ready")
    recorded_model = packet["model_provenance"].get("health", {}).get("model")
    if recorded_model and recorded_model != health.get("model"):
        raise ValueError(
            "Different Jeff model reported; use an explicitly labeled new experiment"
        )
    request = deepcopy(packet["request"])
    labels = tuple(request["questions"]["save_approval"]["criteria"])
    if labels not in (LABELS, BINARY_LABELS) or set(request) != {
        "state",
        "questions",
        "model",
    }:
        raise ValueError("Unsupported checkpoint request")
    response = endpoint(request)
    parse_approval_response(response, labels=labels)
    return {
        "parent_checkpoint_sha256": packet["checkpoint_sha256"],
        "recorded_model_provenance": packet["model_provenance"],
        "current_health": health,
        "request": request,
        "response": response,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "shadow_only": True,
        "write_outcome": {"status": "not_attempted"},
    }


def followup_request(packet, *, ask, previous_requests=None, previous_response=None):
    """Prepare a no-tools chat branch for the existing local model-matrix runner."""
    if bool(previous_requests) != bool(previous_response):
        raise ValueError("Continuation requires both its prior request and response")
    case_id = packet["case_id"] + "_" + packet["checkpoint_sha256"][:12]
    if previous_requests:
        if len(previous_requests) != 1 or previous_requests[0]["id"] != case_id:
            raise ValueError("Prior request belongs to a different checkpoint")
        messages = deepcopy(previous_requests[0]["messages"])
        replies = previous_response["results"]
        if (
            len(replies) != 1
            or replies[0]["id"] != case_id
            or not replies[0].get("ok")
            or replies[0].get("messages_sha256") != messages_digest(messages)
            or not isinstance(replies[0].get("content"), str)
            or not replies[0]["content"].strip()
        ):
            raise ValueError(
                "Reply missing, failed or not matched to this conversation branch"
            )
        messages.append({"role": "assistant", "content": replies[0]["content"]})
    else:
        data = {
            key: packet[key]
            for key in (
                "boundary",
                "messages",
                "request",
                "response",
                "policy_result",
                "write_outcome",
            )
        }
        # Keep the captured conversation as user-level evidence, not fresh commands or fabricated model turns.
        messages = [
            {
                "role": "system",
                "content": render_prompt(
                    "experimental.project_comment.jeff_approval.review"
                ).text.strip(),
            },
            {
                "role": "user",
                "content": "Saved experiment:\n"
                + json.dumps(data, ensure_ascii=False, indent=2),
            },
        ]
    messages.append({"role": "user", "content": ask})
    return [{"id": case_id, "messages": messages, "max_tokens": 1024}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    thresholds = subs.add_parser("thresholds")
    thresholds.add_argument("records", type=Path)
    thresholds.add_argument("--output", type=Path, required=True)
    export = subs.add_parser("export")
    export.add_argument("records", type=Path)
    export.add_argument("--arm", required=True)
    export.add_argument("--case", action="append", default=[])
    export.add_argument("--output-dir", type=Path, required=True)
    replay = subs.add_parser("replay")
    replay.add_argument("checkpoint", type=Path)
    replay.add_argument("--url", required=True)
    replay.add_argument("--output", type=Path, required=True)
    followup = subs.add_parser("followup")
    followup.add_argument("checkpoint", type=Path)
    followup.add_argument("--ask", required=True)
    followup.add_argument("--output", type=Path, required=True)
    followup.add_argument("--previous-requests", type=Path)
    followup.add_argument("--previous-response", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        export_checkpoints(
            args.records, arm=args.arm, case_ids=args.case, output=args.output_dir
        )
        return 0
    if args.output.exists():
        parser.error("Choose a new output file; preserve previous evidence")
    if args.command == "thresholds":
        result = threshold_sweep(read_rows(args.records))
    elif args.command == "replay":
        result = replay_checkpoint(
            load_checkpoint(args.checkpoint), LocalJeffEndpoint(url=args.url)
        )
    else:
        result = followup_request(
            load_checkpoint(args.checkpoint),
            ask=args.ask,
            previous_requests=json.loads(args.previous_requests.read_text())
            if args.previous_requests
            else None,
            previous_response=json.loads(args.previous_response.read_text())
            if args.previous_response
            else None,
        )
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
