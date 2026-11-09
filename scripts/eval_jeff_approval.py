"""Read-only CPU-classifier trial; fixed synthetic corpus, no tool execution."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import shutil

from ispec.assistant.jeff_approval import (
    ApprovalPromptVariant,
    ApprovalTrialSettings,
    CONTEXT_SUFFIXES,
    CHOICE_SUFFIXES,
    QUESTION_SUFFIXES,
    JeffUnavailable,
    LocalJeffEndpoint,
    PendingNoteExchange,
    evaluate_pending_approval,
    prepare_approval_request,
)
from ispec.assistant.reply_interpretation import classify_short_reply


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT
    / "tests/behavioral/data/model_followup_20261025/project_comment_approval_cases.json",
    ROOT / "tests/behavioral/data/jeff_approval_cases.json",
)
SHORT_CASES = ROOT / "tests/behavioral/data/jeff_short_prompt_cases.json"
CHOICE_CASES = ROOT / "tests/behavioral/data/jeff_choice_cases.json"


def load_cases() -> list[dict]:
    previous, fresh = [json.loads(path.read_text()) for path in SOURCES]
    cases = []
    labels = {
        "approve_save": "yes",
        "deny_save": "no",
        "draft_only": "no",
        "revise_draft": "no",
        "requires_explicit_confirmation": "no",
        "unrelated_or_unclear": "unsure",
        None: "unsure",
    }
    for row in previous["cases"]:
        cases.append(
            {
                "id": "prior_" + row["id"],
                "split": "prior",
                "expected_label": labels[row["expected_label"]],
                "exchange": PendingNoteExchange(
                    project_id=42,
                    note="Eleven saliva samples arrived; peptide coverage review is pending.",
                    initial_request="Make a note for project 42: eleven saliva samples arrived; peptide coverage review is pending.",
                    assistant_draft=previous["prior_assistant_message"],
                    latest_reply=row["reply"],
                    pending=row["pending"],
                ),
            }
        )
    for row in fresh["cases"]:
        cases.append(
            {
                "id": row["id"],
                "split": row["split"],
                "expected_label": row["expected_label"],
                "exchange": PendingNoteExchange(
                    project_id=42,
                    note=fresh["note"],
                    initial_request=fresh["initial_request"],
                    assistant_draft=f'Draft: "{fresh["note"]}" Shall I save this exact note?',
                    latest_reply=row["reply"],
                    pending=row.get("pending", True),
                ),
            }
        )
    # Matched length controls change context length, not the current approval request.
    for repeats in (0, 10, 40):
        detail = (
            "Labels will be checked against the sample sheet before any experiments begin. "
            * repeats
        )
        for label, reply in (
            ("yes", "Yes, save the exact draft."),
            ("no", "Yes, but make it shorter before saving."),
        ):
            note = fresh["note"] + " " + detail
            cases.append(
                {
                    "id": f"length_{repeats}_{label}",
                    "split": "length",
                    "expected_label": label,
                    "exchange": PendingNoteExchange(
                        project_id=42,
                        note=note,
                        initial_request=fresh["initial_request"] + " " + detail,
                        assistant_draft=f'Draft: "{note}" Shall I save this exact note?',
                        latest_reply=reply,
                    ),
                }
            )
    return cases


def unavailable(_request):
    raise JeffUnavailable("simulated_offline")


def load_short_cases():
    """Keep seen diagnostics separate from fresh wording; no padded length controls."""
    selected = {
        "example_confirm",
        "example_shorten",
        "prior_deny_after_yes",
        "held_cancel",
        "held_approve",
        "held_social",
    }
    cases = [
        {**case, "split": "seen"} for case in load_cases() if case["id"] in selected
    ]
    source = json.loads(SHORT_CASES.read_text())
    for row in source["cases"]:
        cases.append(
            {
                "id": row["id"],
                "split": "fresh",
                "expected_label": row["expected_label"],
                "exchange": PendingNoteExchange(
                    **source["scenarios"][row["scenario"]],
                    latest_reply=row["reply"],
                ),
            }
        )
    return cases


def short_prompt_variants():
    return [
        ApprovalPromptVariant(question=question, context=context)
        for question in ("baseline", "affirmation", "authorize", "save_now")
        for context in ("object", "messages", "live_last", "transcript")
    ]


def load_choice_cases():
    cases = [{**case, "split": "seen"} for case in load_short_cases()]
    source = json.loads(CHOICE_CASES.read_text())
    cases.extend(
        {
            "id": row["id"],
            "split": "fresh",
            "evaluation_role": row.get("evaluation_role", "approval_reply"),
            "expected_label": row["expected_label"],
            "exchange": PendingNoteExchange(
                **{**source["scenario"], **row.get("context", {})},
                latest_reply=row["reply"],
            ),
        }
        for row in source["cases"]
    )
    return cases


def choice_variants():
    return [
        ApprovalPromptVariant(question=question, context=context, choices=choices)
        for question in ("authorize", "approval")
        for context in ("live_last", "reply")
        for choices in ("original", "binary")
    ]


def evaluate_case(case, endpoint, settings, *, variant=ApprovalPromptVariant()):
    wire = []

    def query(request):
        response = endpoint(request)
        wire.append({"request": request, "response": response})
        return response

    result = evaluate_pending_approval(
        case["exchange"], query=query, settings=settings, variant=variant
    )
    fallback = evaluate_pending_approval(
        case["exchange"], query=unavailable, settings=settings, variant=variant
    )
    expected = case["expected_label"]
    # Binary scoring merges non-approval/unclear only for label accuracy, not for authorization.
    expected_raw = (
        "no" if variant.choices == "binary" and expected == "unsure" else expected
    )
    return {
        "id": case["id"],
        "split": case["split"],
        "evaluation_role": case.get("evaluation_role", "approval_reply"),
        "arm": f"{variant.question}/{variant.context}/{variant.choices}",
        "prompt_families": variant.families,
        "expected_label": expected,
        "exchange": asdict(case["exchange"]),
        "result": result.to_dict(),
        "raw_label_correct": result.raw_label == expected_raw,
        "expected_raw_label": expected_raw,
        "decision_correct": result.decision == expected,
        "false_approval": result.would_save and expected != "yes",
        "missed_approval": not result.would_save and expected == "yes",
        "simulated_unavailable": fallback.to_dict(),
        "legacy_short_reply_kind": classify_short_reply(
            case["exchange"].latest_reply
        ).kind,
        "wire": wire,
    }


def summarize(rows):
    timings = sorted(
        row["result"]["latency_ms"] for row in rows if row["result"]["source"] == "jeff"
    )
    token_counts = [
        wire["response"].get("usage", {}).get("input_tokens")
        for row in rows
        for wire in row["wire"]
    ]
    token_counts = [count for count in token_counts if isinstance(count, int)]
    return {
        "cases": len(rows),
        "raw_label_correct": sum(row["raw_label_correct"] for row in rows),
        "decision_correct": sum(row["decision_correct"] for row in rows),
        "false_approvals": [row["id"] for row in rows if row["false_approval"]],
        "missed_approvals": [row["id"] for row in rows if row["missed_approval"]],
        "fallback_used": [
            row["id"] for row in rows if row["result"]["source"] == "lexical"
        ],
        "errors": [row["id"] for row in rows if row["result"]["error"]],
        "median_ms": statistics.median(timings) if timings else None,
        "max_ms": max(timings) if timings else None,
        "median_input_tokens": statistics.median(token_counts)
        if token_counts
        else None,
        "simulated_fallback_false_approvals": [
            row["id"]
            for row in rows
            if row["simulated_unavailable"]["would_save"]
            and row["expected_label"] != "yes"
        ],
        "shadow_only": True,
        "write_count": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18765")
    parser.add_argument("--model", default="jeff-latest")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=15)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--layout-probe", action="store_true")
    mode.add_argument("--prompt-sweep", action="store_true")
    mode.add_argument("--choice-sweep", action="store_true")
    mode.add_argument(
        "--preview",
        metavar="CASE_ID",
        help="Print a request without calling any endpoint",
    )
    parser.add_argument("--question", choices=QUESTION_SUFFIXES, default="baseline")
    parser.add_argument("--context", choices=CONTEXT_SUFFIXES, default="object")
    parser.add_argument("--choices", choices=CHOICE_SUFFIXES, default="original")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only these case IDs, in the supplied order",
    )
    args = parser.parse_args(argv)
    variant = ApprovalPromptVariant(
        question=args.question, context=args.context, choices=args.choices
    )
    if args.preview:
        case = next(
            (
                case
                for case in load_cases() + load_short_cases() + load_choice_cases()
                if case["id"] == args.preview
            ),
            None,
        )
        if case is None:
            parser.error("Unknown preview case")
        print(
            json.dumps(
                prepare_approval_request(case["exchange"], variant=variant), indent=2
            )
        )
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for evaluation")
    endpoint = LocalJeffEndpoint(url=args.url, timeout_seconds=args.timeout)
    health = endpoint.request(method="GET", path="/health")
    if health.get("status") != "ready":
        raise SystemExit("Jeff is not ready; no evaluation run started")
    if args.model != "jeff-latest":
        raise SystemExit("This frozen pilot uses the base model alias jeff-latest")
    settings = ApprovalTrialSettings()
    cases = load_cases()
    arms = [variant]
    if args.prompt_sweep:
        cases = load_short_cases()
        arms = short_prompt_variants()
    if args.choice_sweep:
        cases = load_choice_cases()
        arms = choice_variants()
    if args.layout_probe:
        cases = [
            case
            for case in cases
            if case["id"]
            in {
                "example_confirm",
                "example_shorten",
                "prior_deny_after_yes",
                "held_cancel",
            }
        ]
        arms = [
            ApprovalPromptVariant(context=layout, choices=wording)
            for layout in ("object", "transcript")
            for wording in ("original", "descriptive")
        ]
    if args.case:
        by_id = {case["id"]: case for case in cases}
        if len(set(args.case)) != len(args.case) or set(args.case) - set(by_id):
            parser.error("Case IDs must exist in the selected suite and be unique")
        cases = [by_id[case_id] for case_id in args.case]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Preserve the actual inputs and source, not merely pointers to a dirty checkout.
    sources = (
        (*SOURCES, SHORT_CASES, CHOICE_CASES)
        if args.choice_sweep
        else ((*SOURCES, SHORT_CASES) if args.prompt_sweep else SOURCES)
    )
    implementation = (
        Path(__file__),
        ROOT / "scripts/jeff_approval_review.py",
        ROOT / "src/ispec/assistant/jeff_approval.py",
        *sorted(
            (ROOT / "src/ispec/prompts").glob(
                "experimental.project_comment.jeff_approval*.md"
            )
        ),
    )
    for path in (*sources, *implementation):
        target = args.output_dir / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (args.output_dir / "run.json").write_text(
        json.dumps(
            {
                "health": health,
                "url": args.url,
                "settings": asdict(settings),
                "layout_probe": args.layout_probe,
                "prompt_sweep": args.prompt_sweep,
                "choice_sweep": args.choice_sweep,
                "arms": [asdict(arm) for arm in arms],
                "planned_cases": len(cases) * len(arms),
                "sources": {
                    str(path.relative_to(ROOT)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sources
                },
                "implementation": {
                    str(path.relative_to(ROOT)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in implementation
                },
            },
            indent=2,
        )
        + "\n"
    )
    rows = []
    # Rotate arm order across cases to distribute startup/warm-cache effects.
    jobs = [
        (case, arm)
        for index, case in enumerate(cases)
        for arm in arms[index % len(arms) :] + arms[: index % len(arms)]
    ]
    for case, arm in jobs:
        row = evaluate_case(case, endpoint, settings, variant=arm)
        rows.append(row)
        with (args.output_dir / "results.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "arm": row["arm"],
                    "expected": row["expected_label"],
                    "result": row["result"],
                }
            ),
            flush=True,
        )
        if row["result"]["error"]:
            # Preserve the failed attempt; an unavailable endpoint is not a model benchmark.
            break
    summary = summarize(rows)
    summary["planned_cases"] = len(jobs)
    summary["complete"] = len(rows) == len(jobs) and not summary["errors"]
    summary["by_split"] = {
        split: summarize([row for row in rows if row["split"] == split])
        for split in sorted({row["split"] for row in rows})
    }
    summary["by_evaluation_role"] = {
        role: summarize([row for row in rows if row["evaluation_role"] == role])
        for role in sorted({row["evaluation_role"] for row in rows})
    }
    summary["by_arm"] = {
        arm: {
            **summarize([row for row in rows if row["arm"] == arm]),
            "by_split": {
                split: summarize(
                    [row for row in rows if row["arm"] == arm and row["split"] == split]
                )
                for split in sorted({row["split"] for row in rows})
            },
        }
        for arm in sorted({row["arm"] for row in rows})
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.choice_sweep:
        from scripts.jeff_approval_review import threshold_sweep

        (args.output_dir / "thresholds.json").write_text(
            json.dumps(threshold_sweep(rows), indent=2) + "\n"
        )
    print(json.dumps(summary, indent=2))
    return int(
        bool(
            summary["errors"]
            or summary["false_approvals"]
            or summary["missed_approvals"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
