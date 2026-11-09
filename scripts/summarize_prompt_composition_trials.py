"""Summarize saved synthetic composition records; never reruns inference."""

import argparse
import json
from pathlib import Path
import statistics


def read_records(path):
    return [
        json.loads(line.removeprefix("COMPOSITION_EVAL "))
        for line in path.read_text().splitlines()
        if line.startswith("COMPOSITION_EVAL ")
    ]


def summarize(paths):
    reports = []
    for path in paths:
        records = read_records(path)
        turns = []
        for record in records:
            wire = record["wire"]
            usages = [(call.get("response") or {}).get("usage") or {} for call in wire]
            layouts = record["layouts"]
            main_calls = [call for call in wire if call.get("base_occurrences", 0) > 0]
            main_layouts = [
                layout for layout in layouts if layout["base_occurrences"] > 0
            ]
            first_main_usage = (
                (main_calls[0].get("response", {}).get("usage") or {})
                if main_calls
                else {}
            )
            turns.append(
                {
                    "case": record["case"],
                    "turn": record["turn"],
                    "response": record["response"],
                    "errors": record["errors"],
                    "elapsed_ms": record["elapsed_ms"],
                    "comment_count": record["comment_count"],
                    "tools": [
                        {"name": call["name"], "ok": call["result"].get("ok")}
                        for call in record["executed"]
                    ],
                    "model_calls": len(wire),
                    "input_tokens": sum(
                        usage.get("prompt_tokens", 0) for usage in usages
                    ),
                    "output_tokens": sum(
                        usage.get("completion_tokens", 0) for usage in usages
                    ),
                    "first_input_tokens": (
                        usages[0].get("prompt_tokens") if usages else None
                    ),
                    "first_system_messages": (
                        layouts[0]["roles"].count("system") if layouts else None
                    ),
                    "first_main_input_tokens": first_main_usage.get("prompt_tokens"),
                    "first_main_system_messages": (
                        main_layouts[0]["roles"].count("system")
                        if main_layouts
                        else None
                    ),
                    "base_occurrences": [
                        layout["base_occurrences"] for layout in layouts
                    ],
                    "history_messages": record["metadata"]
                    .get("context", {})
                    .get("history_messages_used"),
                }
            )
        reports.append(
            {
                "log": str(path),
                "recorded_turns": len(turns),
                "passing_turns": sum(not turn["errors"] for turn in turns),
                "median_ms": (
                    statistics.median(turn["elapsed_ms"] for turn in turns)
                    if turns
                    else None
                ),
                "turns": turns,
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.logs)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(
        json.dumps(
            [
                {key: value for key, value in report.items() if key != "turns"}
                for report in result
            ]
        )
    )


if __name__ == "__main__":
    main()
