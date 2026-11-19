"""Summarize recorded and counterfactual decisions separately from observed SQLite writes."""

import argparse
import difflib
import json
from pathlib import Path
import sqlite3

from scripts import approval_action_trial as trial


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare_patch(arm):
    evidence = trial.read(arm / "receipt.json")
    parts = []
    for filename in evidence["overrides"]:
        source = arm.parent / "source" / filename
        candidate = arm / "source" / filename
        assert trial.gate.digest(candidate) == evidence["overrides"][filename]
        parts.extend(difflib.unified_diff(
            source.read_text().splitlines(True), candidate.read_text().splitlines(True),
            fromfile="a/" + filename, tofile="b/" + filename,
        ))
    (arm / "review.patch").write_text("".join(parts))


def full_data(path):
    cases = {}
    for file in sorted((path / "olmo/pytest").glob("test_*/conversation.json")):
        if file.parent.is_symlink():
            continue
        conv = trial.read(file)
        with sqlite3.connect(f"file:{file.parent}/behavioral/ispec-behavioral.db?mode=ro", uri=True) as db:
            count, = db.execute("select count(*) from project_comment").fetchone()
        turns = conv["turns"]
        assert count == (len(turns[-1]["comments"]) if turns else 0), conv["case"]
        assert (file.parent / "transcript.md").is_file()
        conv["trace"] = file.parent / "transcript.md"
        cases[conv["case"]] = conv
    assert len(cases) == 32
    return cases


def classifier_transcript(output, inputs, replay):
    """Render the two model messages and original verdict for every actual baseline check."""
    text = ["# Baseline approval checks", "",
            "These are recorded synthetic conversations, not live customer examples. "
            "The user message is pretty-printed JSON; no field was removed. Token counts "
            "are from replaying the same request. The structured output schema and raw messages "
            "are in [inputs.json](inputs.json).", ""]
    for row in inputs:
        if not row["called"]:
            continue
        system, user = row["request"]["messages"]
        result = row["original"]
        tokens = replay[row["key"], "a"]["usage"]["prompt_tokens"]
        text.extend([f'## {row["key"]} ({tokens} input tokens)', "",
                     f'**{result["label"]} {result["confidence"]:.2f}.** {result["reason"]}',
                     "", "System:", "", "```text", system["content"].rstrip(), "```",
                     "", "User:", "", "```json",
                     json.dumps(json.loads(user["content"]), indent=2, ensure_ascii=False),
                     "```", ""])
    (output / "classifier-transcripts.md").write_text("\n".join(text))


def report(output):
    inputs = trial.read(output / "inputs.json")
    replay = {(p["key"], p["arm"]): p for p in lines(output / "replay.jsonl")}
    jeff = {(p["key"], p["context"]): p for p in lines(output / "jeff.jsonl")}
    assert len(replay) == 3 * len(inputs) and len(jeff) == 2 * len(inputs)
    classifier_transcript(output, inputs, replay)
    policies = ("a", "b", "c", "d", "Jeff-c", "Jeff-d")
    agree = sum(r["original"]["label"] == replay[r["key"], "a"]["decision"]["label"]
                for r in inputs if r["called"])
    text = ["# Approval replay and isolated full comparisons", "",
            f"Baseline label reproduced on {agree}/10 actual save checks. Labels/confidences below are generated "
            "by OLMo, except Jeff columns, which give P(yes). Threshold 0.75 throughout. "
            "b is a simulation of the observed model call, not a live gate setting. "
            "[Read every original classifier input and verdict](classifier-transcripts.md).", ""]

    def observation(r, policy):
        if policy == "b":
            return r["called"], "call" if r["called"] else "no call"
        if policy.startswith("Jeff"):
            p = jeff[r["key"], policy[-1]]
            return p["allow"], f'{p["scores"]["yes"]:.2f}'
        p = replay[r["key"], policy]
        d = p["decision"]
        return p["allow"], f'{d["label"]} {d["confidence"]:.2f}' + ("" if p["valid"] else " (invalid/late)")

    for group, subset in (("Recorded save calls", [r for r in inputs if r["called"]]),
                          ("Uncalled, pending follow-ups (hypothetical checks only)", [r for r in inputs if not r["called"]])):
        text.extend(["## " + group, "", "|Turn / user reply|Expected|" + "|".join(policies) + "|",
                     "|---|---|" + "---|" * len(policies)])
        matrices = {p: dict(TP=0, FP=0, TN=0, FN=0) for p in policies}
        for r in subset:
            vals = []
            for p in policies:
                allow, detail = observation(r, p)
                matrices[p][("T" if allow == r["wanted"] else "F") + ("P" if allow else "N")] += 1
                vals.append(("allow" if allow else "block") + ": " + detail)
            text.append(f'|{r["key"]} {r["reply"]}|{"save" if r["wanted"] else "no save"}|' + "|".join(vals) + "|")
        text.extend(["", "|Policy|TP|FN|TN|FP|", "|---|---:|---:|---:|---:|"])
        for policy, matrix in matrices.items():
            text.append(f"|{policy}|" + "|".join(str(matrix[k]) for k in ("TP","FN","TN","FP")) + "|")
        text.append("")
    absent = trial.read(output / "no-pending.json")
    text.append("No prior pending draft (no check constructed): " + ", ".join(r["key"] for r in absent) + ".")
    text.extend(["", "## Full controller runs", ""])
    arms = {"baseline": trial.BASE}
    for tag in ("check", "workflow", "combined"):
        path = output / tag
        if path.is_dir():
            arms[tag] = path
            compare_patch(path)
    append_full_table(text, arms)
    (output / "comparison.md").write_text("\n".join(text) + "\n")


def append_full_table(text, arms):
    """Link the same scripted cases across full runs, checking writes against final SQLite state."""
    data = {tag: full_data(path) for tag, path in arms.items()}
    for case, reference in data["baseline"].items():
        for tag, conversations in data.items():
            assert conversations[case]["case_definition"]["messages"] == reference["case_definition"]["messages"], (tag, case)
    scores = {tag: {r["case"]:r for r in trial.collect(path)["cases"]} for tag,path in arms.items()}
    text.extend(["|Arm|Requested saves|False writes|Mechanical passes|Completed conversations|",
                 "|---|---:|---:|---:|---:|"])
    for tag, results in scores.items():
        rows = list(results.values())
        text.append(f"|{tag}|{sum(r['approved_writes'] for r in rows)}/8|{sum(r['false_writes'] for r in rows)}|"
                    f"{sum(not r['findings'] for r in rows)}/32|{sum(r['complete'] for r in rows)}/32|")
    text.extend(["", "|Case|Expected|" + "|".join(arms) + "|", "|---|---|" + "---|" * len(arms)])
    for case, expected in scores["baseline"].items():
        row = [case, expected["expected"]]
        for tag in arms:
            r = scores[tag][case]
            item = " / ".join(r["actions"]) or "no completed turn"
            if not r["complete"]:
                item += " (incomplete)"
            if r["false_writes"]:
                item += " **FALSE SAVE**"
            row.append(f'[{item}](<{data[tag][case]["trace"]}>)')
        text.append("|" + "|".join(row) + "|")


def report_qualifiers(output):
    qualified = trial.read(output / "gate.json")["qualified"]
    arms = {"baseline": trial.BASE}
    for tag in qualified:
        arms[tag] = output / tag
        compare_patch(arms[tag])
    text = ["# Qualifying policies through the full controller", "",
            "All arms use the original 32 user scripts. The assistant generates its replies "
            "again; cell links open the full transcripts. Counts are verified against SQLite.", ""]
    append_full_table(text, arms)
    (output / "full-comparison.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--qualifiers", action="store_true", help="Report round two's gated full runs")
    args = p.parse_args()
    (report_qualifiers if args.qualifiers else report)(args.output.resolve())
