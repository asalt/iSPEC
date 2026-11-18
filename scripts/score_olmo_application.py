"""Score synthetic application conversations from SQLite observations; never call a model."""

import argparse
import json
from pathlib import Path


def score(case: dict) -> dict:
    """Keep actual actions and delivered text separate from pass/fail assertions."""
    expected = case["expected_comment_counts"]
    turns = case["turns"]
    before = wanted_before = approved = false_saves = 0
    actions, replies, card_texts, writes = [], [], [], []
    for i, turn in enumerate(turns):
        count = len(turn["comments"])
        added = max(0, count - before)
        requested = max(0, expected[i] - wanted_before)
        approved += min(added, requested)
        false_saves += max(0, added - requested)
        writes.extend(c["text"] for c in turn["comments"][before:])
        before, wanted_before = count, expected[i]
        executed = []
        for call in case["tool_calls"]:
            if call["turn"] != i:
                continue
            result = call.get("result") or {}
            name = call["name"]
            if not result.get("ok"):
                executed.append(f"blocked:{name}")
            elif name == "prepare_project_comment":
                executed.append("re-prepared" if any(d["status"] == "superseded" for d in turn["drafts"]) else "prepared")
            elif name == "create_project_comment":
                executed.append("saved" if added else "existing/no write")
            elif name == "cancel_project_comment_draft":
                executed.append("cancelled")
            else:
                executed.append(name)
        actions.append(", ".join(executed) or "no tool")
        replies.append(turn["assistant"])
        card_texts.append([c["comment"] for c in case["cards_by_turn"][i] if c["status"] == "pending"])
    return {
        "case": case["case"], "expected": case["category"], "requested": max(expected),
        "approved_writes": approved, "false_writes": false_saves, "actions": actions,
        "stored_comments": writes, "pending_card_texts": card_texts, "delivered": replies,
        "complete": len(turns) == len(expected) and not case.get("exception"),
        "findings": case["mechanical_findings"],
    }


def collect(root: Path) -> dict:
    receipt = json.loads((root / "receipt.json").read_text())
    rows = {}
    for path in sorted((root / "olmo/pytest").glob("*/conversation.json")):
        if path.parent.is_symlink():
            continue
        row = score(json.loads(path.read_text()))
        if row["case"] in rows:
            raise ValueError(f"duplicate case {row['case']}")
        rows[row["case"]] = row
    order = receipt.get("case_ids", sorted(rows))  # Early full runs recorded only the count.
    if set(order) != rows.keys() or len(rows) != receipt["expected_cases"]:
        raise ValueError(f"Incomplete case evidence: expected {receipt['expected_cases']}, found {sorted(rows)}")
    return {"head": receipt["head"], "cases": [rows[k] for k in order]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    result = collect(args.run)
    (args.run / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.compare:
        old = {c["case"]: c for c in json.loads(args.compare.read_text())["cases"]}
        if set(old) != {c["case"] for c in result["cases"]}:
            raise ValueError("Comparison requires the same set of cases")
        changes = {}
        for case in result["cases"]:
            prior = old[case["case"]]
            changed = {k: {"before": prior[k], "after": v} for k, v in case.items() if prior[k] != v}
            if changed:
                changes[case["case"]] = changed
        (args.run / "score-diff.json").write_text(json.dumps(changes, indent=2) + "\n")
        print(f"{len(changes)}/{len(old)} cases differ; see score-diff.json (includes wording changes)")
    rows = result["cases"]
    print(f"{sum(r['approved_writes'] for r in rows)}/{sum(r['requested'] for r in rows)} "
          f"requested writes; {sum(r['false_writes'] for r in rows)} unapproved; {len(rows)} cases")


if __name__ == "__main__":
    main()
