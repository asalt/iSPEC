"""Compare strict Think final answers, on-time decisions, and saved Instruct replay scores."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

from scripts import replay_approval_wording as wording


def prediction(record, *, deadline=False):
    valid = record["valid"] and (not deadline or record["seconds"] <= 6)
    return valid and record["label"] == "yes" and record["confidence"] >= 0.75


def score(expected, allowed):
    return ("T" if expected == allowed else "F") + ("P" if allowed else "N")


def report(out):
    parent = out.parent
    references = {}
    for folder, policies in [("approval-action-4ebd366", {"a"}), ("approval-wording-round2-4ebd366", {"y1", "r1"})]:
        references.update({(r["arm"], r["key"]): r for r in wording.records(parent / folder / "replay.jsonl")
                           if r["arm"] in policies})
    rows = wording.trial.read(parent / "approval-wording-round2-4ebd366/inputs.json")
    receipt = wording.trial.read(out / "receipt.json")
    results = {(r["off"], r["key"]): r for r in wording.records(out / "results.jsonl")}
    assert receipt["services"][-1]["port_closed"]
    summary = {}
    for off in ([False, True] if receipt["full_off_arm"] else [False]):
        subset = [results[off, row["key"]] for row in rows]
        counts = Counter(r["finish"] for r in subset)
        spans = [r["reasoning_tokens"] for r in subset if r["reasoning_tokens"] is not None]
        summary[str(off)] = {
            "n": len(subset), "finish": counts, "valid": sum(r["valid"] for r in subset),
            "separated": sum(r["separated"] for r in subset), "within_deadline": sum(r["seconds"] <= 6 for r in subset),
            "reasoning_token_range": [min(spans), median(spans), max(spans)] if spans else None,
            "seconds_range": [min(r["seconds"] for r in subset), median(r["seconds"] for r in subset),
                              max(r["seconds"] for r in subset)],
        }
    matrix = {}
    text = ["# Think vs Instruct: identical y1 classifier inputs", "",
            "a = baseline Instruct; y1 = binary Instruct; r1 = six-label Instruct. "
            "Think uses the binary question and original conversation. Label and self-reported confidence "
            "are shown as returned; invalid finals are not repaired. Tokens are whole generated tokens "
            "before the closing reasoning tag (n/a if no delimiter), not the server's usage estimate.", ""]
    for called, title in ((True, "Recorded save calls"), (False, "Pending follow-ups without save calls")):
        counters = {}
        text.extend(["## " + title, "", "|Turn / user reply|Expected|a|y1|r1|Think final|Reasoning tokens|Seconds|",
                     "|---|---|---|---|---|---|---:|---:|"])
        for row in rows:
            if row["called"] != called:
                continue
            key = row["key"]
            cells = [key + " " + row["reply"], "save" if row["wanted"] else "no save"]
            for arm in ("a", "y1", "r1"):
                ref = references[arm, key]
                counters.setdefault(arm, Counter())[score(row["wanted"], ref["allow"])] += 1
                dec = ref["decision"]
                cells.append(f'{dec["label"]} {dec["confidence"]:.2f}')
            for off in ([False, True] if receipt["full_off_arm"] else [False]):
                r = results[off, key]
                arm = "Think off" if off else "Think"
                for deadline in (False, True):
                    name = arm + (" at 6s" if deadline else " eventual")
                    counters.setdefault(name, Counter())[score(row["wanted"], prediction(r, deadline=deadline))] += 1
            r = results[False, key]
            cells += [f'{r["label"]} {r["confidence"]:.2f}' if r["valid"] else f'invalid ({r["finish"]})',
                      str(r["reasoning_tokens"]) if r["reasoning_tokens"] is not None else "n/a",
                      f'{r["seconds"]:.2f}']
            text.append("|" + "|".join(cells) + "|")
        matrix[title] = counters
        text.extend(["", "|Policy|TP|FN|TN|FP|", "|---|---:|---:|---:|---:|"])
        for name, counter in counters.items():
            text.append("|" + name + "|" + "|".join(str(counter[k]) for k in ("TP", "FN", "TN", "FP")) + "|")
        text.append("")
    text.append("The 6s row is a counterfactual: slow or invalid checks block. It is not semantic correctness.")
    (out / "comparison.md").write_text("\n".join(text) + "\n")
    wording.trial.write(out / "summary.json", dict(timing=summary, matrices=matrix, off_probe=receipt["off_probe"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
