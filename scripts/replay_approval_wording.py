"""Second-round synthetic replay. Full native runs require five true saves and zero false allows."""

import argparse
import json
from pathlib import Path
import shutil

from scripts import approval_action_trial as trial


PREVIOUS = trial.ROOT / "tests/behavioral/local/approval-action-4ebd366"
ARMS = ("y1", "y2", "r1", "r2")
BINARY = ("y1", "y2")
MIXED = (
    '"The timeline checks out. Log this while I have the notebook open.": approve_save.\n'
    '"The timeline checks out. Keep this on my desk until the handover.": draft_only.\n'
)


def prompt(arm):
    if arm in BINARY:
        return trial.QUESTION + "\nReturn JSON with label (yes or no) and confidence (0 to 1).\n"
    original = trial.PROMPT.read_text()
    assert original.count(" in a wording review") == 1
    ablated = original.replace(" in a wording review", "")
    if arm == "r1":
        return ablated
    assert arm == "r2"
    return ablated.replace("Choose approve_save,", MIXED + "Choose approve_save,")


def request(row, arm):
    req = trial.request_for(row, "d" if arm == "y2" else "a")
    req["messages"][0]["content"] = prompt(arm)
    if arm in BINARY:
        schema = req["structured_outputs"]["json"]
        schema["properties"]["label"]["enum"] = ["yes", "no"]
        del schema["properties"]["reason"]
        schema["required"].remove("reason")
    return req


def configure_snapshot(source, arm):
    """Apply the replayed policy to the isolated staged classifier, retaining the original write gates."""
    path = source / trial.CHECK_FILE
    front, _ = path.read_text().rsplit("+++\n", 1)
    path.write_text(front + "+++\n" + prompt(arm))
    if arm == "y2":
        trial.replace_once(source, trial.HANDLER, '"presented_draft": presented,',
                           '"presented_draft": ' + trial.FIXED + ',')
    if arm not in BINARY:
        return
    module = "src/ispec/assistant/small_classifier.py"
    trial.replace_once(source, module, '    task = get_small_classifier_task(task_name)\n    return {',
                       '''    task = get_small_classifier_task(task_name)
    if task_name == "staged_comment_approval":
        return ''' + repr(request({"request": trial.read(PREVIOUS / "inputs.json")[0]["request"]},
                                  "y1")["structured_outputs"]["json"]) + '\n    return {')
    trial.replace_once(source, module, '    if label not in task.labels:',
                       '''    if task_name == "staged_comment_approval":
        if set(decision) != {"label", "confidence"} or label not in {"yes", "no"}:
            return None
        label = {"yes": "approve_save", "no": "deny_save"}[label]
        decision = {**decision, "reason": "Binary save-intent check."}
    if label not in task.labels:''')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def evaluate(out):
    rows = trial.read(out / "inputs.json")
    actual = [r for r in rows if r["called"]]
    shadow = [r for r in rows if not r["called"]]
    assert (len(actual), len(shadow), sum(r["wanted"] for r in actual)) == (10, 14, 7)
    scores = {(p["arm"], p["key"]): p for path in (PREVIOUS, out)
              for p in records(path / "replay.jsonl")}
    columns = ("a", "d", "c", *ARMS)
    text = ["# OLMo save-intent replay, round two", "",
            "Each cell is the predicted label and its self-reported confidence. "
            "Only a timely affirmative at confidence >= 0.75 allows a save. "
            "a, c and d reuse the previous replay of the identical inputs.", ""]
    matrices = {}
    for title, group in (("Actual calls", actual), ("Hypothetical only", shadow)):
        text += ["## " + title, "", "|Turn / latest reply|Expected|" + "|".join(columns) + "|",
                 "|---|---|" + "---|" * len(columns)]
        counts = {arm: dict(TP=0, FN=0, TN=0, FP=0) for arm in columns}
        for row in group:
            cells = [row["key"] + " " + row["reply"], "save" if row["wanted"] else "no save"]
            for arm in columns:
                result = scores[arm, row["key"]]
                allowed = result["allow"]
                counts[arm][("T" if row["wanted"] == allowed else "F") + ("P" if allowed else "N")] += 1
                decision = result["decision"]
                cells.append(f'{decision["label"]} {decision["confidence"]:.2f}' +
                             ("" if result["valid"] else " (invalid/late)"))
            text.append("|" + "|".join(cells) + "|")
        text += ["", "|Policy|TP|FN|TN|FP|", "|---|---:|---:|---:|---:|"]
        text += [f"|{a}|" + "|".join(str(v) for v in matrix.values()) + "|" for a, matrix in counts.items()]
        text.append("")
        matrices[title] = counts
    (out / "comparison.md").write_text("\n".join(text) + "\n")
    measured, hypothetical = matrices.values()
    qualified = [a for a in ARMS if measured[a]["TP"] >= measured["a"]["TP"]
                 and not measured[a]["FP"] and not hypothetical[a]["FP"]]
    trial.write(out / "gate.json", dict(matrices=matrices, qualified=qualified))
    print("GATE", matrices, qualified, flush=True)
    return qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PREVIOUS / "inputs.json", out / "inputs.json")
    rows = trial.read(out / "inputs.json")
    trial.write(out / "design.json", {
        "inputs_sha256": trial.gate.digest(out / "inputs.json"),
        "reference_scores_sha256": trial.gate.digest(PREVIOUS / "replay.jsonl"),
        "prompts": {arm: prompt(arm) for arm in ARMS},
        "gate": ">=5/7 true allows, 0/3 actual false allows, 0/14 hypothetical false allows",
    })
    with trial.isolated_olmo(out) as (port, source, receipt):
        trial.replay(port, rows, out, arms=ARMS, build=request)
        for arm in evaluate(out):
            trial.full_arm(output=out, snapshot=source, port=port, tag=arm, check=arm,
                           wording=False, configure=configure_snapshot)
            receipt.setdefault("full_runs", []).append(arm)


if __name__ == "__main__":
    main()
