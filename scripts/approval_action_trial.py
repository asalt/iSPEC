"""Offline, frozen approval inputs and isolated full application comparisons."""

import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from scripts import run_olmo_application_gate as gate
from scripts.run_model_regression import endpoint, stop
from scripts.score_olmo_application import collect

ROOT = gate.ROOT
BASE = ROOT / "tests/behavioral/local/application-baseline-37fe2be"
PROMPT = ROOT / "tests/behavioral/experiments/action-save-check.txt"
CHECK_FILE = "src/ispec/prompts/assistant.small_classifier.staged_comment_approval.md"
WORKFLOW_FILE = "src/ispec/prompts/assistant.project_comment.workflow.md"
HANDLER = "src/ispec/assistant/project_comment_workflow.py"
QUESTION = "Is the user asking to save this draft to the project's history now?"
WORKFLOW = ("For a project-history draft, call prepare_project_comment(project_id, comment). "
            "The server shows the exact stored draft in a confirmation card. On the web, a person can click "
            "Save; any authorized caller, including a headless API client, can also ask you to save it. "
            "When they do, call create_project_comment(draft_id, confirm=true), which checks the request "
            "before saving. When presenting a draft, ask whether to save it to the project's history. "
            "For wording-only help, respond in prose.")
FIXED = """f'Save this note to project {draft.project_id}\\'s history? "{draft.comment}"'"""


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixed_question(draft):
    return f'Save this note to project {draft["project_id"]}\'s history? "{draft["comment"]}"'


def inputs():
    """Use actual captured requests, plus separately marked off-path pending follow-ups."""
    rows, unscorable = [], []
    template = None
    for case_dir in sorted((BASE / "olmo/pytest").glob("test_*")):
        if case_dir.is_symlink():
            continue
        case = read(case_dir / "conversation.json")
        wire = read(case_dir / "wire.json")["calls"]
        checks = {}
        for call in wire:
            req = call["request"]
            if not call["path"].endswith("completions") or "label" not in req.get(
                "structured_outputs", {}).get("json", {}).get("properties", {}):
                continue
            state = json.loads(req["messages"][1]["content"])
            decision = json.loads(call["response"]["choices"][0]["message"]["content"])
            checks[state["latest_user_reply"]] = (req, decision)
            template = req
        counts = case["expected_comment_counts"]
        for i, reply in enumerate(case["case_definition"]["messages"][1:], 1):
            row = {"key": f'{case["case"].split("_")[0]}:{i+1}', "reply": reply,
                   "wanted": counts[i] > counts[i-1], "called": reply in checks}
            if row["called"]:
                row["request"], row["original"] = checks.pop(reply)
            else:
                prior = case["turns"][i-1]
                pending = [d for d in prior["drafts"] if d["status"] == "pending"]
                if not pending:
                    unscorable.append(row)
                    continue
                prep = next(c for c in case["tool_calls"] if c["name"] == "prepare_project_comment"
                            and c.get("result", {}).get("result", {}).get("draft_id") == pending[-1]["id"])
                row["state"] = {"initial_request": case["turns"][prep["turn"]]["user"],
                                "presented_draft": prior["assistant"],
                                "pending_comment": prep["result"]["result"], "latest_user_reply": reply}
            rows.append(row)
        assert not checks
    for row in rows:
        if "state" in row:
            row["request"] = copy.deepcopy(template)
            row["request"]["messages"][1]["content"] = json.dumps(row.pop("state"), ensure_ascii=False)
    return sorted(rows, key=lambda r: r["key"]), unscorable


def request_for(row, arm):
    req = copy.deepcopy(row["request"])
    if arm in {"c", "d"}:
        req["messages"][0]["content"] = PROMPT.read_text()
    if arm == "d":
        state = json.loads(req["messages"][1]["content"])
        state["presented_draft"] = fixed_question(state["pending_comment"])
        req["messages"][1]["content"] = json.dumps(state, ensure_ascii=False)
    return req


def replay(port, rows, output, *, arms=("a", "c", "d"), build=request_for):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (output / "replay.jsonl").open("x", buffering=1) as log:
        for row in rows:
            for arm in arms:
                req = build(row, arm)
                post = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                             json.dumps(req).encode(), {"Content-Type": "application/json"})
                started = time.monotonic()
                with opener.open(post, timeout=60) as reply:
                    result = json.load(reply)
                elapsed = time.monotonic() - started
                choice = result["choices"][0]
                answer = json.loads(choice["message"]["content"])
                valid = choice["finish_reason"] == "stop" and elapsed <= 6
                labels = req["structured_outputs"]["json"]["properties"]["label"]["enum"]
                affirmative = "yes" if labels == ["yes", "no"] else "approve_save"
                allow = valid and answer["label"] == affirmative and answer["confidence"] >= 0.75
                print(json.dumps(dict(key=row["key"], arm=arm, decision=answer, allow=allow,
                                      request=req, seconds=elapsed, usage=result["usage"], valid=valid)), file=log)


def selection(output, rows):
    predictions = {(r["key"], r["arm"]): r["allow"] for r in
                   map(json.loads, (output / "replay.jsonl").read_text().splitlines())}
    counts = {}
    for arm in "abcd":
        matrix = dict(TP=0, FP=0, TN=0, FN=0)
        for r in rows:
            if r["called"]:
                allow = arm == "b" or predictions[r["key"], arm]
                matrix[("T" if allow == r["wanted"] else "F") + ("P" if allow else "N")] += 1
        counts[arm] = matrix
    # b is an offline ceiling, never a bypass installed in the application.
    winner = min("cda", key=lambda a: (counts[a]["FP"], -counts[a]["TP"]))
    write(output / "selection.json", {"counts": counts, "winner": winner})
    print("REPLAY", counts, "selected", winner, flush=True)
    return winner


def jeff(rows, output):
    import torch
    from jeff.models import load_decision_model
    torch.set_num_interop_threads(1)
    model = load_decision_model(checkpoint=Path("/mnt/e/tools/jeff/jeff-base-v1.3"),
                                device="cpu", cpu_threads=1)
    assert model.device_name == "cpu"
    criteria = {"yes": "The user asks to record this version now.",
                "no": "The user does not ask to record this version now.",
                "unsure": "The requested action is unclear."}
    with (output / "jeff.jsonl").open("x", buffering=1) as log:
        for r in rows:
            for state_arm in ("c", "d"):
                state = request_for(r, state_arm)["messages"][1]["content"]
                row = {"state": state, "question": {"type": "choice", "criteria": criteria,
                       "instructions": PROMPT.read_text().split("Choose approve_save")[0].replace(
                           ": approve_save.", ": yes.").replace(": draft_only.", ": no.").strip()}}
                probabilities, = model.predict(rows=[row], batch_size=1)
                scores = dict(zip(criteria, map(float, probabilities), strict=True))
                print(json.dumps(dict(key=r["key"], context=state_arm, scores=scores,
                                      allow=scores["yes"] >= 0.75, request=row)), file=log)


def replace_once(source, filename, old, new):
    path = source / filename
    text = path.read_text()
    assert text.count(old) == 1, (filename, old)
    path.write_text(text.replace(old, new))


def full_arm(*, output, snapshot, port, tag, check, wording, configure=None):
    dest = output / tag
    dest.mkdir()
    source = dest / "source"
    shutil.copytree(snapshot, source, ignore=shutil.ignore_patterns("__pycache__"))
    if configure is not None:
        configure(source, check)
    if check in {"c", "d"}:
        path = source / CHECK_FILE
        front, _ = path.read_text().rsplit("+++\n", 1)
        path.write_text(front + "+++\n" + PROMPT.read_text())
    if check == "d":
        replace_once(source, HANDLER, '"presented_draft": presented,', '"presented_draft": ' + FIXED + ',')
    if wording:
        line = (source / WORKFLOW_FILE).read_text().splitlines()[4]
        assert line.startswith("For a project-history draft")
        replace_once(source, WORKFLOW_FILE, line, WORKFLOW)
        replace_once(source, HANDLER, "after the user's text approval.", "when the user asks you to save it.")
        replace_once(source, "src/ispec/assistant/tools.py", "classifier-checked text approval",
                     "classifier checks whether the user asked to save")
    overrides = {str(p.relative_to(source)): gate.digest(p) for p in source.rglob("*") if p.is_file()
                 and gate.digest(p) != gate.digest(snapshot / p.relative_to(source))}
    old = read(BASE / "receipt.json")
    env = {k: v.replace(str(BASE), str(dest)) for k, v in old["test_env"].items()}
    assert env["ISPEC_EVAL_SOURCE"] == str(source)
    env["ISPEC_VLLM_URL"] = f"http://127.0.0.1:{port}"
    command = [s.replace(str(BASE), str(dest)) for s in old["test_command"]]
    evidence = {"head": "4ebd366", "overrides": overrides, "tag": tag, "check": check,
                "wording": wording, "case_ids": old["case_ids"], "expected_cases": old["expected_cases"],
                "test_command": command, "port": port}
    (dest / "olmo").mkdir()
    write(dest / "receipt.json", evidence)
    print("FULL START", tag, overrides, flush=True)
    with (dest / "olmo/pytest.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=1800)
        finally:
            stop(proc)
    assert rc in (0, 1), rc
    scores = collect(dest)
    write(dest / "scores.json", scores)
    evidence.update(test_exit_code=rc)
    write(dest / "receipt.json", evidence)
    rows = scores["cases"]
    metric = sum(r["approved_writes"] for r in rows), sum(r["false_writes"] for r in rows)
    print("FULL END", tag, metric, flush=True)
    return metric


@contextmanager
def isolated_olmo(out, *, backend_ref="4ebd366"):
    """Check GPU availability and serve the frozen baseline on an owned loopback port."""
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip()
    src, manifest = gate.snapshot(output=out, committed=True, backend_ref=backend_ref, wrapper_ref="ca57e20")
    profile = gate.read_profile(out / "model-profile.env")
    template = gate.resolve_chat_template(source=src, python=gate.GPU_PYTHON, profile=profile)
    old = read(BASE / "receipt.json")["test_env"]
    env = {k: old[k].replace(str(BASE), str(out)) for k in old if not k.startswith(("ISPEC_", "VLLM_"))}
    env["VLLM_CACHE_ROOT"] = str(out / "vllm-cache")
    def command(fd, port):
        return gate.server_command(source=src, python=gate.GPU_PYTHON, checkpoint=gate.CHECKPOINT,
                                   profile=profile, template=template, fd=fd, port=port)
    receipt = {"services": [], "baseline": manifest["head"], "production": False}
    try:
        with endpoint(out, "server", command, env, 18030, receipt) as (port, _):
            receipt["ready"] = gate.verify_inference(port=port, model=profile["ISPEC_VLLM_MODEL"])
            print("READY", flush=True)
            yield port, src, receipt
    finally:
        write(out / "runtime.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "gpu", "jeff"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.action == "prepare":
        out.mkdir(parents=True, exist_ok=False)
        rows, absent = inputs()
        write(out / "inputs.json", rows)
        write(out / "no-pending.json", absent)
        print(len(rows), "pending follow-ups;", sum(r["called"] for r in rows), "recorded;", len(absent), "no pending draft")
        return
    rows = read(out / "inputs.json")
    if args.action == "jeff":
        jeff(rows, out)
        return
    with isolated_olmo(out) as (port, src, receipt):
        replay(port, rows, out)
        winner = selection(out, rows)
        metrics = {}
        for tag, check, wording in (("check", winner, False), ("workflow", "a", True)):
            metrics[tag] = full_arm(output=out, snapshot=src, port=port, tag=tag, check=check, wording=wording)
        if winner != "a" and all(saves >= 5 and false <= 1 and (saves, false) != (5, 1)
                                 for saves, false in metrics.values()):
            metrics["combined"] = full_arm(output=out, snapshot=src, port=port, tag="combined",
                                            check=winner, wording=True)
        receipt["metrics"] = metrics


if __name__ == "__main__":
    main()
