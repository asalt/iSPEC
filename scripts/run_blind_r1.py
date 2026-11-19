"""Run the prelabelled synthetic blind set from the r1 commit, then shadow its exact checks with y1."""

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from scripts import approval_action_trial as trial
from scripts import replay_approval_wording as wording
from scripts.score_olmo_application import score


R1 = "5e816b5"
CASES = Path("tests/behavioral/data/blind_r1_20261118.json")
TEST = Path("tests/behavioral/test_blind_r1_live.py")


def controller_run(*, out, src, port):
    old = trial.read(trial.BASE / "receipt.json")
    env = {k: v.replace(str(trial.BASE), str(out)) for k, v in old["test_env"].items()}
    env["ISPEC_VLLM_URL"] = f"http://127.0.0.1:{port}"
    assert env["ISPEC_EVAL_SOURCE"] == str(src)
    tail = old["test_command"][old["test_command"].index("-p"):]
    command = [trial.gate.GPU_PYTHON, "-m", "pytest", str(TEST), *(x.replace(str(trial.BASE), str(out)) for x in tail)]
    (out / "olmo").mkdir()
    with (out / "olmo/pytest.log").open("x") as log:
        child = subprocess.Popen(command, cwd=src, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = child.wait(timeout=1500)
        finally:
            trial.stop(child)
    if rc not in (0, 1):
        raise RuntimeError(f"pytest exited {rc}; see {log.name}")
    return rc


def collect(out, cases):
    """Cross-check final project_comment counts and extract only actual classifier HTTP requests."""
    results, checks = {}, []
    for path in (out / "olmo/pytest").glob("test_*/conversation.json"):
        if path.parent.is_symlink():
            continue
        conversation = trial.read(path)
        case = conversation["case"]
        assert case not in results and cases[case]["messages"] == conversation["case_definition"]["messages"]
        assert conversation["evidence_role"] == "r1_first_blind_run"
        trace = path.with_name("transcript.md")
        assert trace.is_file()
        db_uri = f"file:{path.parent}/behavioral/ispec-behavioral.db?mode=ro"
        with sqlite3.connect(db_uri, uri=True) as db:
            stored = db.execute("select count(*) from project_comment").fetchone()[0]
        row = score(conversation)
        assert len(row["stored_comments"]) == stored
        results[case] = {**row, "trace": str(trace)}
        for call in trial.read(path.with_name("wire.json"))["calls"]:
            req = call["request"]
            if not call["path"].endswith("completions") or "label" not in req.get(
                "structured_outputs", {}).get("json", {}).get("properties", {}):
                continue
            state = json.loads(req["messages"][1]["content"])
            index = cases[case]["messages"].index(state["latest_user_reply"])
            counts = cases[case]["comment_counts"]
            checks.append({"key": f"{case}:{index+1}:{len(checks)+1}", "case": case, "request": req,
                           "wanted": counts[index] > (counts[index-1] if index else 0),
                           "original": json.loads(call["response"]["choices"][0]["message"]["content"])})
    assert cases.keys() == results.keys()
    return results, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = trial.read(trial.ROOT / CASES)
    cases = {case["id"]: case for case in fixture["cases"]}
    design = {"origin": fixture["origin"], "gate": fixture["gate"], "r1": R1,
              "overlay_sha256": {str(f): trial.gate.digest(trial.ROOT / f) for f in (CASES, TEST)}}
    assert set(cases) == {f"B{i:02d}" for i in range(1, 13)}
    trial.write(out / "design.json", design)
    with trial.isolated_olmo(out, backend_ref=R1) as (port, src, receipt):
        for f in (CASES, TEST):
            assert not (src / f).exists()
            shutil.copyfile(trial.ROOT / f, src / f)
            assert trial.gate.digest(src / f) == design["overlay_sha256"][str(f)]
        receipt["pytest_exit"] = controller_run(out=out, src=src, port=port)
        results, checks = collect(out, cases)
        trial.write(out / "cases.json", results)
        trial.write(out / "actual-check-inputs.json", checks)
        print(f"Full conversations: {len(results)}; actual checks for y1: {len(checks)}", flush=True)
        trial.replay(port, checks, out, arms=("y1",), build=wording.request)


if __name__ == "__main__":
    main()
