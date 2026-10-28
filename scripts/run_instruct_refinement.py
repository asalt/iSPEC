"""Run isolated Instruct trials against an already-started loopback model.

No env files, credentials, production DBs, or service lifecycle operations.
The selected Python must already have the backend/test dependencies installed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/behavioral/data/model_followup_20261025/instruct_refinement_cases.json"


def trial_environment(output, *, url, model, trial, split):
    target = urlsplit(url)
    if (target.scheme != "http" or target.hostname != "127.0.0.1" or not target.port
            or target.username or target.password or target.path or target.query or target.fragment):
        raise ValueError("Use a credential-free loopback URL with explicit port and no path")
    # Start a clean child environment instead of inheriting a running app's
    # configuration or credentials. Tests replace these DB paths with fixtures.
    env = {"PATH": os.defpath, "HOME": str(output / "home"), "LANG": "C.UTF-8",
           "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
           "ISPEC_RUN_LLM_TESTS": "1", "ISPEC_INSTRUCT_TRIAL": trial,
           "ISPEC_INSTRUCT_SPLIT": split, "ISPEC_VLLM_URL": url,
           "ISPEC_VLLM_MODEL": model, "ISPEC_VLLM_TOOL_CALL_PARSER": "olmo3",
           "ISPEC_VLLM_REASONING_PARSER": "", "ISPEC_VLLM_STRICT_PROTOCOL": "1",
           "ISPEC_ASSISTANT_PROVIDER": "vllm", "ISPEC_INFERENCE_USAGE_LOG_ENABLED": "0",
           "ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED": "0",
           "ISPEC_LOG_DIR": str(output / "logs"), "ISPEC_STATE_DIR": str(output / "state")}
    for name in ("DB", "ASSISTANT_DB", "SCHEDULE_DB", "AGENT_DB", "AGENT_STATE_DB", "PROMPTS_DB", "OMICS_DB"):
        env["ISPEC_" + name + "_PATH"] = str(output / (name.lower() + ".db"))
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", nargs="+", choices=["baseline", "protocol", "schema", "candidate"], required=True)
    parser.add_argument("--split", choices=["dev", "held"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18006")
    parser.add_argument("--model", default="olmo-eval")
    parser.add_argument("--select", help="Optional pytest -k expression")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    trial_environment(output, url=args.url, model=args.model, trial=args.trial[0], split=args.split)
    output.mkdir(parents=True, exist_ok=False)
    files = [CORPUS, Path(__file__), ROOT / "tests/behavioral/test_instruct_refinement_live.py",
             ROOT / "src/ispec/assistant/tools.py", ROOT / "src/ispec/assistant/service.py",
             ROOT / "src/ispec/api/routes/support.py"]
    files.extend((ROOT / "src/ispec/prompts").glob("experimental.*refinement.md"))
    files.extend((ROOT / "src/ispec/prompts").glob("experimental.slack.note_proposal.*.md"))
    files.append(ROOT / "src/ispec/prompts/assistant.tools.olmo3_preamble.md")
    receipt = {"started_at": datetime.now(timezone.utc).isoformat(), "root": str(ROOT),
               "python": sys.executable, "model": args.model, "url": args.url,
               "split": args.split, "trials": args.trial,
               "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               "runs": [], "production": False}
    for trial in args.trial:
        folder = output / trial
        folder.mkdir()
        (folder / "home").mkdir()
        env = trial_environment(folder, url=args.url, model=args.model, trial=trial, split=args.split)
        command = [sys.executable, "-m", "pytest", "tests/behavioral/test_instruct_refinement_live.py",
                   "--run-behavioral", "-q", "-s", "--basetemp=" + str(folder / "pytest"),
                   "--junitxml=" + str(folder / "results.xml")]
        if args.select:
            command.extend(["-k", args.select])
        with (folder / "pytest.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["runs"].append({"trial": trial, "exit_code": result.returncode, "command": command})
        (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"trial": trial, "exit_code": result.returncode, "log": str(folder / "pytest.log")}), flush=True)
    return int(any(run["exit_code"] for run in receipt["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
