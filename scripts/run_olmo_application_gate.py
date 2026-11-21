"""One local OLMo batch from a hashed working-tree snapshot and native template."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_model_regression import endpoint, stop  # noqa: E402

CHECKPOINT = Path("/mnt/e/ispec-model-cache/olmo3-instruct-6e5971d9")
GPU_PYTHON = "/home/alex/venvs/py3.12/bin/python"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(*, output: Path, committed: bool = False, backend_ref: str = "HEAD",
             wrapper_ref: str = "HEAD") -> tuple[Path, dict]:
    """Copy Git-listed working files, including untracked application additions."""
    source = output / "source"
    if committed:
        head = subprocess.check_output(["git", "rev-parse", "--verify", f"{backend_ref}^{{commit}}"], cwd=ROOT, text=True).strip()
        wrapper = ROOT.parent
        wrapper_head = subprocess.check_output(["git", "rev-parse", "--verify", f"{wrapper_ref}^{{commit}}"], cwd=wrapper, text=True).strip()
        with subprocess.Popen(["git", "archive", head], cwd=ROOT, stdout=subprocess.PIPE) as proc:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
                archive.extractall(source, filter="data")
            if proc.wait():
                raise RuntimeError("Could not export committed backend")
        profile = output / "model-profile.env"
        profile.write_bytes(subprocess.check_output(
            ["git", "show", f"{wrapper_head}:configs/models/olmo7b-instruct-vllm.env"], cwd=wrapper,
        ))
        manifest = {
            "head": head, "wrapper_head": wrapper_head, "snapshot_kind": "git_archive",
            "source": str(source), "profile_sha256": digest(profile),
            "files": {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return source, manifest
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
    ).decode().split("\0")
    hashes = {}
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        if not path.is_file():
            continue
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[name] = digest(destination)
    profile = ROOT.parent / "configs/models/olmo7b-instruct-vllm.env"
    shutil.copy2(profile, output / "model-profile.env")
    manifest = {
        "source": str(source), "files": hashes,
        "profile_sha256": digest(profile),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "snapshot_kind": "working_tree_including_untracked_not_ignored",
        "excluded": ["ignored local artifacts/databases/env files", ".git"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "working-tree.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    return source, manifest


def read_profile(path: Path) -> dict[str, str]:
    """Parse the public checked-in model profile without shell evaluation."""
    return dict(line.split("=", 1) for line in path.read_text().splitlines()
                if line.strip() and not line.startswith("#"))


def resolve_chat_template(*, source: Path, python: str, profile: dict[str, str]) -> Path:
    """Resolve the reviewed template from the frozen source, not the model cache."""
    resolved = subprocess.check_output(
        [python, str(source / "src/ispec/assistant/chat_templates.py"),
         profile["VLLM_CHAT_TEMPLATE_NAME"]],
        text=True,
    ).strip()
    return Path(resolved)


def server_command(
    *, source: Path, python: str, checkpoint: Path, profile: dict[str, str],
    template: Path, fd: int, port: int,
) -> list[str]:
    """Build the exact isolated server command retained in its launch receipt."""
    command = [
        python, str(source / "scripts/run_model_regression.py"), "--vllm-child", str(fd),
        "--model", str(checkpoint), "--served-model-name", profile["ISPEC_VLLM_MODEL"],
        "--host", "127.0.0.1", "--port", str(port), "--dtype", profile["VLLM_DTYPE"],
        "--max-model-len", profile["VLLM_MAX_MODEL_LEN"], "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.8", "--enforce-eager", "--attention-backend", "TRITON_ATTN",
        "--generation-config", "vllm", "--enable-auto-tool-choice", "--tool-call-parser", "olmo3",
        "--chat-template", str(template), "--no-enable-log-requests",
    ]
    if profile.get("VLLM_STRUCTURED_OUTPUTS_CONFIG"):
        command.extend(["--structured-outputs-config", profile["VLLM_STRUCTURED_OUTPUTS_CONFIG"]])
    return command


def verify_inference(*, port: int, model: str) -> dict:
    """Require a real generated reply before starting the behavioral batch."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly ISPEC_MODEL_OK."}],
        "temperature": 0,
        "max_tokens": 24,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    if not content or "ISPEC_MODEL_OK" not in content:
        raise RuntimeError(f"Readiness inference returned unexpected content: {content!r}")
    return {"request": payload, "response": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", help="Exact case IDs for a focused retest")
    parser.add_argument("--head", action="store_true", help="Export committed backend and wrapper profile")
    parser.add_argument("--compare", type=Path, help="Compare application outcomes against a saved scores.json")
    parser.add_argument("--directory-test", action="store_true", help="Isolated eight-case read-only directory batch")
    parser.add_argument("--project-search", action="store_true", help="Also run project retrieval controls on the same model")
    parser.add_argument("--local-search-case", type=Path, help="Private local search fixture; never copied to source")
    parser.add_argument("--local-search-backup", type=Path, help="Completed offline core DB backup for the real case (opened immutable)")
    options = parser.parse_args()
    if options.local_search_case and not (options.project_search and options.local_search_case.is_file()):
        parser.error("--local-search-case needs --project-search and a readable private fixture")
    if options.local_search_backup and not (options.local_search_case and options.local_search_backup.is_file()):
        parser.error("--local-search-backup requires a private case and an existing snapshot")
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    occupancy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
    ).strip()
    if occupancy:
        raise RuntimeError("Existing GPU compute workload; refusing startup")
    if options.compare and (options.directory_test or not options.compare.is_file()):
        parser.error("--compare needs an existing application baseline scores file")
    source, manifest = snapshot(output=output, committed=options.head)
    test_node = "tests/behavioral/test_olmo_application_live.py::test_application_conversation"
    if options.directory_test:
        from tests.behavioral.directory_fixture import adapt_snapshot
        route = source / "src/ispec/api/routes/support.py"
        adapt_snapshot(route)
        manifest["files"]["src/ispec/api/routes/support.py"] = digest(route)
        manifest["test_adapter"] = "directory_fixture.adapt_snapshot"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        test_node = "tests/behavioral/test_directory_live.py::test_directory"
    profile = read_profile(output / "model-profile.env")
    template = resolve_chat_template(source=source, python=GPU_PYTHON, profile=profile)
    shutil.copy2(template, output / "chat_template.jinja")
    gate_cases = json.loads((source / "tests/behavioral/data/staged_promotion_20261107.json").read_text())["cases"]
    turn_counts = {case["id"]: len(case["messages"]) for case in gate_cases}
    turn_counts.update({name: 1 for name in (
        "C01_project", "C02_person", "C03_runs", "C04_greeting", "C05_missing_id",
    )})
    scenarios = source / "tests/behavioral/data/scenario_cases_20261120.json"
    if scenarios.exists():
        turn_counts.update({case["id"]: len(case["messages"]) for case in json.loads(scenarios.read_text())["cases"]})
    selected = options.cases or list(turn_counts)
    if options.directory_test:
        turn_counts = {f"D{i}": 1 for i in range(1, 9)}
        selected = options.cases or list(turn_counts)
    unknown = set(selected) - turn_counts.keys()
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    receipt = {
        "state": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
        "runner_pid": os.getpid(), "source_manifest_sha256": digest(output / "manifest.json"),
        "profile": profile, "expected_cases": len(selected),
        "expected_turns": sum(turn_counts[name] for name in selected), "case_ids": selected,
        "checkpoint": str(CHECKPOINT), "head": manifest["head"], "production": False,
        "services": [], "custom_template": False, "prompt_plugin": None,
        "template_fix": "Packaged native template plus the tools default guard; no wording changes",
        "served_chat_template": {
            "path": str(template), "sha256": digest(template), "source": "frozen_application_package",
            "explicit_override": True,
        },
        "versions": {name: importlib.metadata.version(name)
                     for name in ("vllm", "torch", "transformers", "pytest")},
        "checkpoint_artifacts": {name: digest(CHECKPOINT / name)
                                 for name in ("config.json", "tokenizer_config.json", "chat_template.jinja")},
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv",], text=True),
    }

    def record() -> None:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    common = {
        "PATH": "/home/alex/venvs/py3.12/bin:/usr/bin:/bin", "HOME": str(output / "home"),
        "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_CACHE_ROOT": str(output / "vllm-cache"),
        "TRITON_CACHE_DIR": str(output / "triton-cache"), "CUDA_CACHE_PATH": str(output / "cuda-cache"),
    }

    def command(fd, port):
        argv = server_command(
            source=source, python=GPU_PYTHON, checkpoint=CHECKPOINT, profile=profile,
            template=template, fd=fd, port=port,
        )
        receipt["server_command"] = argv
        record()
        return argv

    record()
    try:
        with endpoint(base=output, label="olmo", command=command, env=common,
                      preferred=18030, receipt=receipt) as (port, folder):
            receipt["readiness_inference"] = verify_inference(port=port, model=profile["ISPEC_VLLM_MODEL"])
            receipt["state"] = "inference_verified"
            record()
            print("ISPEC_GATE_EVENT " + json.dumps({
                "state": "inference_verified", "port": port,
                "reply": receipt["readiness_inference"]["response"]["choices"][0]["message"]["content"],
            }), flush=True)
            env = {
                **common, **profile, "PYTHONPATH": os.pathsep.join((str(source / "src"), str(source))),
                "ISPEC_EVAL_SOURCE": str(source), "ISPEC_RUN_LLM_TESTS": "1",
                "ISPEC_MODEL_REGRESSION_CAPTURE": "1", "ISPEC_VLLM_URL": f"http://127.0.0.1:{port}",
                "ISPEC_STATE_DIR": str(folder / "state"), "ISPEC_LOG_DIR": str(folder / "logs"),
                "ISPEC_INFERENCE_USAGE_LOG_ENABLED": "0", "ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED": "0",
                "ISPEC_ASSISTANT_READ_ONLY": "0", "ISPEC_ASSISTANT_ALLOWED_WRITE_TOOLS": "create_project_comment",
                "ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_EVAL_MODE": "own",
            }
            for name in ("DB", "ASSISTANT_DB", "AGENT_DB", "AGENT_STATE_DB", "SCHEDULE_DB", "PROMPTS_DB"):
                env[f"ISPEC_{name}_PATH"] = str(folder / (name.lower() + ".db"))
            receipt["test_env"] = env
            extra_tests = []
            if options.project_search:
                extra_tests.append("tests/behavioral/test_project_search_live.py")
                if options.local_search_case:
                    env["ISPEC_LOCAL_PROJECT_SEARCH_CASE"] = str(options.local_search_case.resolve())
                if options.local_search_backup:
                    env["ISPEC_LOCAL_SEARCH_BACKUP"] = str(options.local_search_backup.resolve())
            receipt["test_command"] = [
                GPU_PYTHON, "-m", "pytest",
                *extra_tests,
                *[f"{test_node}[{name}]"
                  for name in selected],
                "-p", "scripts.model_regression_capture", "--run-behavioral", "-q", "-s", "--tb=short",
                "--basetemp=" + str(folder / "pytest"), "--junitxml=" + str(folder / "results.xml"),
            ]
            with (folder / "pytest.log").open("x") as log:
                process = subprocess.Popen(
                    receipt["test_command"], cwd=source, env=env, start_new_session=True,
                    stdout=log, stderr=subprocess.STDOUT,
                )
                receipt.update(state="testing", test_pid=process.pid, test_started_at=time.time())
                record()
                print("ISPEC_GATE_EVENT " + json.dumps({"state": "testing", "cases": len(selected)}), flush=True)
                try:
                    receipt["test_exit_code"] = process.wait(timeout=1800)
                finally:
                    stop(process)
            receipt["state"] = "finished" if receipt["test_exit_code"] == 0 else "test_failures"
    except Exception as exc:
        receipt.update(state="error", error=f"{type(exc).__name__}: {exc}")
        print("ISPEC_GATE_EVENT " + json.dumps({"state": "error", "error": receipt["error"]}), flush=True)
        raise
    finally:
        receipt["ended_at"] = datetime.now(timezone.utc).isoformat()
        receipt["source_changed"] = [name for name, value in manifest["files"].items()
                                     if digest(source / name) != value]
        receipt["checkpoint_artifacts_changed"] = [name for name, value in receipt["checkpoint_artifacts"].items()
                                                    if digest(CHECKPOINT / name) != value]
        record()
        (output / "finished.json").write_text(json.dumps({
            key: receipt.get(key) for key in ("state", "test_exit_code", "error", "ended_at", "source_changed")
        }, indent=2) + "\n")
        print("ISPEC_GATE_EVENT " + json.dumps({
            "state": receipt["state"], "test_exit_code": receipt.get("test_exit_code"),
            "output": str(output),
        }), flush=True)
    if not options.directory_test:
        command = [sys.executable, str(source / "scripts/score_olmo_application.py"), str(output)]
        if options.compare:
            command.extend(["--compare", str(options.compare.resolve())])
        subprocess.run(command, check=True)
    raise SystemExit(receipt["test_exit_code"])


if __name__ == "__main__":
    main()
