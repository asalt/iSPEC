"""Historical November 5 Tulu/OLMo batch; never reads application env files.

Controller probes predate save offers; see docs/evaluation-checkpoints.md before
replay. This is not the current staged-workflow promotion gate.

Loads each cached GPU model once. All tools run in synthetic fixture databases;
network capture restricts tests to owned loopback inference endpoints. Earlier
held-out cases are now regression inputs, not new held-out evidence.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.local_test_port import reserve_loopback_socket  # noqa: E402
from scripts.run_instruct_refinement import trial_environment  # noqa: E402

MODELS = (
    ("tulu", "/home/alex/.cache/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B/snapshots/666943798adbde0b1aff34626007e26986a3c107", "llama3_json"),
    ("olmo-instruct", "/mnt/e/ispec-model-cache/olmo3-instruct-6e5971d9", "olmo3"),
)


def vllm_child(fd, args):
    sys.argv = [sys.argv[0], *args]
    from vllm.entrypoints.launchers import launcher
    from vllm.entrypoints.launchers.api_server.entry import main
    held = socket.socket(fileno=fd)

    def reserved(addr, *, reuse_port):
        assert not reuse_port and addr == held.getsockname()
        return held

    launcher.create_server_socket = reserved
    main()


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


@contextmanager
def endpoint(base, label, command, env, preferred, receipt):
    folder = base / label
    folder.mkdir()
    held = reserve_loopback_socket(preferred)
    port = held.getsockname()[1]
    record = {"label": label, "port": port, "started_at": time.time()}
    receipt["services"].append(record)
    process = None
    try:
        with (folder / "server.log").open("x") as log:
            process = subprocess.Popen(command(held.fileno(), port), cwd=ROOT, env=env,
                pass_fds=(held.fileno(),), start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        held.close()
        record["pid"] = process.pid
        print(json.dumps({"starting": record}), flush=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError(f"{label} exited; inspect server.log")
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200:
                        record["ready_at"] = time.time()
                        break
            except (OSError, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"{label} startup timed out")
        yield port, folder
    finally:
        held.close()
        if process is not None:
            stop(process)
            record["exit_code"] = process.returncode
        record["ended_at"] = time.time()
        with socket.socket() as sock:
            record["port_closed"] = sock.connect_ex(("127.0.0.1", port)) != 0
        (base / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"stopped": record}), flush=True)


def batches(suite="full"):
    if suite == "save-contract":
        yield "runtime-dev", "baseline", "dev", "test_model_runtime_followup_live.py", [
            "-k", "test_controller_with_real_fixture_tools and (approve or deny) and not approve_required_only",
        ], {}
        yield "refinement-dev", "baseline", "dev", "test_instruct_refinement_live.py", [
            "-k", "test_comment_controller and dev_approval",
        ], {}
        yield "refinement-held", "baseline", "held", "test_instruct_refinement_live.py", [
            "-k", "test_comment_controller and held_explicit_person",
        ], {}
        yield "wording-jeff", "baseline", "dev", "test_draft_approval_conversation_live.py", [], {}
        return
    if suite != "full":
        raise ValueError(f"Unknown regression suite: {suite}")
    yield "runtime-dev", "baseline", "dev", "test_model_runtime_followup_live.py", [], {}
    yield "thread-held", "baseline", "held", "test_model_runtime_followup_live.py", ["-k", "test_thread_context_controller"], {}
    for split in ("dev", "held"):
        selection = [] if split == "dev" else ["-k", "not test_project_lookup_controller"]
        yield f"refinement-{split}", "baseline", split, "test_instruct_refinement_live.py", selection, {}
    for corpus in ("project_comment_approval_cases.json", "model_followup_20261025/project_comment_approval_cases.json"):
        name = "approval-followup" if "/" in corpus else "approval-original"
        yield name, "baseline", "dev", "test_project_comment_approval_vllm_live.py", [], {
            "ISPEC_APPROVAL_EVAL_CASES_PATH": str(ROOT / "tests/behavioral/data" / corpus),
        }
    for trial in ("baseline", "concise", "compact"):
        yield f"composition-{trial}", trial, "dev", "test_prompt_composition_live.py", [], {}
    yield "wording-jeff", "baseline", "dev", "test_draft_approval_conversation_live.py", [], {}


def run_tests(base, label, port, jeff_port, model, parser, receipt, *, suite="full"):
    for name, trial, split, file, selection, extra in batches(suite):
        folder = base / label / name
        folder.mkdir()
        env = trial_environment(folder, url=f"http://127.0.0.1:{port}", model=model,
                                trial=trial, split=split, tool_parser=parser)
        env.update(extra)
        env.update(ISPEC_MODEL_REGRESSION_CAPTURE="1", ISPEC_VLLM_TIMEOUT_SECONDS="45")
        if name == "wording-jeff":
            env["ISPEC_EVAL_JEFF_URL"] = f"http://127.0.0.1:{jeff_port}"
        command = [sys.executable, "-m", "pytest", "tests/behavioral/" + file,
                   "-p", "scripts.model_regression_capture", "--run-behavioral", "-q", "-s", "--tb=short",
                   "--basetemp=" + str(folder / "pytest"), "--junitxml=" + str(folder / "results.xml"), *selection]
        record = {"model": label, "batch": name, "started_at": time.time(), "command": command}
        receipt["batches"].append(record)
        print(json.dumps({"testing": label, "batch": name}), flush=True)
        with (folder / "pytest.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                record["exit_code"] = process.wait(timeout=900)
            finally:
                stop(process)
        record["ended_at"] = time.time()
        (base / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"tested": label, "batch": name, "exit_code": record["exit_code"]}), flush=True)


def run_matrix(base, label, port, model, parser, receipt):
    for name, data in (("matrix-original", "data"), ("matrix-followup", "data/model_followup_20261025")):
        folder = base / label / name
        env = trial_environment(folder, url=f"http://127.0.0.1:{port}", model=model,
                                trial="baseline", split="dev", tool_parser=parser)
        # Separate raw-output probes from the actual controller/policy tests.
        commands = [
            ["prepare", str(folder), "--data-dir", str(ROOT / "tests/behavioral" / data)],
            ["run", str(folder / "requests.json"), "--output", str(folder / "results.json"),
             "--url", f"http://127.0.0.1:{port}", "--model", model, "--provider", "vllm"],
            ["score", str(folder / "manifest.json"), str(folder / "results.json"),
             "--output", str(folder / "scores.json")],
        ]
        record = {"model": label, "batch": name, "started_at": time.time()}
        receipt["batches"].append(record)
        with (base / label / (name + ".log")).open("x") as log:
            for args in commands:
                process = subprocess.Popen([sys.executable, str(ROOT / "scripts/eval_model_matrix.py"), *args],
                    cwd=ROOT, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=900)
                finally:
                    stop(process)
                if code:
                    raise RuntimeError(f"{name} command failed: {args[0]}")
        record.update(exit_code=0, ended_at=time.time())
        print(json.dumps({"tested": label, "batch": name, "scores": str(folder / "scores.json")}), flush=True)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("output", type=Path)
    args.add_argument("--gpu-python", default="/home/alex/venvs/py3.12/bin/python")
    args.add_argument("--suite", choices=["full", "save-contract"], default="full")
    options = args.parse_args()
    base = options.output.resolve()
    base.mkdir(parents=True, exist_ok=False)
    common = {"PATH": "/home/alex/venvs/py3.12/bin:/usr/bin:/bin", "HOME": str(base / "home"),
              "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
              "VLLM_CACHE_ROOT": str(base / "vllm-cache"), "TRITON_CACHE_DIR": str(base / "triton-cache"),
              "CUDA_CACHE_PATH": str(base / "cuda-cache")}
    receipt = {"production": False, "suite": options.suite, "started_at": time.time(), "services": [], "batches": [], "sha256": {}}
    # Retain exact source and fixtures for replay, not just the current working tree.
    for directory in ("src/ispec", "scripts", "tests/behavioral"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json", ".md", ".jinja"}:
                continue
            rel = path.relative_to(ROOT)
            if "local" in rel.parts or "__pycache__" in rel.parts:
                continue
            dest = base / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            receipt["sha256"][str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with endpoint(base, "jeff", lambda fd, port: [sys.executable, str(ROOT / "scripts/local_test_port.py"),
                      "--fd", str(fd), "--cpu-threads", "1"],
                      {**common, "CUDA_VISIBLE_DEVICES": "", "JEFF_DEVICE": "cpu",
                       "JEFF_CHECKPOINT": "/mnt/e/tools/jeff/jeff-base-v1.3"}, 18765, receipt) as (jeff_port, _):
            for label, checkpoint, parser in MODELS:
                occupancy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
                if occupancy:
                    raise RuntimeError("GPU compute workload present; refusing to start")
                model = "regression-" + label

                def command(fd, port):
                    argv = [options.gpu_python, str(Path(__file__).resolve()), "--vllm-child", str(fd),
                            "--model", checkpoint, "--served-model-name", model, "--host", "127.0.0.1", "--port", str(port),
                            "--dtype", "float16", "--max-model-len", "8192", "--max-num-seqs", "1",
                            "--gpu-memory-utilization", "0.8", "--enforce-eager", "--attention-backend", "TRITON_ATTN",
                            "--generation-config", "vllm", "--enable-auto-tool-choice", "--tool-call-parser", parser,
                            "--no-enable-log-requests"]
                    if parser == "olmo3":
                        argv += ["--chat-template", str(ROOT / "tests/behavioral/data/model_followup_20261025/olmo_instruct_tools_defined.jinja")]
                    return argv

                with endpoint(base, label, command, {**common, "CUDA_VISIBLE_DEVICES": "0"}, 18030, receipt) as (port, _):
                    run_tests(base, label, port, jeff_port, model, parser, receipt, suite=options.suite)
                    if options.suite == "full":
                        run_matrix(base, label, port, model, parser, receipt)
                time.sleep(3)
    finally:
        receipt["ended_at"] = time.time()
        (base / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--vllm-child":
        vllm_child(int(sys.argv[2]), sys.argv[3:])
    else:
        main()
