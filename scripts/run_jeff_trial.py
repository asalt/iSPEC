"""Own one offline CPU Jeff test server, select a free port, run a trial, clean up."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from scripts.local_test_port import reserve_loopback_socket

ROOT = Path(__file__).resolve().parents[1]


def child_environment(*, output, checkpoint, port, threads):
    """Allowlisted environment only: no Slack, DB, proxy or auth variables inherited."""
    return {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(output / "isolated-home"),
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "JEFF_DEVICE": "cpu",
        "JEFF_CHECKPOINT": str(checkpoint),
        "JEFF_HOST": "127.0.0.1",
        "PORT": str(port),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
        "ISPEC_LOG_DIR": str(output / "logs"),
        "ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED": "0",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=["choices", "prompts", "baseline"], default="choices"
    )
    parser.add_argument("--start-port", type=int, default=18765)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=90)
    parser.add_argument("--max-run-seconds", type=int, default=1800)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("The fd-handoff launcher requires POSIX")
    if min(args.cpu_threads, args.request_timeout, args.max_run_seconds) < 1:
        parser.error("Thread and timeout settings must be positive")
    checkpoint = args.checkpoint.resolve(strict=True)
    for name in ("config.json", "decision_config.json", "tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            parser.error(f"Existing standard checkpoint is missing {name}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "started_at": time.time(),
        "device": "cpu",
        "requested_cpu_threads": args.cpu_threads,
        "checkpoint": str(checkpoint),
        "suite": args.suite,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "jeff")
        },
        "checkpoint_metadata_sha256": {
            name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            for name in ("config.json", "decision_config.json", "tokenizer_config.json")
        },
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "server_helper_sha256": hashlib.sha256(
            (ROOT / "scripts/local_test_port.py").read_bytes()
        ).hexdigest(),
    }
    server, port = None, None
    try:
        with reserve_loopback_socket(args.start_port) as reserved, (
            output / "server.log"
        ).open("x") as log:
            port = reserved.getsockname()[1]
            receipt["port"] = port
            env = child_environment(
                output=output,
                checkpoint=checkpoint,
                port=port,
                threads=args.cpu_threads,
            )
            server = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/local_test_port.py"),
                    "--fd",
                    str(reserved.fileno()),
                    "--cpu-threads",
                    str(args.cpu_threads),
                ],
                cwd=ROOT,
                env=env,
                pass_fds=(reserved.fileno(),),
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            receipt["owned_pid"] = server.pid
            reserved.close()
            print(json.dumps({"owned_pid": server.pid, "port": port}), flush=True)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Owned server exited; inspect server.log")
                try:
                    with opener.open(
                        f"http://127.0.0.1:{port}/health", timeout=1
                    ) as response:
                        health = json.load(response)
                    if health.get("status") == "ready":
                        receipt["health"] = health
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Owned server not ready within 180 seconds")
            flags = {
                "choices": ["--choice-sweep"],
                "prompts": ["--prompt-sweep"],
                "baseline": [],
            }[args.suite]
            flags += [item for case in args.case for item in ("--case", case)]
            with (output / "evaluation.log").open("x") as evaluation_log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.eval_jeff_approval",
                        "--url",
                        f"http://127.0.0.1:{port}",
                        "--output-dir",
                        str(output / "evaluation"),
                        "--timeout",
                        str(args.request_timeout),
                        *flags,
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=evaluation_log,
                    stderr=subprocess.STDOUT,
                    timeout=args.max_run_seconds,
                )
            receipt["evaluation_exit_code"] = result.returncode
            return result.returncode
    except BaseException as exc:
        receipt["interrupted_by"] = type(exc).__name__
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        receipt["server_exit_code"] = server.returncode if server else None
        receipt["ended_at"] = time.time()
        if port is not None:
            with socket.socket() as sock:
                receipt["port_closed"] = sock.connect_ex(("127.0.0.1", port)) != 0
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"cleanup": receipt}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
