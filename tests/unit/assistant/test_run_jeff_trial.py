import io
import json
from pathlib import Path
import signal
import socket
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_jeff_trial as runner


def test_child_environment_never_inherits_auth_or_gpu_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "synthetic-do-not-inherit")
    monkeypatch.setenv("JEFF_API_KEY", "synthetic-do-not-inherit")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = runner.child_environment(
        output=tmp_path, checkpoint=Path("/example"), port=18766, threads=1
    )
    assert "SLACK_BOT_TOKEN" not in env and "JEFF_API_KEY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["JEFF_DEVICE"] == "cpu"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PORT"] == "18766"


@pytest.mark.parametrize("timeout", [False, True])
def test_launcher_records_outcome_and_cleans_up_only_owned_child(
    monkeypatch, tmp_path, timeout
):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    for name in ("config.json", "decision_config.json", "tokenizer_config.json"):
        (checkpoint / name).write_text("{}")
    output = tmp_path / "trial"
    calls = []
    # Keep a separate listener alive across launch and cleanup.
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    start_port = occupied.getsockname()[1]
    occupied.listen(1)

    class Child:
        pid = 123456789
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15
            return self.returncode

    child = Child()

    def popen(command, **kwargs):
        assert "--fd" in command and kwargs["pass_fds"]
        assert kwargs["env"]["JEFF_DEVICE"] == "cpu"
        assert int(kwargs["env"]["PORT"]) > start_port
        return child

    def run(command, **kwargs):
        assert command[1:3] == ["-m", "scripts.eval_jeff_approval"]
        assert command[-2:] == ["--case", "example_confirm"]
        url = command[command.index("--url") + 1]
        assert int(url.rsplit(":", 1)[1]) > start_port
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(
        runner.urllib.request,
        "build_opener",
        lambda *_: SimpleNamespace(
            open=lambda *args, **kwargs: io.StringIO('{"status":"ready"}')
        ),
    )
    args = [
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--start-port",
        str(start_port),
        "--case",
        "example_confirm",
    ]
    try:
        if timeout:
            with pytest.raises(subprocess.TimeoutExpired):
                runner.main(args)
        else:
            assert runner.main(args) == 1
        with socket.create_connection(("127.0.0.1", start_port), timeout=1):
            pass
    finally:
        occupied.close()
    assert calls == [(child.pid, signal.SIGTERM)]
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["owned_pid"] == child.pid and receipt["port_closed"]
    assert receipt["port"] > start_port
    assert receipt.get("interrupted_by") == ("TimeoutExpired" if timeout else None)
