from __future__ import annotations

from ispec.agent.long_task import inspect_long_task


def test_inspect_long_task_marks_sentinel_and_outputs_ready(tmp_path):
    sentinel = tmp_path / "done.ok"
    sentinel.write_text("done\n", encoding="utf-8")
    output = tmp_path / "report.pdf"
    output.write_bytes(b"%PDF-1.4\n")
    log_file = tmp_path / "job.log"
    log_file.write_text("finished\n", encoding="utf-8")

    state = inspect_long_task(
        log_file=log_file,
        output_paths=[output],
        sentinel_file=sentinel,
    )

    assert state["complete"] is True
    assert state["outputs_ready"] is True
    assert state["should_wake"] is True
    assert state["log"]["nonempty"] is True


def test_inspect_long_task_keeps_running_pid_from_waking(monkeypatch, tmp_path):
    output = tmp_path / "report.pdf"
    output.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("ispec.agent.long_task.pid_is_alive", lambda pid: True)
    state = inspect_long_task(pid=123, output_paths=[output])

    assert state["pid_alive"] is True
    assert state["complete"] is False
    assert state["should_wake"] is False


def test_inspect_long_task_requires_nonempty_outputs(tmp_path):
    sentinel = tmp_path / "done.ok"
    sentinel.write_text("done\n", encoding="utf-8")
    output = tmp_path / "empty.pdf"
    output.write_bytes(b"")

    state = inspect_long_task(output_paths=[output], sentinel_file=sentinel)

    assert state["complete"] is True
    assert state["outputs_ready"] is False
    assert state["should_wake"] is False
