"""Real CLI children, temporary databases, and a loopback-only legacy stub."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor import loop


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("ISPEC_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("ISPEC_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ISPEC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ISPEC_LEGACY_CONF", str(tmp_path / "no-credentials.conf"))
    set_main_thread(owner="pytest")
    return tmp_path


@pytest.mark.parametrize("http_status", [200, 500])
def test_cli_sync_child_keeps_dispatcher_responsive(isolated, http_status):
    entered, release = threading.Event(), threading.Event()
    paths = []
    # The real mapping is host-local and must not be required by a clean checkout.
    mapping_path = isolated / "mapping.json"
    mapping_path.write_text(json.dumps({"tables": {
        "iSPEC_Projects": {"pk": {"legacy": "prj_PRJRecNo"}, "modified_ts": "prj_ModificationTS"},
        "iSPEC_People": {"pk": {"legacy": "ppl_PeopleRecNo"}, "modified_ts": "ppl_ModificationTS"},
        "iSPEC_Experiments": {"pk": {"legacy": "exp_EXPRecNo"}, "modified_ts": "exp_ModificationTS"},
        "iSPEC_ExperimentRuns": {
            "pk": {"legacy": ["exprun_EXPRecNo", "exprun_EXPRunNo", "exprun_EXPSearchNo"]},
            "modified_ts": "exprun_ModificationTS",
        },
    }}), encoding="utf-8")

    class LegacyStub(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            entered.set()
            release.wait(15)
            body = json.dumps({"items": [], "has_more": False}).encode()
            self.send_response(http_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), LegacyStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with get_agent_session() as db:
        db.add(
            AgentRun(
                run_id="integration",
                agent_id="test",
                kind="supervisor",
                status="running",
            )
        )
    sync_id = loop._enqueue_command(
        command_type=COMMAND_LEGACY_SYNC_ALL,
        payload={
            "legacy_url": f"http://127.0.0.1:{server.server_port}",
            "mapping_path": str(mapping_path),
            "max_pages": 1,
            "max_project_comments": 0,
            "backfill_missing": False,
        },
    )
    processor = loop._SupervisorCommandProcessor(
        agent_id="test", run_id="integration", broker=None
    )
    try:
        assert processor.tick()
        assert processor._sync is not None
        worker = processor._sync.worker
        assert entered.wait(15), Path(worker.log_path).read_text()
        other_id = loop._enqueue_command(
            command_type="test_unknown_no_external_effects"
        )
        assert processor.tick()
        with get_agent_session() as db:
            assert db.get(AgentCommand, other_id).status == "failed"
            row = db.get(AgentCommand, sync_id)
            assert row.status == "running"
            assert row.result_json["worker"]["pid"] != os.getpid()
            assert row.result_json["sync_run"]["status"] == "running"
        release.set()
        exit_code = 0 if http_status == 200 else 1
        outcome = "succeeded" if http_status == 200 else "failed"
        assert worker.process.wait(timeout=20) == exit_code, Path(
            worker.log_path
        ).read_text()
        assert processor.tick()
        with get_agent_session() as db:
            row = db.get(AgentCommand, sync_id)
            assert row.status == outcome
            assert row.result_json["worker"]["exit_code"] == exit_code
            assert row.result_json["sync_run"]["status"] == outcome
        assert len(paths) == (4 if http_status == 200 else 1)
        assert all(path.startswith("/api/v2/legacy/tables/") for path in paths)
    finally:
        release.set()
        processor.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_sigterm_updates_state_without_health_steps(isolated):
    log_path = isolated / "supervisor.log"
    state_path = isolated / "state" / "supervisor.json"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "ispec", "supervisor", "run"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if state_path.exists():
                    # Wait until startup has committed the run, not just its PID.
                    with get_agent_session() as db:
                        if db.query(AgentRun).count():
                            break
                assert process.poll() is None, log_path.read_text()
                time.sleep(0.05)
            else:
                pytest.fail(log_path.read_text())
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == 0, log_path.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    assert json.loads(state_path.read_text())["status"] == "stopped"
    assert not state_path.with_suffix(".pid").exists()
    with get_agent_session() as db:
        assert db.query(AgentRun).one().status == "stopped"
        assert db.query(AgentStep).count() == 0
