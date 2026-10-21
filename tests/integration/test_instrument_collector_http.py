from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from ispec.instruments.collector import (
    CollectorConfig,
    blank_observation,
    send_observation,
)


def test_phone_home_uses_only_ingest_endpoint_and_checks_receipt(tmp_path):
    calls = []
    event = blank_observation(
        CollectorConfig(collector_id="test", instrument_id="test", root=str(tmp_path))
    )
    token = "isolated-test-secret-000000000000000"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), value))
            body = json.dumps(
                {"event_id": value["event_id"], "stored": True, "duplicate": False}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        receipt = send_observation(
            f"http://127.0.0.1:{server.server_port}", event, token
        )
        assert receipt["stored"] is True
        assert calls == [("/api/instruments/observations", "Bearer " + token, event)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
