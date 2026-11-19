"""Held-out synthetic cases using the unchanged full support.chat conversation harness."""

import json
from pathlib import Path

import pytest

from tests.behavioral import test_olmo_application_live as application
from tests.behavioral.test_olmo_application_live import application_store as application_store


pytestmark = application.pytestmark
CASES = Path(__file__).parent / "data" / "blind_r1_20261118.json"


@pytest.mark.parametrize(
    "case", [application.ApplicationCase(**row) for row in json.loads(CASES.read_text())["cases"]],
    ids=lambda case: case.id,
)
def test_blind_conversation(application_store, monkeypatch, tmp_path, case):  # noqa: F811 - pytest fixture injection
    try:
        application.test_application_conversation(application_store, monkeypatch, tmp_path, case)
    finally:
        # The shared historical harness labels non-control runs as development retests.
        path = tmp_path / "conversation.json"
        if path.is_file():
            record = json.loads(path.read_text())
            record["evidence_role"] = "r1_first_blind_run"
            path.write_text(json.dumps(record, indent=2) + "\n")
