"""Project retrieval through support.chat. Real input is opt-in, file-local only."""

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ispec.api.routes import support
from ispec.api.security import _ApiKeyServiceUser
from ispec.assistant.connect import get_assistant_session
from ispec.db.connect import get_session
from ispec.db.models import AuthUser, Project
from ispec.schedule.connect import get_schedule_session
from tests.behavioral.test_olmo_application_live import application_store  # noqa: F401

pytestmark = [
    pytest.mark.behavioral,
    pytest.mark.skipif(
        os.getenv("ISPEC_RUN_LLM_TESTS") != "1",
        reason="Opt-in local model run",
    ),
]
CASES = {
    "pi": ("Which project number belongs to PI Mira Ellerton?", 731),
    "contact": (
        "I only remember the lab contact, Rowan Fenwick. Which project was it?",
        732,
    ),
    "topic": ("What number did we give the COVID-19 project?", 733),
    "title": ("Which project had 'winter orchard' in the title?", 732),
    "typo": ("What was the project number for lab contact Rowan Fenwck?", 732),
    "missing": ("Which project number is for the silver quasar lattice?", None),
}
RECORDS = [
    dict(id=731, prj_ProjectTitle="Cedar canopy profiling", prj_PI="Mira Ellerton"),
    dict(
        id=732,
        prj_ProjectTitle="Winter orchard survey",
        prj_Project_LabContact="Rowan Fenwick",
    ),
    dict(id=733, prj_ProjectTitle="SARS-CoV-2 archive inventory"),
]


@contextmanager
def catalog_session(store, *, backup, records):
    if backup is None:
        with get_session(store.core_db_path) as core:
            core.add_all(Project(prj_AddedBy="fixture", **row) for row in records)
            core.flush()
            yield core
        return
    # Only a completed offline snapshot; immutable prevents even SQLite sidecar writes.
    wal = Path(str(backup) + "-wal")
    assert backup.is_file() and (not wal.exists() or wal.stat().st_size == 0)
    engine = create_engine(f"sqlite:///{backup.as_uri()}?mode=ro&immutable=1&uri=true")
    try:
        with Session(engine) as core:
            yield core
    finally:
        engine.dispose()


@pytest.mark.parametrize("case", list(CASES) + ["real"])
def test_project_search(case, application_store, monkeypatch, tmp_path):  # noqa: F811
    records = list(RECORDS)
    origin = "hand_authored_synthetic"
    backup = ui = user_id = None
    if case == "real":
        path = os.getenv("ISPEC_LOCAL_PROJECT_SEARCH_CASE")
        if not path:
            pytest.skip("No private production example supplied")
        fixture = json.loads(Path(path).read_text())
        assert fixture["provenance"] == "real_production"
        message, expected = fixture["message"], fixture["record"]["id"]
        records.append(fixture["record"])
        origin = "real_production_with_synthetic_distractors"
        if path := os.getenv("ISPEC_LOCAL_SEARCH_BACKUP"):
            backup = Path(path).resolve(strict=True)
            user_id = fixture["user"]["id"]
            ui = support.UIContext(**fixture["ui_route"])
            origin = "real_production_on_read_only_backup"
    else:
        message, expected = CASES[case]

    calls = []
    executor = support.run_tool

    def observe(**kwargs):
        assert kwargs["name"] in {
            "search_projects",
            "my_projects",
            "get_project",
            "projects",
            "search_people",
            "assistant_list_tools",
        }
        result = executor(**kwargs)
        calls.append(
            {"name": kwargs["name"], "arguments": kwargs["args"], "result": result}
        )
        return result

    monkeypatch.setattr(support, "run_tool", observe)
    store = application_store
    with (
        catalog_session(store, backup=backup, records=records) as core,
        get_assistant_session(store.assistant_db_path) as assistant,
        get_schedule_session(store.schedule_db_path) as schedule,
    ):
        actor = (
            core.get(AuthUser, user_id)
            if backup
            else _ApiKeyServiceUser(can_write_project_comments=True)
        )
        assert actor is not None
        reply = support.chat(
            support.ChatRequest(
                sessionId="lookup-" + case, message=message, history=[], ui=ui
            ),
            core_db=core,
            assistant_db=assistant,
            schedule_db=schedule,
            user=actor,
        ).message
    evidence = {
        "case": case,
        "origin": origin,
        "expected": expected,
        "calls": calls,
        "turns": [{"user": message, "assistant": reply}],
    }
    (tmp_path / "search.json").write_text(json.dumps(evidence, indent=2))
    searches = [
        c["result"]["result"]
        for c in calls
        if c["name"] in {"my_projects", "search_projects"} and c["result"]["ok"]
    ]
    assert searches, reply
    rows = [
        p
        for result in searches
        for p in result.get("matches", result.get("projects", []))
    ]
    if expected is None:
        assert not rows and not re.search(r"\b\d+\b", reply), reply
        assert re.search(
            r"no |not |couldn.t|unable|didn.t|could not", reply, re.I
        ), reply
    else:
        assert rows and rows[0]["id"] == expected and rows[0]["evidence"], rows
        assert re.search(rf"\b(?:MSPC0*)?{expected}\b", reply, re.I), reply
