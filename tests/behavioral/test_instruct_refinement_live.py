"""Paired Instruct trials; temporary DBs, real iSPEC controller, no external sends.

ISPEC_INSTRUCT_TRIAL selects baseline, protocol, schema, or candidate.
ISPEC_INSTRUCT_SPLIT selects dev or held. Freeze candidates before running held.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from ispec.api.routes import support
from ispec.assistant import service
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.db.connect import get_session
from ispec.db.models import Person, Project, ProjectAccessMode, ProjectComment, UserRole
from ispec.prompt.loader import render_prompt
from ispec.schedule.connect import get_schedule_session
from tests.behavioral.test_model_runtime_followup_live import TOOL, isolated_runtime


pytestmark = [pytest.mark.behavioral, pytest.mark.skipif(
    os.getenv("ISPEC_RUN_LLM_TESTS") != "1", reason="Explicit isolated inference only")]
DATA = json.loads((Path(__file__).parent / "data/model_followup_20261025/instruct_refinement_cases.json").read_text())
TRIAL = os.getenv("ISPEC_INSTRUCT_TRIAL", "baseline")
SPLIT = os.getenv("ISPEC_INSTRUCT_SPLIT", "dev")
assert TRIAL in {"baseline", "protocol", "schema", "candidate"}
assert SPLIT in {"dev", "held"}
PERSON_DESCRIPTION = (
    "Optional person attribution. Omit unless a numeric person id was explicitly supplied by the user "
    "or returned by a person lookup. Do not guess it from project, account or session ids. "
    "Omission uses the iSPEC Assistant person record."
)


def cases(kind):
    return [case for case in DATA[kind] if case["split"] == SPLIT]


def emit(kind, case, **data):
    print("\nREFINEMENT " + json.dumps({"trial": TRIAL, "split": SPLIT, "kind": kind,
        "case": case["id"], **data}, default=str), flush=True)


@pytest.fixture
def trial_runtime(isolated_runtime, monkeypatch):
    if TRIAL == "protocol":
        text = render_prompt("experimental.tools.olmo3_refinement").text
        monkeypatch.setattr(service, "_native_tool_protocol", lambda: text)
    if TRIAL == "candidate":
        monkeypatch.setenv("ISPEC_VLLM_TOOL_PREAMBLE_FAMILY", "assistant.tools.olmo3_preamble")
    return isolated_runtime


@pytest.mark.parametrize("case", cases("native"), ids=lambda c: c["id"])
def test_auto_tool_selection(trial_runtime, case):
    system = "Use the supplied tool to answer; do not guess."
    if TRIAL == "protocol":
        system += "\n" + service._native_tool_protocol()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": case["text"]}]
    start = time.monotonic()
    first = service.generate_reply(messages=messages, tools=[TOOL], tool_choice="auto", timeout_seconds=30)
    emit("native_first", case, reply=first.__dict__, messages_sha256=hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode()).hexdigest())
    assert first.ok, first.error
    if case["project_id"] is None:
        assert not first.tool_calls
        return
    assert len(first.tool_calls or []) == 1
    call = first.tool_calls[0]
    assert call["function"]["name"] == "get_project"
    assert json.loads(call["function"]["arguments"]) == {"project_id": case["project_id"]}
    messages.extend([
        {"role": "assistant", "content": first.content, "tool_calls": first.tool_calls},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(
            {"project_id": case["project_id"], "sample_count": case["sample_count"]})},
    ])
    second = service.generate_reply(messages=messages, tools=[TOOL], tool_choice="none", timeout_seconds=30)
    emit("native_final", case, reply=second.__dict__, elapsed_ms=round((time.monotonic()-start)*1000))
    assert second.ok and not second.tool_calls
    assert str(case["sample_count"]) in second.content or {
        8: "eight", 11: "eleven", 19: "nineteen"}[case["sample_count"]] in second.content.lower()


@pytest.mark.parametrize("identified", [True, False], ids=["explicit_id", "missing_id"])
def test_project_lookup_controller(trial_runtime, monkeypatch, identified):
    """Post-held-out diagnostic with the real schema, not the wire-test stub."""
    store = trial_runtime
    original_schemas = support.openai_tools_for_user
    monkeypatch.setattr(support, "openai_tools_for_user", lambda user: [
        tool for tool in original_schemas(user) if tool["function"]["name"] == "get_project"])
    original_run = support.run_tool
    calls = []

    def run(**kw):
        assert kw["name"] == "get_project"
        assert str(kw["core_db"].bind.url.database) == str(store.core_db_path)
        calls.append({"name": kw["name"], "arguments": copy.deepcopy(kw["args"])})
        return original_run(**kw)

    monkeypatch.setattr(support, "run_tool", run)
    user = SimpleNamespace(id=1, username="fixture", role=UserRole.viewer,
        project_access_mode=ProjectAccessMode.all_projects, can_write_project_comments=False)
    with get_session(store.core_db_path) as core, get_assistant_session(store.assistant_db_path) as adb, get_schedule_session(store.schedule_db_path) as sdb:
        project_id = store.project_ids[1]
        title = core.get(Project, project_id).prj_ProjectTitle
        session = SupportSession(session_id="lookup", user_id=1, state_json="{}")
        adb.add(session)
        adb.commit()
        text = (f"Please check the database and tell me the title for project {project_id}."
                if identified else "Please check the database and tell me the title for that project.")
        start = time.monotonic()
        result = support.chat(support.ChatRequest(sessionId=session.session_id, message=text, history=[]),
            assistant_db=adb, core_db=core, schedule_db=sdb, user=user)
        meta = json.loads(adb.get(SupportMessage, result.messageId).meta_json)
        emit("controller_lookup", {"id": "explicit_id" if identified else "missing_id"},
             response=result.message, metadata=meta, calls=calls,
             elapsed_ms=round((time.monotonic()-start)*1000))
        assert core.query(ProjectComment).count() == 0
        if identified:
            assert calls == [{"name": "get_project", "arguments": {"id": project_id}}]
            assert title in result.message
        else:
            assert not calls, "A project lookup needs a grounded identifier"


@pytest.mark.parametrize("case", cases("controller"), ids=lambda c: c["id"])
def test_comment_controller(trial_runtime, monkeypatch, case):
    store = trial_runtime
    allowed = {"count_all_projects", "get_project", "create_project_comment"}
    original_schemas = support.openai_tools_for_user
    def schemas(user):
        tools = copy.deepcopy([t for t in original_schemas(user) if t["function"]["name"] in allowed])
        if TRIAL in {"schema", "candidate"}:
            create = next(t for t in tools if t["function"]["name"] == "create_project_comment")
            create["function"]["parameters"]["properties"]["person_id"]["description"] = PERSON_DESCRIPTION
        return tools
    monkeypatch.setattr(support, "openai_tools_for_user", schemas)
    original_run = support.run_tool
    calls = []
    def run(**kw):
        assert kw["name"] in allowed
        assert str(kw["core_db"].bind.url.database) == str(store.core_db_path)
        calls.append({"name": kw["name"], "arguments": copy.deepcopy(kw["args"])})
        return original_run(**kw)
    monkeypatch.setattr(support, "run_tool", run)
    user = SimpleNamespace(id=1, username="fixture", role=UserRole.viewer,
        project_access_mode=ProjectAccessMode.all_projects, can_write_project_comments=True)
    with get_session(store.core_db_path) as core, get_assistant_session(store.assistant_db_path) as adb, get_schedule_session(store.schedule_db_path) as sdb:
        core.add_all([Person(id=2, ppl_Name_First="Unrelated", ppl_Name_Last="Person", ppl_AddedBy="fixture"),
                      Person(id=7, ppl_Name_First="Requested", ppl_Name_Last="Person", ppl_AddedBy="fixture")])
        core.commit()
        project_id = store.project_ids[0]
        session = SupportSession(session_id="refinement", user_id=1,
            state_json=json.dumps({"current_project_id": project_id}))
        adb.add(session); adb.flush()
        adb.add(SupportMessage(session_pk=session.id, role="assistant", content=
            f'Draft for project {project_id}: "{case["draft"]}" Shall I save this exact note to project history?'))
        adb.commit()
        start = time.monotonic()
        result = support.chat(support.ChatRequest(sessionId=session.session_id, message=case["reply"], history=[]),
            assistant_db=adb, core_db=core, schedule_db=sdb, user=user)
        meta = json.loads(adb.get(SupportMessage, result.messageId).meta_json)
        comments = core.query(ProjectComment).all()
        emit("controller", case, response=result.message, metadata=meta, calls=calls,
            comments=[{"project_id":c.project_id,"comment":c.com_Comment,"person_id":c.person_id} for c in comments],
            elapsed_ms=round((time.monotonic()-start)*1000))
        assert len(comments) == int(case["save"])
        writes = [c for c in calls if c["name"] == "create_project_comment"]
        if case["save"]:
            assert len(writes) == 1
            args = writes[0]["arguments"]
            assert args["confirm"] is True
            assert comments[0].project_id == project_id
            assert comments[0].com_Comment == case["draft"]
            if case["person_id"] is None:
                assert "person_id" not in args, "Unprovided optional identity was guessed"
                person = core.get(Person, comments[0].person_id)
                assert (person.ppl_Name_First, person.ppl_Name_Last) == ("iSPEC", "Assistant")
            else:
                assert args["person_id"] == comments[0].person_id == case["person_id"]
            assert meta["project_comment_approval_eval"]["write_outcome"]["status"] == "succeeded"
        else:
            assert not writes


@pytest.mark.parametrize("case", cases("notes"), ids=lambda c: c["id"])
def test_note_proposal(trial_runtime, case):
    family = "experimental.slack.note_proposal." + ("grounded" if TRIAL == "candidate" else "compact")
    prompt = render_prompt(family)
    inputs = {key: case[key] for key in ("date", "project_id", "thread", "existing_comments")}
    messages = [{"role": "system", "content": prompt.text}, {"role": "user", "content": json.dumps(inputs)}]
    start = time.monotonic()
    reply = service.generate_reply(messages=messages, tools=None, tool_choice="none", timeout_seconds=30,
        vllm_extra_body={"max_tokens":512, "temperature":0})
    emit("note", case, reply=reply.__dict__, prompt_family=family, prompt_sha256=prompt.source.body_sha256,
         messages_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest(),
         elapsed_ms=round((time.monotonic()-start)*1000))
    assert reply.ok and not reply.tool_calls
    value = json.loads(reply.content)
    assert set(value) == {"decision", "note", "reason"}
    assert all(isinstance(value[k],str) for k in value)
    assert value["decision"] == case["decision"]
    assert bool(value["note"].strip()) == (case["decision"] == "propose")
    text = value["note"].casefold()
    assert all(any(word.casefold() in text for word in group) for group in case["required"])
    assert not any(word.casefold() in text for word in case["forbidden"])
