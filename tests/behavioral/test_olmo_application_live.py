"""Working-tree OLMo evaluation through unmodified support.chat and native tools.

All cases are hand-authored synthetic evaluation data, except S02, which is an
anonymized derivative of a real Slack thread. The first 27 are retained
development cases. Only storage, network isolation, and execution observation
differ from the application.
"""

import copy
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time

import pytest

from ispec.api.routes import support
from ispec.api.security import _ApiKeyServiceUser
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.project_comment_workflow import COMMENT_WORKFLOW_TOOLS
from ispec.db.connect import get_session
from ispec.db.models import Experiment, ExperimentRun, Person, Project, ProjectComment, ProjectCommentDraft
from ispec.schedule.connect import get_schedule_session
from tests.behavioral.staged_comment_evidence import RecordedComment, RecordedDraft, RecordedTurn


pytestmark = [
    pytest.mark.behavioral,
    pytest.mark.skipif(os.getenv("ISPEC_RUN_LLM_TESTS") != "1", reason="Explicit local inference only"),
]


@dataclass(frozen=True, kw_only=True)
class ApplicationCase:
    id: str
    category: str
    messages: list[str]
    comment_counts: list[int]
    fresh: bool = True
    revised_turns: list[int] = field(default_factory=list)
    cancelled: bool = False
    expected_tool: str | None = None
    no_tools: bool = False
    no_draft_turns: list[int] = field(default_factory=list)
    data_origin: str = "hand_authored_synthetic"


GATE = [ApplicationCase(**row) for row in json.loads(
    (Path(__file__).parent / "data/staged_promotion_20261107.json").read_text()
)["cases"]]

CONTROLS = [
    ApplicationCase(
        id="C01_project", category="project lookup",
        messages=["Look up project 57 in iSPEC and tell me its title and current status."],
        comment_counts=[0], expected_tool="get_project",
    ),
    ApplicationCase(
        id="C02_person", category="person lookup",
        messages=["Look up person ID 73 and tell me their name and institution."],
        comment_counts=[0], expected_tool="get_person",
    ),
    ApplicationCase(
        id="C03_runs", category="experiment run listing",
        messages=["List the two latest experiment runs in iSPEC, including their run and search numbers."],
        comment_counts=[0], expected_tool="latest_experiment_runs",
    ),
    ApplicationCase(
        id="C04_greeting", category="greeting",
        messages=["Hi iSPEC, good morning!"], comment_counts=[0], no_tools=True,
    ),
    ApplicationCase(
        id="C05_missing_id", category="missing project identification",
        messages=["What is the current status of my project? I haven't told you which project yet."],
        comment_counts=[0], no_tools=True,
    ),
]

SCENARIOS = [ApplicationCase(**row) for row in json.loads(
    (Path(__file__).parent / "data/scenario_cases_20261120.json").read_text()
)["cases"]]

# The app decides which tools a turn offers; the test only keeps every data
# store inside the sandbox.
SANDBOX_PATHS = (
    "ISPEC_ANALYSIS_DB_PATH", "ISPEC_PSM_DB_PATH", "ISPEC_OMICS_DB_PATH",
    "ISPEC_AGENT_ARCHIVE_DB_PATH",
)
SANDBOX_DIRS = ("ISPEC_DB_DIR", "ISPEC_INSTRUMENT_MOUNT_ROOT", "ISPEC_INFERENCE_USAGE_LOG_DIR")
DATA_PATHS = SANDBOX_PATHS + SANDBOX_DIRS + (
    "ISPEC_DB_PATH", "ISPEC_ASSISTANT_DB_PATH", "ISPEC_SCHEDULE_DB_PATH",
    "ISPEC_AGENT_DB_PATH", "ISPEC_AGENT_STATE_DB_PATH", "ISPEC_PROMPTS_DB_PATH",
)


@pytest.fixture
def application_store(monkeypatch, behavioral_datastore, tmp_path):
    """Redirect persistence while retaining checked-in profile and app defaults."""
    for key, value in behavioral_datastore.env_overrides().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    for key in SANDBOX_PATHS:
        monkeypatch.setenv(key, str(tmp_path / f"{key.lower()}.db"))
    for key in SANDBOX_DIRS:
        (tmp_path / key.lower()).mkdir()
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    assert support.generate_reply.__module__ == "ispec.assistant.service"
    assert Path(support.__file__).is_relative_to(Path(os.environ["ISPEC_EVAL_SOURCE"]))
    assert os.environ["ISPEC_ASSISTANT_MAX_PROMPT_TOKENS"] == "6000"
    assert os.environ["ISPEC_ASSISTANT_MAX_OUTPUT_TOKENS"] == "1024"
    assert os.environ["ISPEC_VLLM_TOOL_PREAMBLE_FAMILY"] == "assistant.tools.olmo3_preamble"
    return behavioral_datastore


def seed_records(*, store) -> None:
    """Create explicit identities for lookups and keep unrelated records visible."""
    with get_session(store.core_db_path) as core:
        core.add(Project(id=57, prj_AddedBy="fixture", prj_ProjectTitle="Synthetic sample follow-up"))
        core.add(Project(
            id=64, prj_AddedBy="fixture", prj_ProjectTitle="Viral protein sequence verification",
            prj_PI="Dana Holt", prj_Project_LabContact="Sam Ortiz", prj_Status="closed",
        ))
        core.add(Person(
            id=73, ppl_AddedBy="fixture", ppl_Name_First="Morgan",
            ppl_Name_Last="Reed", ppl_Institution="Synthetic Research Institute",
        ))
        core.add(Experiment(id=58101, record_no="58101", project_id=57,
                            exp_Name="Synthetic serum profiling"))
        core.flush()
        core.add_all([
            ExperimentRun(id=901, experiment_id=58101, run_no=1, search_no=1),
            ExperimentRun(id=902, experiment_id=58101, run_no=2, search_no=3),
        ])


def count_findings(*, case, turns, calls) -> list[str]:
    """Score observed writes and explicit tool controls; retain prose for review."""
    findings = []
    if len(turns) != len(case.messages):
        findings.append(f"Incomplete conversation: {len(turns)}/{len(case.messages)} turns")
    for index, turn in enumerate(turns):
        error = turn.metadata.get("provider_meta", {}).get("error")
        if error:
            findings.append(f"Turn {index + 1}: inference failed: {error}")
        expected = case.comment_counts[index]
        if turn.comment_count != expected:
            findings.append(f"Turn {index + 1}: expected {expected} comments; observed {turn.comment_count}")
        if index in case.revised_turns:
            previous = turns[index - 1].pending_drafts if index else []
            current = turn.pending_drafts
            if not previous or not current or previous[-1].id == current[-1].id:
                findings.append(f"Turn {index + 1}: revision not registered")
    if case.cancelled and turns and not any(draft.status == "cancelled" for draft in turns[-1].drafts):
        findings.append("Cancellation not recorded")
    if case.expected_tool and not any(
        call.get("result", {}).get("tool", call["name"]) == case.expected_tool
        and call.get("result", {}).get("ok") is True
        for call in calls
    ):
        findings.append(f"No successful {case.expected_tool} call")
    if case.no_tools and calls:
        findings.append("Unexpected tool call for clarification/greeting control")
    for index in case.no_draft_turns:
        if any(call["turn"] == index and call["name"] in COMMENT_WORKFLOW_TOOLS for call in calls):
            findings.append(f"Turn {index + 1}: unexpected note tool call")
    return findings


@pytest.mark.parametrize("case", GATE + CONTROLS + SCENARIOS, ids=lambda case: case.id)
def test_application_conversation(application_store, monkeypatch, tmp_path, case):
    store = application_store
    seed_records(store=store)
    original_run = support.run_tool
    calls, turns, cards = [], [], []
    current_turn = 0

    def observed_tool(**kwargs):
        call = {"turn": current_turn, "name": kwargs["name"], "arguments": copy.deepcopy(kwargs["args"])}
        calls.append(call)
        sandbox = tmp_path
        for key in DATA_PATHS:
            assert Path(os.environ[key]).resolve().is_relative_to(sandbox), f"{key} outside sandbox"
        assert Path(kwargs["core_db"].bind.url.database) == store.core_db_path
        result = original_run(**kwargs)
        call["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(support, "run_tool", observed_tool)
    actor = _ApiKeyServiceUser(can_write_project_comments=True)
    report = {
        "case": case.id, "category": case.category, "case_definition": asdict(case),
        "data_origin": case.data_origin, "production": False,
        "evidence_role": "fresh_control" if case.id.startswith("C") else "development_retest",
        "tool_calls": calls, "cards_by_turn": cards,
        "expected_comment_counts": case.comment_counts,
        "source": str(Path(support.__file__).resolve()),
    }
    try:
        for current_turn, message in enumerate(case.messages):
            with (get_session(store.core_db_path) as core,
                  get_assistant_session(store.assistant_db_path) as adb,
                  get_schedule_session(store.schedule_db_path) as schedule):
                started = time.monotonic()
                response = support.chat(
                    support.ChatRequest(sessionId="application-" + case.id, message=message, history=[]),
                    assistant_db=adb, core_db=core, schedule_db=schedule, user=actor,
                )
                row = adb.get(SupportMessage, response.messageId)
                comments = core.query(ProjectComment).order_by(ProjectComment.id).all()
                drafts = core.query(ProjectCommentDraft).order_by(ProjectCommentDraft.created_at).all()
                turns.append(RecordedTurn(
                    user=message, assistant=response.message,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                    metadata=json.loads(row.meta_json),
                    comments=[RecordedComment(id=c.id, project_id=c.project_id, text=c.com_Comment, person_id=c.person_id)
                              for c in comments],
                    drafts=[RecordedDraft(id=d.id, text=d.comment, status=d.status, comment_id=d.comment_id)
                            for d in drafts],
                ))
                cards.append([card.model_dump(mode="json") for card in response.commentCards])
    except Exception as exc:
        report["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["turns"] = [asdict(turn) for turn in turns]
        report["mechanical_findings"] = count_findings(case=case, turns=turns, calls=calls)
        (tmp_path / "conversation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("\nAPPLICATION " + json.dumps({
            "case": case.id, "counts": [turn.comment_count for turn in turns],
            "calls": [call["name"] for call in calls], "findings": report["mechanical_findings"],
        }), flush=True)
    assert not report["mechanical_findings"], "\n".join(report["mechanical_findings"])
