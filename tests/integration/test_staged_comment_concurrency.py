"""Concurrent staged saves across request sessions, without external IO."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.orm import Session

from ispec.assistant.models import AssistantBase, SupportMessage, SupportSession
from ispec.assistant.project_comment_approval import ProjectCommentApprovalSettings
from ispec.assistant.project_comment_workflow import ProjectCommentWorkflow
from ispec.assistant.service import AssistantReply
from ispec.assistant.tools import run_tool
from ispec.db.models import Base, Project, ProjectAccessMode, ProjectComment, ProjectCommentDraft, UserRole, sqlite_engine


def test_concurrent_save_requests_commit_one_comment(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", "0")
    monkeypatch.delenv("ISPEC_ASSISTANT_ALLOWED_WRITE_TOOLS", raising=False)
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "0")
    core_engine = sqlite_engine(f"sqlite:///{tmp_path / 'core.db'}")
    assistant_engine = sqlite_engine(f"sqlite:///{tmp_path / 'assistant.db'}")
    Base.metadata.create_all(core_engine)
    AssistantBase.metadata.create_all(assistant_engine)
    user = SimpleNamespace(id=1, username="test", role=UserRole.editor,
                           project_access_mode=ProjectAccessMode.all_projects)
    settings = ProjectCommentApprovalSettings(mode="own")
    with Session(core_engine) as core, Session(assistant_engine) as adb:
        core.add(Project(id=57, prj_AddedBy="test", prj_ProjectTitle="Synthetic"))
        core.commit()
        session = SupportSession(session_id="s")
        adb.add(session)
        adb.flush()
        initial = SupportMessage(session_pk=session.id, role="user", content="Prepare a note")
        adb.add(initial)
        adb.commit()
        workflow = ProjectCommentWorkflow(core_db=core, assistant_db=adb, user=user, session_id="s",
            turn_id=initial.id, user_message=initial.content, settings=settings,
            generate_reply_fn=None, generate_classifier_reply_fn=None)
        draft_id = workflow.prepare(args={"project_id": 57, "comment": "Samples arrived."})["result"]["draft_id"]
        workflow.run(name="offer_project_comment_save", args={"draft_id": draft_id})
        presented = SupportMessage(session_pk=session.id, role="assistant", content=workflow.response.text)
        adb.add(presented)
        adb.flush()
        presented.meta_json = json.dumps({"project_comment_presentation": workflow.record_delivery(message=presented)})
        approval = SupportMessage(session_pk=session.id, role="user", content="Yes, save it.")
        adb.add(approval)
        adb.commit()
        turn_id = approval.id

    barrier = Barrier(2)
    def submit():
        with Session(core_engine) as core, Session(assistant_engine) as adb:
            def classify(**kwargs):
                assert not core.in_transaction()
                barrier.wait(timeout=10)
                return AssistantReply(content='{"label":"approve_save","confidence":0.99,"reason":"Approved"}',
                                      provider="stub", model="stub")
            workflow = ProjectCommentWorkflow(core_db=core, assistant_db=adb, user=user, session_id="s",
                turn_id=turn_id, user_message="Yes, save it.", settings=settings,
                generate_reply_fn=None, generate_classifier_reply_fn=classify)
            return run_tool(name="create_project_comment", args={"draft_id": draft_id, "confirm": True},
                            core_db=core, user=user, project_comment_workflow=workflow)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit) for _ in range(2)]
            results = [future.result(timeout=20) for future in futures]
        assert sum(bool(result["ok"]) for result in results) == 1, results
        assert next(result for result in results if not result["ok"])["error_code"] == "draft_changed"
        with Session(core_engine) as core:
            comments = core.scalars(select(ProjectComment)).all()
            assert len(comments) == 1
            draft = core.get(ProjectCommentDraft, draft_id)
            assert draft.status == "saved"
            assert draft.comment_id == comments[0].id
    finally:
        assistant_engine.dispose()
        core_engine.dispose()
