"""Exercise a dedicated comment reader without Slack, a model or live data."""

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from ispec.api.routes import auth, routes, support
from ispec.api.security import (
    create_session,
    require_access,
    require_assistant_access,
    session_cookie_name,
)
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.service import AssistantReply
from ispec.db.connect import get_session_dep, make_session_factory, sqlite_engine
from ispec.db.models import AuthUser, Person, Project, ProjectComment, UserRole
from ispec.schedule.connect import get_schedule_session

pytestmark = pytest.mark.testclient


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "1")
    monkeypatch.setenv("ISPEC_API_KEY", "synthetic-service-key")
    sessions = make_session_factory(sqlite_engine(f"sqlite:///{tmp_path}/core.db"))
    app = FastAPI()
    app.include_router(auth.router, prefix="/api")
    app.include_router(
        routes.router, prefix="/api", dependencies=[Depends(require_access)]
    )

    @app.post("/api/support/chat")
    def stub_model(user=Depends(require_assistant_access)):
        return {
            "speaker": user.username,
            "write_authority": user.can_write_project_comments,
        }

    def database():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_session_dep] = database
    with sessions() as db:
        person = Person(
            ppl_AddedBy="fixture", ppl_Name_First="Test", ppl_Name_Last="Person"
        )
        db.add(person)
        for i in (23, 58):
            project = Project(
                id=i, prj_AddedBy="fixture", prj_ProjectTitle=f"Synthetic {i}"
            )
            db.add(
                ProjectComment(project=project, person=person, com_Comment=f"Note {i}")
            )
        admin = AuthUser(
            username="test-admin",
            role=UserRole.admin,
            password_hash="x",
            password_salt="x",
        )
        db.add(admin)
        db.flush()
        admin_cookie = create_session(db, user=admin)
    with TestClient(app) as client:
        client.cookies.set(session_cookie_name(), admin_cookie)
        yield client


def enroll(client, name, agent):
    body = {
        "username": name,
        "role": "viewer",
        "password": "synthetic-test-only",
        "is_comment_agent": agent,
    }
    assert client.post("/api/auth/users", json=body).status_code == 201
    client.cookies.clear()
    assert client.post("/api/auth/login", json=body).status_code == 200


def test_agent_reads_all_project_comments_and_chats_but_cannot_act(api):
    enroll(api, "fixture-agent", True)
    assert api.get("/api/auth/me").json()["is_comment_agent"] is True
    for i in (23, 58):
        rows = api.get(f"/api/project_comment/by_project/{i}?limit=1").json()
        assert [row["com_Comment"] for row in rows] == [f"Note {i}"]
    assert api.post("/api/support/chat").json() == {
        "speaker": "fixture-agent",
        "write_authority": False,
    }
    assert api.get("/api/auth/users").status_code == 403
    assert api.get("/api/projects/23").status_code == 403
    assert api.delete("/api/project_comment/1").status_code == 403
    assert api.post("/api/project_comment/", json={}).status_code == 403


def test_ordinary_viewer_and_service_key_do_not_inherit_comment_reads(api):
    enroll(api, "ordinary", False)
    assert api.get("/api/project_comment/by_project/23").status_code == 403
    api.cookies.clear()
    assert (
        api.get(
            "/api/project_comment/by_project/23",
            headers={"X-API-Key": "synthetic-service-key"},
        ).status_code
        == 401
    )


def test_agent_cannot_be_created_with_elevated_role(api):
    body = {
        "username": "bad",
        "password": "synthetic-test-only",
        "role": "admin",
        "is_comment_agent": True,
    }
    assert api.post("/api/auth/users", json=body).status_code == 422


def test_chat_continues_own_session_only(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_ROUTER_MODE", "off")
    monkeypatch.setattr(
        support,
        "generate_reply",
        lambda **kwargs: AssistantReply(
            content="Hello.", provider="stub", model="test"
        ),
    )
    user = AuthUser(
        id=19, role=UserRole.viewer, is_comment_agent=True, username="synthetic"
    )
    with get_assistant_session(
        tmp_path / "assistant.db"
    ) as session, get_schedule_session(tmp_path / "schedule.db") as schedule:

        def send(thread):
            return support.chat(
                support.ChatRequest(sessionId=thread, message="Hello!"),
                core_db=db_session,
                assistant_db=session,
                schedule_db=schedule,
                user=user,
            )

        for _ in range(2):
            assert send("agent:19:example").message == "Hello."
        with pytest.raises(HTTPException) as exc:
            send("agent:20:example")
        assert exc.value.status_code == 403
