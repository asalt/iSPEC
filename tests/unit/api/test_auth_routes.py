from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.auth import router as auth_router
from ispec.api.security import create_session, session_cookie_name
from ispec.db.connect import (
    get_session_dep,
    initialize_db,
    make_session_factory,
    sqlite_engine,
)
from ispec.db.models import AuthUser, AuthUserProject, Project, UserRole

pytestmark = pytest.mark.testclient


@pytest.fixture
def auth_client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/auth.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    session_factory = make_session_factory(engine)

    app = FastAPI()
    app.include_router(auth_router)

    def override_session():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_session

    with TestClient(app) as client:
        client.session_factory = session_factory  # type: ignore[attr-defined]
        yield client


def _create_user(
    session_factory,
    *,
    username: str,
    role: UserRole,
    assistant_brief: str | None = None,
) -> AuthUser:
    with session_factory() as db:
        user = AuthUser(
            username=username,
            password_hash="hash",
            password_salt="salt",
            password_iterations=1,
            role=role,
            is_active=True,
            assistant_brief=assistant_brief,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        db.expunge(user)
        return user


def _admin_cookie(session_factory, *, admin_user_id: int) -> dict[str, str]:
    with session_factory() as db:
        admin = db.get(AuthUser, admin_user_id)
        assert admin is not None
        token = create_session(db, user=admin)
    return {session_cookie_name(): token}


def test_auth_me_returns_assistant_brief(auth_client):
    user = _create_user(
        auth_client.session_factory,  # type: ignore[attr-defined]
        username="alex",
        role=UserRole.admin,
        assistant_brief="Developer using tmux and repo tools.",
    )
    cookies = _admin_cookie(auth_client.session_factory, admin_user_id=int(user.id))  # type: ignore[attr-defined]

    resp = auth_client.get("/auth/me", cookies=cookies)
    assert resp.status_code == 200
    assert resp.json()["assistant_brief"] == "Developer using tmux and repo tools."


def test_auth_staff_can_update_and_clear_user_assistant_brief(auth_client):
    admin = _create_user(auth_client.session_factory, username="admin", role=UserRole.admin)  # type: ignore[attr-defined]
    target = _create_user(auth_client.session_factory, username="alex", role=UserRole.editor)  # type: ignore[attr-defined]
    cookies = _admin_cookie(auth_client.session_factory, admin_user_id=int(admin.id))  # type: ignore[attr-defined]

    resp = auth_client.put(
        f"/auth/users/{int(target.id)}/assistant-brief",
        json={"assistant_brief": "Developer focused on local devops workflows."},
        cookies=cookies,
    )
    assert resp.status_code == 200
    assert (
        resp.json()["assistant_brief"] == "Developer focused on local devops workflows."
    )

    with auth_client.session_factory() as db:  # type: ignore[attr-defined]
        refreshed = db.get(AuthUser, int(target.id))
        assert refreshed is not None
        assert (
            refreshed.assistant_brief == "Developer focused on local devops workflows."
        )

    cleared = auth_client.put(
        f"/auth/users/{int(target.id)}/assistant-brief",
        json={"assistant_brief": "   "},
        cookies=cookies,
    )
    assert cleared.status_code == 200
    assert cleared.json()["assistant_brief"] is None

    with auth_client.session_factory() as db:  # type: ignore[attr-defined]
        refreshed = db.get(AuthUser, int(target.id))
        assert refreshed is not None
        assert refreshed.assistant_brief is None


def test_auth_admin_can_create_user_but_editor_cannot(auth_client):
    admin = _create_user(auth_client.session_factory, username="admin", role=UserRole.admin)  # type: ignore[attr-defined]
    editor = _create_user(auth_client.session_factory, username="editor", role=UserRole.editor)  # type: ignore[attr-defined]
    admin_cookies = _admin_cookie(auth_client.session_factory, admin_user_id=int(admin.id))  # type: ignore[attr-defined]
    editor_cookies = _admin_cookie(auth_client.session_factory, admin_user_id=int(editor.id))  # type: ignore[attr-defined]

    editor_resp = auth_client.post(
        "/auth/users",
        json={
            "username": "blocked-client",
            "password": "temporary-password",
            "role": "client",
            "must_change_password": True,
        },
        cookies=editor_cookies,
    )
    assert editor_resp.status_code == 403

    admin_resp = auth_client.post(
        "/auth/users",
        json={
            "username": "demo-client",
            "password": "temporary-password",
            "role": "client",
            "must_change_password": True,
        },
        cookies=admin_cookies,
    )
    assert admin_resp.status_code == 201
    body = admin_resp.json()
    assert body["username"] == "demo-client"
    assert body["role"] == "client"
    assert body["project_access_mode"] == "explicit_projects"
    assert body["must_change_password"] is True
    assert body["project_count"] == 0
    assert body["effective_project_access"] == "none"


def test_auth_users_list_reports_project_access_summary(auth_client):
    admin = _create_user(auth_client.session_factory, username="admin", role=UserRole.admin)  # type: ignore[attr-defined]
    client_user = _create_user(auth_client.session_factory, username="demo", role=UserRole.client)  # type: ignore[attr-defined]
    viewer = _create_user(auth_client.session_factory, username="viewer", role=UserRole.viewer)  # type: ignore[attr-defined]
    cookies = _admin_cookie(auth_client.session_factory, admin_user_id=int(admin.id))  # type: ignore[attr-defined]

    with auth_client.session_factory() as db:  # type: ignore[attr-defined]
        project = Project(prj_AddedBy="tester", prj_ProjectTitle="Allowed Project")
        db.add(project)
        db.flush()
        db.add(AuthUserProject(user_id=int(client_user.id), project_id=int(project.id)))
        db.commit()

    resp = auth_client.get("/auth/users", cookies=cookies)
    assert resp.status_code == 200
    by_username = {row["username"]: row for row in resp.json()}

    assert by_username["demo"]["project_count"] == 1
    assert by_username["demo"]["project_access_mode"] == "explicit_projects"
    assert by_username["demo"]["effective_project_access"] == "restricted"
    assert by_username["viewer"]["project_count"] == 0
    assert by_username["viewer"]["project_access_mode"] == "explicit_projects"
    assert by_username["viewer"]["effective_project_access"] == "none"


def test_auth_staff_can_replace_user_project_grants(auth_client):
    editor = _create_user(auth_client.session_factory, username="editor", role=UserRole.editor)  # type: ignore[attr-defined]
    client_user = _create_user(auth_client.session_factory, username="demo", role=UserRole.client)  # type: ignore[attr-defined]
    cookies = _admin_cookie(auth_client.session_factory, admin_user_id=int(editor.id))  # type: ignore[attr-defined]

    with auth_client.session_factory() as db:  # type: ignore[attr-defined]
        project_1 = Project(prj_AddedBy="tester", prj_ProjectTitle="Allowed One")
        project_2 = Project(prj_AddedBy="tester", prj_ProjectTitle="Allowed Two")
        db.add(project_1)
        db.add(project_2)
        db.flush()
        project_1_id = int(project_1.id)
        project_2_id = int(project_2.id)
        db.add(AuthUserProject(user_id=int(client_user.id), project_id=project_1_id))
        db.commit()

    resp = auth_client.put(
        f"/auth/users/{int(client_user.id)}/projects",
        json={"project_ids": [project_2_id, project_1_id, project_1_id]},
        cookies=cookies,
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "user_id": int(client_user.id),
        "project_ids": [project_2_id, project_1_id],
    }

    listed = auth_client.get(
        f"/auth/users/{int(client_user.id)}/projects",
        cookies=cookies,
    )
    assert listed.status_code == 200
    assert listed.json() == {
        "user_id": int(client_user.id),
        "project_ids": [project_1_id, project_2_id],
    }

    with auth_client.session_factory() as db:  # type: ignore[attr-defined]
        rows = (
            db.query(AuthUserProject)
            .filter(AuthUserProject.user_id == int(client_user.id))
            .order_by(AuthUserProject.project_id.asc())
            .all()
        )
        assert [int(row.granted_by_user_id) for row in rows] == [
            int(editor.id),
            int(editor.id),
        ]


def test_auth_project_grants_reject_unknown_project_without_mutating(auth_client):
    editor = _create_user(auth_client.session_factory, username="editor", role=UserRole.editor)  # type: ignore[attr-defined]
    client_user = _create_user(auth_client.session_factory, username="demo", role=UserRole.client)  # type: ignore[attr-defined]
    cookies = _admin_cookie(auth_client.session_factory, admin_user_id=int(editor.id))  # type: ignore[attr-defined]

    with auth_client.session_factory() as db:  # type: ignore[attr-defined]
        project = Project(prj_AddedBy="tester", prj_ProjectTitle="Allowed")
        db.add(project)
        db.flush()
        project_id = int(project.id)
        db.add(AuthUserProject(user_id=int(client_user.id), project_id=project_id))
        db.commit()

    rejected = auth_client.put(
        f"/auth/users/{int(client_user.id)}/projects",
        json={"project_ids": [project_id, 999999]},
        cookies=cookies,
    )
    assert rejected.status_code == 404

    listed = auth_client.get(
        f"/auth/users/{int(client_user.id)}/projects",
        cookies=cookies,
    )
    assert listed.status_code == 200
    assert listed.json() == {
        "user_id": int(client_user.id),
        "project_ids": [project_id],
    }
