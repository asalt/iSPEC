from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.auth import router as auth_router
from ispec.api.security import create_session, session_cookie_name
from ispec.db.connect import get_session_dep, initialize_db, make_session_factory, sqlite_engine
from ispec.db.models import AuthUser, UserRole

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


def _create_user(session_factory, *, username: str, role: UserRole, assistant_brief: str | None = None) -> AuthUser:
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
    assert resp.json()["assistant_brief"] == "Developer focused on local devops workflows."

    with auth_client.session_factory() as db:  # type: ignore[attr-defined]
        refreshed = db.get(AuthUser, int(target.id))
        assert refreshed is not None
        assert refreshed.assistant_brief == "Developer focused on local devops workflows."

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
