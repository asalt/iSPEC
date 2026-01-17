import pytest
from starlette.requests import Request
from fastapi import HTTPException

from ispec.api.security import (
    create_session,
    delete_session,
    get_current_user,
    hash_password,
    require_user,
    require_access,
    require_assistant_access,
    session_cookie_name,
    verify_password,
)
from ispec.db.models import AuthSession, AuthUser, UserRole


def _make_request(*, method: str, path: str = "/", cookie: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if cookie:
        headers.append((b"cookie", cookie.encode("latin-1")))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "query_string": b"",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_password_hash_round_trip(monkeypatch):
    monkeypatch.setenv("ISPEC_PASSWORD_PEPPER", "dev-pepper")
    salt_b64, hash_b64, iterations = hash_password("supersecret")
    assert verify_password(
        "supersecret",
        salt_b64=salt_b64,
        hash_b64=hash_b64,
        iterations=iterations,
    )
    assert not verify_password(
        "wrong",
        salt_b64=salt_b64,
        hash_b64=hash_b64,
        iterations=iterations,
    )


def test_session_round_trip_and_delete(db_session):
    user = AuthUser(
        username="alice",
        password_hash="x",
        password_salt="y",
        password_iterations=1,
        role=UserRole.admin,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_session(db_session, user=user)
    assert isinstance(token, str) and token

    stored = db_session.query(AuthSession).one()
    assert stored.token_hash != token

    cookie = f"{session_cookie_name()}={token}"
    request = _make_request(method="GET", cookie=cookie)
    current = get_current_user(request, db_session)
    assert current is not None
    assert current.username == "alice"

    delete_session(db_session, token=token)
    assert db_session.query(AuthSession).count() == 0


def test_require_access_enforces_viewer_is_read_only(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "1")

    user = AuthUser(
        username="viewer",
        password_hash="x",
        password_salt="y",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_session(db_session, user=user)
    cookie = f"{session_cookie_name()}={token}"
    request = _make_request(method="POST", cookie=cookie)

    with pytest.raises(HTTPException) as exc:
        require_access(request, db_session, provided_api_key=None)
    assert exc.value.status_code == 403


def test_require_user_enforces_password_change_required():
    user = AuthUser(
        username="editor",
        password_hash="x",
        password_salt="y",
        password_iterations=1,
        role=UserRole.editor,
        is_active=True,
        must_change_password=True,
    )

    assert require_user(_make_request(method="GET", path="/api/auth/me"), user=user) is user
    assert (
        require_user(_make_request(method="POST", path="/api/auth/change-password"), user=user) is user
    )

    with pytest.raises(HTTPException) as exc:
        require_user(_make_request(method="GET", path="/api/projects"), user=user)
    assert exc.value.status_code == 403


def test_require_access_blocks_when_password_change_required(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "1")

    user = AuthUser(
        username="editor",
        password_hash="x",
        password_salt="y",
        password_iterations=1,
        role=UserRole.editor,
        is_active=True,
        must_change_password=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_session(db_session, user=user)
    cookie = f"{session_cookie_name()}={token}"
    request = _make_request(method="GET", path="/api/projects", cookie=cookie)

    with pytest.raises(HTTPException) as exc:
        require_access(request, db_session, provided_api_key=None)
    assert exc.value.status_code == 403


def test_require_assistant_access_can_allow_api_key_only(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "1")
    monkeypatch.setenv("ISPEC_API_KEY", "secret")
    monkeypatch.setenv("ISPEC_ASSISTANT_ALLOW_API_KEY_ONLY", "1")

    request = _make_request(method="POST", path="/api/support/chat")
    user = require_assistant_access(request, db_session, provided_api_key="secret")
    assert user is not None
    assert getattr(user, "role", None) == UserRole.viewer
    assert bool(getattr(user, "can_write_project_comments", False)) is False


def test_require_assistant_access_can_allow_api_key_project_comment_writes(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "1")
    monkeypatch.setenv("ISPEC_API_KEY", "secret")
    monkeypatch.setenv("ISPEC_ASSISTANT_ALLOW_API_KEY_ONLY", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_ALLOW_API_KEY_WRITE_PROJECT_COMMENTS", "1")

    request = _make_request(method="POST", path="/api/support/chat")
    user = require_assistant_access(request, db_session, provided_api_key="secret")
    assert user is not None
    assert getattr(user, "role", None) == UserRole.viewer
    assert bool(getattr(user, "can_write_project_comments", False)) is True


def test_require_assistant_access_prefers_cookie_user_over_api_key_only(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "1")
    monkeypatch.setenv("ISPEC_API_KEY", "secret")
    monkeypatch.setenv("ISPEC_ASSISTANT_ALLOW_API_KEY_ONLY", "1")

    user = AuthUser(
        username="editor",
        password_hash="x",
        password_salt="y",
        password_iterations=1,
        role=UserRole.editor,
        is_active=True,
        must_change_password=False,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_session(db_session, user=user)
    cookie = f"{session_cookie_name()}={token}"
    request = _make_request(method="POST", path="/api/support/chat", cookie=cookie)

    resolved = require_assistant_access(request, db_session, provided_api_key="secret")
    assert resolved is not None
    assert getattr(resolved, "id", None) == user.id


def test_require_assistant_access_requires_login_when_disabled(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "1")
    monkeypatch.setenv("ISPEC_API_KEY", "secret")
    monkeypatch.delenv("ISPEC_ASSISTANT_ALLOW_API_KEY_ONLY", raising=False)

    request = _make_request(method="POST", path="/api/support/chat")
    with pytest.raises(HTTPException) as exc:
        require_assistant_access(request, db_session, provided_api_key="secret")
    assert exc.value.status_code == 401
