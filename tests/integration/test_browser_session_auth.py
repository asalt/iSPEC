"""Real application routing and cookie login, isolated DBs and no remote IO."""

from datetime import UTC, datetime, timedelta
import importlib

from fastapi.testclient import TestClient
import pytest

from ispec.api.security import hash_password, session_cookie_name
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.db.connect import get_session_dep, initialize_db, make_session_factory, sqlite_engine
from ispec.db.models import AuthSession, AuthUser, AuthUserProject, Project, UserRole

pytestmark = pytest.mark.testclient
SERVICE_KEY = "isolated-test-service-key"
PASSWORD = "isolated-user-password"
ORIGIN = "https://ispec.example.test"


@pytest.fixture
def browser(tmp_path, monkeypatch):
    for key, value in {
        "ISPEC_DB_DIR": str(tmp_path),
        "ISPEC_DB_PATH": str(tmp_path / "core.db"),
        "ISPEC_ANALYSIS_DB_PATH": str(tmp_path / "analysis.db"),
        "ISPEC_PSM_DB_PATH": str(tmp_path / "psm.db"),
        "ISPEC_ASSISTANT_DB_PATH": str(tmp_path / "assistant.db"),
        "ISPEC_PROMPTS_DB_PATH": str(tmp_path / "prompts.db"),
        "ISPEC_AGENT_DB_PATH": str(tmp_path / "agent.db"),
        "ISPEC_AGENT_STATE_DB_PATH": str(tmp_path / "agent-state.db"),
        "ISPEC_SCHEDULE_DB_PATH": str(tmp_path / "schedule.db"),
        "ISPEC_LOG_DIR": str(tmp_path / "logs"),
        "ISPEC_REQUIRE_LOGIN": "1", "ISPEC_API_KEY": SERVICE_KEY,
        "ISPEC_API_REQUIRE_KEY": "1", "ISPEC_DEV_DEFAULT_ADMIN": "0",
        "ISPEC_SESSION_COOKIE_SECURE": "1", "ISPEC_SESSION_COOKIE_SAMESITE": "lax",
        "ISPEC_PASSWORD_ITERATIONS": "50000", "ISPEC_PASSWORD_PEPPER": "test-only",
        "ISPEC_ASSISTANT_PROVIDER": "stub", "ISPEC_ASSISTANT_EXECUTION_MODE": "inline",
        "ISPEC_ASSISTANT_ALLOW_API_KEY_ONLY": "1",
        "ISPEC_ASSISTANT_ALLOW_API_KEY_WRITE_PROJECT_COMMENTS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    # Any accidental model/Slack/legacy request is a test failure.
    def no_network(*args, **kwargs):
        raise AssertionError("No external requests in browser auth tests")
    monkeypatch.setattr("requests.sessions.Session.request", no_network)
    engine = sqlite_engine(f"sqlite:///{tmp_path}/core.db")
    initialize_db(engine)
    factory = make_session_factory(engine)
    salt, hashed, iterations = hash_password(PASSWORD)
    with factory() as db:
        db.add_all([
            Project(id=101, prj_AddedBy="test", prj_ProjectTitle="Allowed"),
            Project(id=102, prj_AddedBy="test", prj_ProjectTitle="Other"),
        ])
        for username, role in [("client", UserRole.client), ("staff", UserRole.editor), ("admin", UserRole.admin)]:
            db.add(AuthUser(username=username, role=role, password_hash=hashed,
                            password_salt=salt, password_iterations=iterations, is_active=True))
        db.flush()
        client_id = db.query(AuthUser).filter_by(username="client").one().id
        db.add(AuthUserProject(user_id=client_id, project_id=101))
        db.commit()
    main = importlib.import_module("ispec.api.main")
    def session():
        with factory() as db:
            yield db
    with monkeypatch.context() as patches:
        patches.setitem(main.app.dependency_overrides, get_session_dep, session)
        with TestClient(main.app, base_url=ORIGIN) as client:
            client.session_factory = factory
            yield client
    engine.dispose()


def login(browser, username="client"):
    response = browser.post("/api/auth/login", json={"username": username, "password": PASSWORD},
                            headers={"Origin": ORIGIN})
    assert response.status_code == 200, response.text
    return response


def test_login_me_logout_without_shared_key(browser):
    assert browser.get("/api/auth/me").status_code == 401
    response = login(browser)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie and "samesite=lax" in cookie
    assert browser.get("/api/auth/me").json()["username"] == "client"
    assert browser.post("/api/auth/logout", headers={"Origin": ORIGIN}).status_code == 200
    assert browser.get("/api/projects").status_code == 401


def test_wrong_password_remains_rejected(browser):
    response = browser.post("/api/auth/login", json={"username": "client", "password": "wrong"})
    assert response.status_code == 401
    assert session_cookie_name() not in browser.cookies


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": SERVICE_KEY}, {"Authorization": f"Bearer {SERVICE_KEY}"}])
def test_client_cookie_retains_project_scope_even_with_service_key(browser, headers):
    login(browser)
    assert [row["id"] for row in browser.get("/api/projects", headers=headers).json()] == [101]
    assert browser.get("/api/projects/101", headers=headers).status_code == 200
    assert browser.get("/api/projects/102", headers=headers).status_code == 404
    assert browser.put("/api/projects/101", headers=headers, json={
        "prj_ProjectTitle": "Not allowed", "prj_AddedBy": "test",
    }).status_code == 403
    assert browser.get("/api/auth/users", headers=headers).status_code == 403


@pytest.mark.parametrize("username", ["staff", "admin"])
def test_staff_cookie_can_edit_without_key(browser, username):
    login(browser, username)
    assert len(browser.get("/api/projects").json()) == 2
    response = browser.put("/api/projects/101", headers={"Origin": ORIGIN}, json={
        "prj_ProjectTitle": "Updated", "prj_AddedBy": "test",
    })
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("state", ["expired", "disabled", "password_change"])
def test_unusable_cookie_never_falls_back_to_service_authority(browser, state):
    login(browser)
    with browser.session_factory() as db:
        user = db.query(AuthUser).filter_by(username="client").one()
        if state == "expired":
            db.query(AuthSession).filter_by(user_id=user.id).one().expires_at = datetime.now(UTC) - timedelta(days=1)
        elif state == "disabled":
            user.is_active = False
        else:
            user.must_change_password = True
        db.commit()
    expected = 403 if state == "password_change" else 401
    assert browser.get("/api/projects", headers={"X-API-Key": SERVICE_KEY}).status_code == expected
    assert browser.post("/api/support/chat", headers={"X-API-Key": SERVICE_KEY},
                        json={"message": "hello"}).status_code == expected


def test_internal_receipt_route_still_requires_service_key(browser):
    login(browser, "admin")
    scope = {"team_id": "TTEST", "channel": "DTEST", "message_ts": "1790000000.000001"}
    for headers in [{}, {"X-API-Key": "wrong"}]:
        assert browser.post("/api/agents/slack/outbound-context", json=scope, headers=headers).status_code == 401
    assert browser.post("/api/agents/slack/outbound-context", json=scope,
                        headers={"X-API-Key": SERVICE_KEY}).status_code == 200


def test_anonymous_and_key_only_requests_cannot_access_cookie_only_crud(browser):
    for headers in [{}, {"X-API-Key": SERVICE_KEY}]:
        assert browser.get("/api/projects", headers=headers).status_code == 401
    assert browser.post("/api/support/chat", json={"message": "hello"}).status_code == 401


def test_assistant_supports_cookie_and_explicit_service_callers(browser):
    login(browser)
    response = browser.post("/api/support/chat", json={"message": "Hello", "sessionId": "browser-test"})
    assert response.status_code == 200, response.text
    with get_assistant_session() as db:
        session = db.query(SupportSession).filter_by(session_id="browser-test").one()
        assert session.user_id is not None and session.user_id > 0
    browser.cookies.clear()
    response = browser.post("/api/support/chat", json={"message": "Hello", "sessionId": "service-test"},
                            headers={"X-API-Key": SERVICE_KEY})
    assert response.status_code == 200, response.text


def test_forced_password_change_works_without_service_key(browser):
    with browser.session_factory() as db:
        db.query(AuthUser).filter_by(username="client").one().must_change_password = True
        db.commit()
    login(browser)
    original_cookie = browser.cookies.get(session_cookie_name())
    assert browser.get("/api/auth/me").json()["must_change_password"] is True
    assert browser.get("/api/projects").status_code == 403
    response = browser.post("/api/auth/change-password", headers={"Origin": ORIGIN}, json={
        "current_password": PASSWORD, "new_password": "changed-test-password",
    })
    assert response.status_code == 200, response.text
    assert response.json()["must_change_password"] is False
    assert browser.cookies.get(session_cookie_name()) != original_cookie
    assert [row["id"] for row in browser.get("/api/projects").json()] == [101]
    browser.cookies.clear()
    assert browser.get("/api/projects", headers={
        "Cookie": f"{session_cookie_name()}={original_cookie}",
    }).status_code == 401


def test_login_optional_does_not_discard_an_authenticated_clients_scope(browser, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "0")
    login(browser)
    assert [row["id"] for row in browser.get("/api/projects").json()] == [101]
    assert browser.put("/api/projects/101", json={
        "prj_ProjectTitle": "Not allowed", "prj_AddedBy": "test",
    }).status_code == 403


def test_sessions_do_not_depend_on_internal_service_key_configuration(browser, monkeypatch):
    monkeypatch.delenv("ISPEC_API_KEY")
    login(browser)
    assert [row["id"] for row in browser.get("/api/projects").json()] == [101]
    assert browser.post("/api/agents/slack/outbound-context", json={
        "team_id": "TTEST", "channel": "DTEST", "message_ts": "1790000000.000001",
    }).status_code == 503


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example.test"}, {"Origin": "null"},
    {"Sec-Fetch-Site": "cross-site"}, {"Origin": ORIGIN + "/misleading"},
])
def test_cross_origin_login_and_cookie_mutations_rejected(browser, headers):
    assert browser.post("/api/auth/login", headers=headers, json={
        "username": "staff", "password": PASSWORD,
    }).status_code == 403
    login(browser, "staff")
    assert browser.post("/api/auth/logout", headers=headers).status_code == 403
    assert browser.put("/api/projects/101", headers=headers, json={
        "prj_ProjectTitle": "Wrong origin", "prj_AddedBy": "test",
    }).status_code == 403


def test_bootstrap_is_not_public_even_when_login_is_public(browser, monkeypatch):
    payload = {"username": "new-admin", "password": PASSWORD}
    assert browser.post("/api/auth/bootstrap", json=payload).status_code == 401
    assert browser.post("/api/auth/bootstrap", json=payload,
                        headers={"X-API-Key": SERVICE_KEY}).status_code == 409
    monkeypatch.delenv("ISPEC_API_KEY")
    assert browser.post("/api/auth/bootstrap", json=payload).status_code == 503
