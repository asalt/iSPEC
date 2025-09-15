import logging
import pytest
from fastapi.testclient import TestClient

from ispec.api.main import app
from ispec.db.connect import get_session as db_get_session
from ispec.db.models import Person, Project, ProjectComment, logger as db_logger


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provide a TestClient with an isolated temporary database."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))

    db_logger.setLevel(logging.ERROR)

    def override_get_session():
        with db_get_session() as session:
            yield session

    app.dependency_overrides[db_get_session] = override_get_session
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def test_status_endpoint(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_person_crud(client):
    payload = {
        "ppl_AddedBy": "tester",
        "ppl_Name_First": "John",
        "ppl_Name_Last": "Doe",
    }
    resp = client.post("/people/", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["ppl_Name_First"] == "John"

    with db_get_session() as session:
        person_id = session.query(Person.id).scalar()

    resp = client.get(f"/people/{person_id}")
    assert resp.status_code == 200
    assert resp.json()["ppl_Name_Last"] == "Doe"

    update_payload = {
        "ppl_AddedBy": "tester",
        "ppl_Name_First": "Jane",
        "ppl_Name_Last": "Doe",
    }
    resp = client.put(f"/people/{person_id}", json=update_payload)
    assert resp.status_code == 200
    assert resp.json()["ppl_Name_First"] == "Jane"

    resp = client.delete(f"/people/{person_id}")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted", "id": person_id}


def test_project_crud(client):
    payload = {
        "prj_AddedBy": "tester",
        "prj_ProjectTitle": "Test Project",
        "prj_ProjectBackground": "Background",
    }
    resp = client.post("/projects/", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["prj_ProjectTitle"] == "Test Project"

    with db_get_session() as session:
        project_id = session.query(Project.id).scalar()

    resp = client.get(f"/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["prj_ProjectBackground"] == "Background"

    update_payload = {
        "prj_AddedBy": "tester",
        "prj_ProjectTitle": "Updated Title",
        "prj_ProjectBackground": "Background",
    }
    resp = client.put(f"/projects/{project_id}", json=update_payload)
    assert resp.status_code == 200
    assert resp.json()["prj_ProjectTitle"] == "Updated Title"

    resp = client.delete(f"/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted", "id": project_id}


def test_project_comment_crud(client):
    person_payload = {
        "ppl_AddedBy": "tester",
        "ppl_Name_First": "John",
        "ppl_Name_Last": "Doe",
    }
    person_resp = client.post("/people/", json=person_payload)
    with db_get_session() as session:
        person_id = session.query(Person.id).scalar()

    project_payload = {
        "prj_AddedBy": "tester",
        "prj_ProjectTitle": "Test Project",
        "prj_ProjectBackground": "Background",
    }
    project_resp = client.post("/projects/", json=project_payload)
    with db_get_session() as session:
        project_id = session.query(Project.id).scalar()

    payload = {
        "project_id": project_id,
        "person_id": person_id,
        "com_Comment": "Initial comment",
    }
    resp = client.post("/project_comment/", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["com_Comment"] == "Initial comment"

    with db_get_session() as session:
        comment_id = session.query(ProjectComment.id).scalar()

    resp = client.get(f"/project_comment/{comment_id}")
    assert resp.status_code == 200
    assert resp.json()["person_id"] == person_id

    update_payload = {
        "project_id": project_id,
        "person_id": person_id,
        "com_Comment": "Updated comment",
    }
    resp = client.put(f"/project_comment/{comment_id}", json=update_payload)
    assert resp.status_code == 200
    assert resp.json()["com_Comment"] == "Updated comment"

    resp = client.delete(f"/project_comment/{comment_id}")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted", "id": comment_id}
