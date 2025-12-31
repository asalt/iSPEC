import logging
import pytest
from fastapi.testclient import TestClient

from ispec.api.main import app
from ispec.db.connect import get_session as db_get_session, get_session_dep
from ispec.db.models import Person, Project, ProjectComment, logger as db_logger


def _create_person(client: TestClient, *, first: str, last: str, added_by: str = "tester") -> int:
    payload = {
        "ppl_AddedBy": added_by,
        "ppl_Name_First": first,
        "ppl_Name_Last": last,
    }
    resp = client.post("/people/", json=payload)
    assert resp.status_code == 201, resp.text

    with db_get_session() as session:
        person = (
            session.query(Person)
            .filter(
                Person.ppl_Name_First == first,
                Person.ppl_Name_Last == last,
            )
            .one()
        )
        return person.id


def _create_project(
    client: TestClient,
    *,
    title: str,
    background: str,
    added_by: str = "tester",
) -> int:
    payload = {
        "prj_AddedBy": added_by,
        "prj_ProjectTitle": title,
        "prj_ProjectBackground": background,
    }
    resp = client.post("/projects/", json=payload)
    assert resp.status_code == 201, resp.text

    with db_get_session() as session:
        project = (
            session.query(Project)
            .filter(Project.prj_ProjectTitle == title)
            .one()
        )
        return project.id


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provide a TestClient with an isolated temporary database."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))

    db_logger.setLevel(logging.ERROR)

    def override_get_session():
        with db_get_session() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session
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


def test_person_duplicate_conflict(client):
    first_name = "John"
    last_name = "Doe"
    _create_person(client, first=first_name, last=last_name)

    duplicate_payload = {
        "ppl_AddedBy": "tester",
        "ppl_Name_First": first_name.lower(),
        "ppl_Name_Last": last_name.upper(),
    }
    resp = client.post("/people/", json=duplicate_payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "Person already exists"}


def test_person_get_after_delete_returns_404(client):
    person_id = _create_person(client, first="Alice", last="Smith")

    resp = client.delete(f"/people/{person_id}")
    assert resp.status_code == 200

    resp = client.get(f"/people/{person_id}")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Person not found"}


def test_person_options_supports_filtering(client):
    smith_alice = _create_person(client, first="Alice", last="Smith")
    jones_bob = _create_person(client, first="Bob", last="Jones")
    smith_carol = _create_person(client, first="Carol", last="Smith")

    resp = client.get("/people/options", params={"q": "smith"})
    assert resp.status_code == 200
    payload = resp.json()
    assert {item["value"] for item in payload} == {smith_alice, smith_carol}
    assert all("Smith" in item["label"] for item in payload)

    resp = client.get("/people/options", params={"q": "smith", "limit": 1})
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = client.get(
        "/people/options",
        params=[("ids", smith_alice), ("ids", jones_bob)],
    )
    assert resp.status_code == 200
    assert {item["value"] for item in resp.json()} == {smith_alice, jones_bob}

    resp = client.get(
        "/people/options",
        params=[("exclude_ids", jones_bob)],
    )
    assert resp.status_code == 200
    assert all(item["value"] != jones_bob for item in resp.json())


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


def test_project_duplicate_conflict(client):
    title = "Unique Title"
    background = "Background"
    _create_project(client, title=title, background=background)

    duplicate_payload = {
        "prj_AddedBy": "tester",
        "prj_ProjectTitle": title.lower(),
        "prj_ProjectBackground": background,
    }

    resp = client.post("/projects/", json=duplicate_payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "Project already exists"}


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


def test_project_comment_schema_includes_async_select_metadata(client):
    resp = client.get("/project_comment/schema")
    assert resp.status_code == 200

    schema = resp.json()
    properties = schema["properties"]

    project_ui = properties["project_id"]["ui"]
    assert project_ui["component"] == "SelectAsync"
    assert project_ui["optionsEndpoint"] == "/projects/options"

    person_ui = properties["person_id"]["ui"]
    assert person_ui["component"] == "SelectAsync"
    assert person_ui["optionsEndpoint"] == "/people/options"


def test_project_comment_options_for_relationships(client):
    person_id = _create_person(client, first="Eve", last="Stone")
    project_id = _create_project(client, title="Options Project", background="Bg")

    resp = client.get("/project_comment/options/person", params={"q": "Stone"})
    assert resp.status_code == 200
    payload = resp.json()
    assert any(item["value"] == person_id for item in payload)

    resp = client.get("/project_comment/options/project", params={"ids": project_id})
    assert resp.status_code == 200
    assert any(item["value"] == project_id for item in resp.json())


def test_project_comment_options_invalid_relationship_returns_404(client):
    resp = client.get("/project_comment/options/unknown")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "No relationship named 'unknown'"}
