import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import generate_crud_router
from ispec.db.crud import ProjectPersonCRUD
from ispec.db.models import Person, Project, ProjectPerson
from ispec.db.connect import get_session_dep, make_session_factory, sqlite_engine, initialize_db

pytestmark = pytest.mark.testclient


@pytest.fixture
def client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/pp.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    test_session = make_session_factory(engine)

    app = FastAPI()
    route_prefix_map: dict[str, str] = {}
    router = generate_crud_router(
        model=ProjectPerson,
        crud_class=ProjectPersonCRUD,
        prefix="/project_person",
        tag="ProjectPerson",
        exclude_fields=set(),
        create_exclude_fields={"id", "projper_CreationTS", "projper_ModificationTS"},
        route_prefix_by_table=route_prefix_map,
    )
    app.include_router(router)

    def override_get_session():
        with test_session() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session

    with TestClient(app) as client:
        client.session_factory = test_session  # type: ignore[attr-defined]
        client.route_prefix_map = route_prefix_map  # type: ignore[attr-defined]
        yield client


def test_project_person_crud(client):
    # seed person and project
    with client.session_factory() as db:  # type: ignore[attr-defined]
        person = Person(ppl_AddedBy="tester", ppl_Name_First="A", ppl_Name_Last="B")
        project = Project(prj_AddedBy="tester", prj_ProjectTitle="X")
        db.add_all([person, project])
        db.commit()
        db.refresh(person)
        db.refresh(project)
        person_id, project_id = person.id, project.id

    payload = {"person_id": person_id, "project_id": project_id}
    resp = client.post("/project_person/", json=payload)
    assert resp.status_code == 201
    link_id = resp.json()["id"]

    resp = client.get(f"/project_person/{link_id}")
    assert resp.status_code == 200
    assert resp.json()["person_id"] == person_id
    assert resp.json()["project_id"] == project_id

    resp = client.delete(f"/project_person/{link_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    resp = client.get(f"/project_person/{link_id}")
    assert resp.status_code == 404
