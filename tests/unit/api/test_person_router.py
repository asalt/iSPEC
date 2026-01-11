import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import generate_crud_router
from ispec.db.crud import PersonCRUD
from ispec.db.models import Person
from ispec.db.connect import get_session_dep, make_session_factory, sqlite_engine, initialize_db

pytestmark = pytest.mark.testclient


@pytest.fixture
def client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    test_session = make_session_factory(engine)

    app = FastAPI()
    route_prefix_map: dict[str, str] = {}
    person_router = generate_crud_router(
        model=Person,
        crud_class=PersonCRUD,
        prefix="/people",
        tag="Person",
        exclude_fields=set(),
        create_exclude_fields={"id", "ppl_CreationTS", "ppl_ModificationTS"},
        route_prefix_by_table=route_prefix_map,
    )
    app.include_router(person_router)

    def override_get_session():
        with test_session() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session

    with TestClient(app) as client:
        client.session_factory = test_session  # type: ignore[attr-defined]
        client.route_prefix_map = route_prefix_map  # type: ignore[attr-defined]
        yield client


def test_person_router_crud_and_route_prefix(client):
    # Route prefix mapping should register the person table for FK resolution
    assert client.route_prefix_map["person"] == "/people"  # type: ignore[attr-defined]

    # schema endpoint should expose model schema
    resp = client.get("/people/schema")
    assert resp.status_code == 200
    schema = resp.json()
    assert "ppl_Name_First" in schema["properties"]

    # list should be empty initially
    resp = client.get("/people/")
    assert resp.status_code == 200
    assert resp.json() == []

    payload = {
        "ppl_Name_First": "Jane",
        "ppl_Name_Last": "Doe",
        "ppl_AddedBy": "tester",
    }

    # create person
    resp = client.post("/people/", json=payload)
    assert resp.status_code == 201

    with client.session_factory() as db:  # type: ignore[attr-defined]
        person_id = db.query(Person).filter_by(ppl_Name_First="Jane", ppl_Name_Last="Doe").first().id

    # retrieve person
    resp = client.get(f"/people/{person_id}")
    assert resp.status_code == 200
    assert resp.json()["ppl_Name_First"] == payload["ppl_Name_First"]

    # list should now contain one entry
    resp = client.get("/people/")
    assert resp.status_code == 200
    assert any(row["id"] == person_id for row in resp.json())

    # delete person
    resp = client.delete(f"/people/{person_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # ensure person is gone
    resp = client.get(f"/people/{person_id}")
    assert resp.status_code == 404
