import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import (
    _add_schema_endpoint,
    _add_crud_endpoints,
    _add_options_endpoints,
)
from ispec.api.models.modelmaker import make_pydantic_model_from_sqlalchemy
from ispec.db.models import Person
from ispec.db.crud import PersonCRUD
from ispec.db.connect import get_session_dep, make_session_factory, sqlite_engine, initialize_db


def test_add_schema_endpoint_exposes_schema():
    router = APIRouter(prefix="/people", tags=["Person"])
    PersonCreate = make_pydantic_model_from_sqlalchemy(Person, name_suffix="Create")

    mapping = {"person": "/people"}

    def prefix_for_table(name: str) -> str:
        return mapping.get(name, f"/{name}")

    _add_schema_endpoint(router, Person, PersonCreate, route_prefix_for_table=prefix_for_table)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/people/schema")
    assert resp.status_code == 200
    assert "ppl_Name_First" in resp.json()["properties"]


@pytest.fixture
def crud_client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/crud.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    session_factory = make_session_factory(engine)

    router = APIRouter(prefix="/people", tags=["Person"])
    crud = PersonCRUD()
    ReadModel = make_pydantic_model_from_sqlalchemy(
        Person, name_suffix="Read", exclude_fields=set()
    )
    CreateModel = make_pydantic_model_from_sqlalchemy(
        Person,
        name_suffix="Create",
        exclude_fields={"id", "ppl_CreationTS", "ppl_ModificationTS"},
    )
    _add_crud_endpoints(router, crud, ReadModel, CreateModel, tag="Person")

    app = FastAPI()
    app.include_router(router)

    def override_session():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_session

    with TestClient(app) as client:
        client.session_factory = session_factory  # type: ignore[attr-defined]
        yield client


def test_add_crud_endpoints_crud_operations(crud_client):
    payload = {
        "ppl_Name_First": "Jane",
        "ppl_Name_Last": "Doe",
        "ppl_AddedBy": "tester",
    }
    resp = crud_client.post("/people/", json=payload)
    assert resp.status_code == 201
    person_id = resp.json()["id"]

    resp = crud_client.get(f"/people/{person_id}")
    assert resp.status_code == 200
    assert resp.json()["ppl_Name_First"] == "Jane"

    resp = crud_client.delete(f"/people/{person_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


@pytest.fixture
def options_client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/opts.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    session_factory = make_session_factory(engine)

    router = APIRouter(prefix="/people", tags=["Person"])
    crud = PersonCRUD()
    _add_options_endpoints(router, crud, model=Person)

    app = FastAPI()
    app.include_router(router)

    def override_session():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_session

    with TestClient(app) as client:
        client.session_factory = session_factory  # type: ignore[attr-defined]
        yield client


def test_add_options_endpoints_returns_options(options_client):
    with options_client.session_factory() as db:  # type: ignore[attr-defined]
        PersonCRUD().create(
            db,
            {
                "ppl_Name_First": "John",
                "ppl_Name_Last": "Doe",
                "ppl_AddedBy": "tester",
            },
        )

    resp = options_client.get("/people/options")
    assert resp.status_code == 200
    data = resp.json()
    assert data and data[0]["value"] > 0
