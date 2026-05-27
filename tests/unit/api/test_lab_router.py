from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import generate_crud_router
from ispec.db.connect import (
    get_session_dep,
    initialize_db,
    make_session_factory,
    sqlite_engine,
)
from ispec.db.crud import AssayCRUD, ReagentCRUD
from ispec.db.models import Assay, Reagent

pytestmark = pytest.mark.testclient


@pytest.fixture
def client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/lab.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    test_session = make_session_factory(engine)

    app = FastAPI()
    route_prefix_map: dict[str, str] = {}
    app.include_router(
        generate_crud_router(
            model=Reagent,
            crud_class=ReagentCRUD,
            prefix="/reagents",
            tag="Reagent",
            exclude_fields={"id"},
            route_prefix_by_table=route_prefix_map,
        )
    )
    app.include_router(
        generate_crud_router(
            model=Assay,
            crud_class=AssayCRUD,
            prefix="/assays",
            tag="Assay",
            exclude_fields={"id"},
            route_prefix_by_table=route_prefix_map,
        )
    )

    def override_get_session():
        with test_session() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session

    with TestClient(app) as client:
        yield client


def test_lab_crud_and_fk_schema(client):
    schema = client.get("/assays/schema")
    assert schema.status_code == 200
    reagent_ui = schema.json()["properties"]["primary_reagent_id"]["ui"]
    assert reagent_ui["component"] == "SelectAsync"
    assert reagent_ui["optionsEndpoint"] == "/reagents/options"

    reagent_resp = client.post(
        "/reagents",
        json={"name": "Trypsin", "vendor": "Promega", "room_number": "1420"},
    )
    assert reagent_resp.status_code == 201
    reagent_id = reagent_resp.json()["id"]

    assay_resp = client.post(
        "/assays",
        json={"name": "Digest LC-MS", "primary_reagent_id": reagent_id},
    )
    assert assay_resp.status_code == 201
    assert assay_resp.json()["primary_reagent_id"] == reagent_id

    options = client.get("/reagents/options").json()
    assert any(
        row["value"] == reagent_id and "Trypsin" in row["label"] for row in options
    )
