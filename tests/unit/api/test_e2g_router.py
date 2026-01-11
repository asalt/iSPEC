import inspect
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import sessionmaker

from ispec.api.routes.routes import generate_crud_router
from ispec.db.connect import sqlite_engine
from ispec.db.crud import E2GCRUD
from ispec.omics.models import E2G, OmicsBase


def _find_endpoint(router, *, path: str, method: str):
    for route in router.routes:
        if getattr(route, "path", None) != path:
            continue
        methods = getattr(route, "methods", set()) or set()
        if method in methods:
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


class _DummyRequest:
    def __init__(self):
        self.state = SimpleNamespace()


@pytest.fixture
def e2g_router_env(tmp_path):
    db_url = f"sqlite:///{tmp_path}/e2g.db"
    engine = sqlite_engine(db_url)
    OmicsBase.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    router = generate_crud_router(
        model=E2G,
        crud_class=E2GCRUD,
        prefix="/experiment_to_gene",
        tag="E2G",
        exclude_fields=set(),
        create_exclude_fields={"id", "E2G_CreationTS", "E2G_ModificationTS"},
        route_prefix_by_table={},
    )
    return router, SessionLocal


def test_e2g_router_exposes_by_run_endpoint(e2g_router_env):
    router, SessionLocal = e2g_router_env
    endpoint = _find_endpoint(router, path="/experiment_to_gene/by_run/{run_id}", method="GET")
    with SessionLocal() as db:
        assert endpoint(run_id=1, q=None, geneidtype=None, limit=100, offset=0, db=db) == []


def test_e2g_crud(e2g_router_env):
    router, SessionLocal = e2g_router_env
    create = _find_endpoint(router, path="/experiment_to_gene", method="POST")
    get_item = _find_endpoint(router, path="/experiment_to_gene/{item_id}", method="GET")
    list_by_run = _find_endpoint(router, path="/experiment_to_gene/by_run/{run_id}", method="GET")
    delete = _find_endpoint(router, path="/experiment_to_gene/{item_id}", method="DELETE")

    payload_model = inspect.signature(create).parameters["payload"].annotation
    payload = payload_model(experiment_run_id=1, gene="TP53", geneidtype="symbol")

    with SessionLocal() as db:
        created = create(payload, db=db)
        e2g_id = created["id"]

        got = get_item(e2g_id, request=_DummyRequest(), db=db)
        assert got["experiment_run_id"] == 1
        assert got["gene"] == "TP53"

        items = list_by_run(run_id=1, q=None, geneidtype=None, limit=100, offset=0, db=db)
        assert any(row["id"] == e2g_id for row in items)

        resp = delete(e2g_id, db=db)
        assert resp["status"] == "deleted"

        with pytest.raises(HTTPException) as exc:
            get_item(e2g_id, request=_DummyRequest(), db=db)
        assert exc.value.status_code == 404
