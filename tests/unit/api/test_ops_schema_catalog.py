from __future__ import annotations

import json
from pathlib import Path

from ispec.api.routes.ops import schema_catalog, snapshot_schema_catalog
from ispec.db.models import AuthUser, UserRole


def _user(username: str = "alex", *, role: UserRole = UserRole.editor) -> AuthUser:
    return AuthUser(
        id=7,
        username=username,
        password_hash="test",
        password_salt="test",
        role=role,
        is_active=True,
    )


def _resource(catalog: dict, name: str) -> dict:
    for resource in catalog.get("resources") or []:
        if resource.get("resource") == name:
            return resource
    raise AssertionError(f"missing resource: {name}")


def test_ops_schema_catalog_includes_lab_fk_ui(monkeypatch):
    monkeypatch.setenv("ISPEC_API_RESOURCES", "all")

    catalog = schema_catalog(user=_user())

    assert catalog["ok"] is True
    assert catalog["resource_count"] >= 4
    assert catalog["catalog_hash"]

    assays = _resource(catalog, "assays")
    primary_reagent = assays["schema"]["properties"]["primary_reagent_id"]["ui"]
    assert primary_reagent["component"] == "SelectAsync"
    assert primary_reagent["optionsEndpoint"] == "/reagents/options"

    experiments = _resource(catalog, "experiments")
    assay = experiments["schema"]["properties"]["assay_id"]["ui"]
    assert assay["component"] == "SelectAsync"
    assert assay["optionsEndpoint"] == "/assays/options"


def test_ops_schema_snapshot_writes_incremental_json(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_API_RESOURCES", "all")
    monkeypatch.setenv("ISPEC_SCHEMA_SNAPSHOT_DIR", str(tmp_path))

    response = snapshot_schema_catalog(user=_user(role=UserRole.admin))

    assert response["ok"] is True
    path = Path(response["path"])
    assert path.exists()
    assert path.parent == tmp_path
    assert path.name.startswith("schema_catalog_")
    assert path.name.endswith(".json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["catalog_hash"] == response["catalog_hash"]
    assert payload["snapshot_by"]["username"] == "alex"
    assert _resource(payload, "reagents")["table"] == "reagent"
