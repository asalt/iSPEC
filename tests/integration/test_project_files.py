import logging

import pytest
from fastapi.testclient import TestClient

from ispec.api.main import app
from ispec.db.connect import get_session as db_get_session, get_session_dep
from ispec.db.models import Project, logger as db_logger

pytestmark = pytest.mark.testclient


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


def _create_project(client: TestClient) -> int:
    payload = {
        "prj_AddedBy": "tester",
        "prj_ProjectTitle": "Project Files",
        "prj_ProjectBackground": "test",
    }
    resp = client.post("/projects/", json=payload)
    assert resp.status_code == 201, resp.text

    with db_get_session() as session:
        project_id = session.query(Project.id).scalar()
        assert project_id is not None
        return int(project_id)


def test_project_file_upload_list_download_preview_delete(client: TestClient):
    project_id = _create_project(client)

    upload = client.post(
        f"/api/projects/{project_id}/files",
        files={"file": ("plot.png", b"abc", "application/octet-stream")},
    )
    assert upload.status_code == 201, upload.text
    payload = upload.json()
    assert payload["project_id"] == project_id
    assert payload["prjfile_FileName"] == "plot.png"
    assert payload["prjfile_ContentType"] == "image/png"
    assert payload["prjfile_SizeBytes"] == 3
    file_id = payload["id"]

    listing = client.get(f"/api/projects/{project_id}/files")
    assert listing.status_code == 200, listing.text
    assert [row["id"] for row in listing.json()] == [file_id]

    download = client.get(f"/api/projects/{project_id}/files/{file_id}")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("image/png")
    assert download.headers["content-disposition"] == 'attachment; filename="plot.png"'

    preview = client.get(f"/api/projects/{project_id}/files/{file_id}/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/png")
    assert preview.headers["content-disposition"] == 'inline; filename="plot.png"'

    bad_upload = client.post(
        f"/api/projects/{project_id}/files",
        files={"file": ("report.html", b"<h1>hello</h1>", "text/html")},
    )
    assert bad_upload.status_code == 201, bad_upload.text
    bad_id = bad_upload.json()["id"]

    bad_preview = client.get(f"/api/projects/{project_id}/files/{bad_id}/preview")
    assert bad_preview.status_code == 415

    deleted = client.delete(f"/api/projects/{project_id}/files/{file_id}")
    assert deleted.status_code == 204, deleted.text

