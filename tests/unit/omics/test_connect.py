import pytest
from sqlalchemy import select

from ispec.db.connect import get_session
from ispec.db.models import OmicsDatabaseRegistry
from ispec.omics import connect as omics_connect
from ispec.omics.connect import OmicsDatabaseUnavailableError, get_omics_session


def test_get_omics_session_tracks_registry_and_marks_available(tmp_path):
    core_db_path = tmp_path / "core.db"
    omics_db_path = tmp_path / "omics.db"

    with get_session(file_path=str(core_db_path)) as core_session:
        with get_omics_session(file_path=str(omics_db_path), core_session=core_session):
            pass

    assert omics_db_path.exists()

    with get_session(file_path=str(core_db_path)) as core_session:
        row = core_session.execute(
            select(OmicsDatabaseRegistry).where(
                OmicsDatabaseRegistry.omdb_LogicalName == "primary"
            )
        ).scalar_one()

        assert row.omdb_DBPath == str(omics_db_path)
        assert row.omdb_Status == "available"
        assert row.omdb_LastAvailableTS is not None
        assert row.omdb_LastError is None


def test_get_omics_session_refuses_recreate_for_missing_known_db(tmp_path):
    core_db_path = tmp_path / "core.db"
    omics_db_path = tmp_path / "omics.db"

    with get_session(file_path=str(core_db_path)) as core_session:
        with get_omics_session(file_path=str(omics_db_path), core_session=core_session):
            pass

    assert omics_db_path.exists()

    omics_connect._get_engine.cache_clear()
    omics_db_path.unlink()

    with get_session(file_path=str(core_db_path)) as core_session:
        with pytest.raises(OmicsDatabaseUnavailableError, match="Refusing to auto-create"):
            with get_omics_session(file_path=str(omics_db_path), core_session=core_session):
                pass

    assert not omics_db_path.exists()

    with get_session(file_path=str(core_db_path)) as core_session:
        with get_omics_session(
            file_path=str(omics_db_path),
            core_session=core_session,
            allow_recreate_missing=True,
        ):
            pass

    assert omics_db_path.exists()
