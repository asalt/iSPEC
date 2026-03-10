import pytest
from sqlalchemy import select

from ispec.db import connect as db_connect
from ispec.db.connect import get_session
from ispec.db.models import OmicsDatabaseRegistry
from ispec.omics import connect as omics_connect
from ispec.omics.connect import OmicsDatabaseUnavailableError, get_omics_session


def test_get_omics_session_tracks_registry_and_marks_available(tmp_path):
    core_db_path = tmp_path / "core.db"
    omics_db_path = tmp_path / "analysis.db"

    with get_session(file_path=str(core_db_path)) as core_session:
        with get_omics_session(file_path=str(omics_db_path), core_session=core_session):
            pass

    assert omics_db_path.exists()

    with get_session(file_path=str(core_db_path)) as core_session:
        row = core_session.execute(
            select(OmicsDatabaseRegistry).where(
                OmicsDatabaseRegistry.omdb_LogicalName == "analysis"
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


def test_get_omics_db_uri_defaults_to_sibling_analysis_db(tmp_path, monkeypatch):
    main_db_path = tmp_path / "main.db"
    expected = tmp_path / "ispec-analysis.db"
    monkeypatch.delenv("ISPEC_ANALYSIS_DB_PATH", raising=False)
    monkeypatch.delenv("ISPEC_OMICS_DB_PATH", raising=False)
    monkeypatch.setenv("ISPEC_DB_PATH", str(main_db_path))
    db_connect.get_db_dir.cache_clear()
    db_connect.get_db_path.cache_clear()

    assert omics_connect.get_omics_db_uri() == f"sqlite:///{expected}"


def test_get_omics_db_uri_uses_legacy_alias_for_analysis_db(tmp_path, monkeypatch):
    alias_db_path = tmp_path / "alias-analysis.db"
    monkeypatch.delenv("ISPEC_ANALYSIS_DB_PATH", raising=False)
    monkeypatch.setenv("ISPEC_OMICS_DB_PATH", str(alias_db_path))
    db_connect.get_db_dir.cache_clear()
    db_connect.get_db_path.cache_clear()

    assert omics_connect.get_omics_db_uri() == f"sqlite:///{alias_db_path}"


def test_get_omics_db_uri_falls_back_to_default_sibling_db(tmp_path, monkeypatch):
    monkeypatch.delenv("ISPEC_DB_PATH", raising=False)
    monkeypatch.delenv("ISPEC_ANALYSIS_DB_PATH", raising=False)
    monkeypatch.delenv("ISPEC_OMICS_DB_PATH", raising=False)
    monkeypatch.setenv("ISPEC_DB_DIR", str(tmp_path))
    db_connect.get_db_dir.cache_clear()
    db_connect.get_db_path.cache_clear()

    assert omics_connect.get_omics_db_uri() == f"sqlite:///{tmp_path / 'ispec-analysis.db'}"


def test_get_omics_db_uri_uses_psm_database_env(tmp_path, monkeypatch):
    psm_db_path = tmp_path / "psm.db"
    monkeypatch.setenv("ISPEC_PSM_DB_PATH", str(psm_db_path))

    assert omics_connect.get_omics_db_uri(logical_name="psm") == f"sqlite:///{psm_db_path}"
