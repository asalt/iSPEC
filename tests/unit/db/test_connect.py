from ispec.db import connect


def _clear_caches():
    connect.get_db_path.cache_clear()
    connect.get_db_dir.cache_clear()


def test_get_db_path_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_DB_DIR", str(tmp_path))
    _clear_caches()
    expected = "sqlite:///" + str(tmp_path / "ispec.db")
    assert connect.get_db_path() == expected


def test_get_db_path_custom(tmp_path):
    _clear_caches()
    custom = tmp_path / "custom.db"
    expected = "sqlite:///" + str(custom)
    assert connect.get_db_path(custom) == expected

