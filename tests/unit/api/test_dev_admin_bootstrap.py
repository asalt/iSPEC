import importlib


def test_disabled_bootstrap_cannot_open_auth_database_even_with_stale_reset_flag(monkeypatch):
    main = importlib.import_module("ispec.api.main")
    monkeypatch.setenv("ISPEC_DEV_DEFAULT_ADMIN", "0")
    monkeypatch.setenv("ISPEC_DEV_ADMIN_RESET_PASSWORD", "1")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Disabled bootstrap must not touch auth storage")

    monkeypatch.setattr("ispec.db.connect.get_session", forbidden)
    main._bootstrap_dev_admin_user()
    # The startup hook catches exceptions, so asserting only inside the stub
    # would falsely pass if a future regression touched storage.
    assert calls == []
