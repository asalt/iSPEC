import argparse
import types
from unittest.mock import MagicMock

import pytest

from ispec.cli import auth
from ispec.cli.auth import ProvisionedCredential


@pytest.fixture(autouse=True)
def _fast_password_iterations(monkeypatch):
    """Avoid slow PBKDF2 in CLI provisioning tests."""

    import ispec.api.security as security

    monkeypatch.setattr(security, "_password_iterations", lambda: 1)


def test_register_subcommands_parses_provision_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    auth.register_subcommands(subparsers)

    args = parser.parse_args(["provision", "antrixj", "shirley", "--output", "out.csv"])
    assert args.subcommand == "provision"
    assert args.usernames == ["antrixj", "shirley"]
    assert args.role == "editor"
    assert args.reset_existing is False
    assert args.password_length == 16
    assert args.output == "out.csv"


def test_register_subcommands_parses_check_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    auth.register_subcommands(subparsers)

    args = parser.parse_args(["check", "antrixj", "pw", "--database", "db.sqlite"])
    assert args.subcommand == "check"
    assert args.username == "antrixj"
    assert args.password == "pw"
    assert args.database == "db.sqlite"


def test_dispatch_provision_wires_through(monkeypatch, tmp_path):
    creds = [
        ProvisionedCredential(
            user_id=1,
            username="antrixj",
            password="pw",
            role="editor",
            action="created",
        )
    ]

    provision_mock = MagicMock(return_value=creds)
    write_mock = MagicMock(return_value="csv")
    render_mock = MagicMock()
    monkeypatch.setattr(auth, "provision_users", provision_mock)
    monkeypatch.setattr(auth, "_write_credentials", write_mock)
    monkeypatch.setattr(auth, "_render_summary", render_mock)

    args = types.SimpleNamespace(
        subcommand="provision",
        usernames=["antrixj"],
        database=None,
        role="editor",
        reset_existing=False,
        password_length=16,
        output=str(tmp_path / "out.csv"),
        print_passwords=False,
        activate=True,
        update_role=True,
    )

    auth.dispatch(args)

    provision_mock.assert_called_once()
    write_mock.assert_called_once()
    render_mock.assert_called_once()


@pytest.mark.parametrize("reset_existing", [False, True])
def test_provision_users_creates_users(tmp_path, reset_existing):
    db_path = tmp_path / "test.db"

    creds = auth.provision_users(
        ["antrixj", "shirley"],
        database=str(db_path),
        role="editor",
        reset_existing=reset_existing,
        password_length=12,
        activate=True,
        update_role=True,
    )

    assert {c.username for c in creds} == {"antrixj", "shirley"}
    assert all(c.action == "created" for c in creds)
    assert all(len(c.password) == 12 for c in creds)

    from ispec.db.connect import get_session
    from ispec.db.models import AuthUser, UserRole

    with get_session(str(db_path)) as session:
        users = session.query(AuthUser).order_by(AuthUser.username.asc()).all()
        assert [u.username for u in users] == ["antrixj", "shirley"]
        assert all(u.role == UserRole.editor for u in users)
        assert all(u.must_change_password is True for u in users)
        assert all(u.is_active is True for u in users)


def test_provision_users_resets_existing_when_enabled(tmp_path):
    db_path = tmp_path / "test.db"

    first = auth.provision_users(
        ["antrixj"],
        database=str(db_path),
        role="editor",
        reset_existing=False,
        password_length=12,
        activate=True,
        update_role=True,
    )
    assert first[0].action == "created"

    second = auth.provision_users(
        ["antrixj"],
        database=str(db_path),
        role="editor",
        reset_existing=True,
        password_length=12,
        activate=True,
        update_role=True,
    )
    assert second[0].action == "reset"
    assert second[0].password != first[0].password


def test_provision_users_errors_on_existing_without_reset(tmp_path):
    db_path = tmp_path / "test.db"

    _ = auth.provision_users(
        ["antrixj"],
        database=str(db_path),
        role="editor",
        reset_existing=False,
        password_length=12,
        activate=True,
        update_role=True,
    )

    with pytest.raises(SystemExit):
        auth.provision_users(
            ["antrixj"],
            database=str(db_path),
            role="editor",
            reset_existing=False,
            password_length=12,
            activate=True,
            update_role=True,
        )


def test_check_credentials_round_trip(tmp_path):
    db_path = tmp_path / "test.db"
    creds = auth.provision_users(
        ["antrixj"],
        database=str(db_path),
        role="editor",
        reset_existing=False,
        password_length=12,
        activate=True,
        update_role=True,
    )
    password = creds[0].password

    ok = auth.check_credentials(
        username="antrixj",
        password=password,
        database=str(db_path),
    )
    assert ok.found is True
    assert ok.password_ok is True

    bad = auth.check_credentials(
        username="antrixj",
        password=password + "x",
        database=str(db_path),
    )
    assert bad.found is True
    assert bad.password_ok is False

    missing = auth.check_credentials(
        username="missing",
        password="pw",
        database=str(db_path),
    )
    assert missing.found is False
    assert missing.password_ok is False
