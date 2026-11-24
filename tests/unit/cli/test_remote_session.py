from unittest.mock import Mock

import pytest

from ispec.cli.remote_session import RemoteSession
from ispec.cli.support import _post_chat


def session(directory, origin="https://ispec.test"):
    return RemoteSession(server=origin, account="dot", directory=directory)


@pytest.mark.parametrize(
    "origin",
    ["http://ispec.test", "https://user@ispec.test", "https://ispec.test/path"],
)
def test_login_requires_bare_https_origin(tmp_path, origin):
    with pytest.raises(ValueError):
        session(tmp_path, origin)


def test_cookie_is_private_and_bound_to_origin_and_account(tmp_path):
    login = session(tmp_path)
    assert login.path != session(tmp_path, "https://ispec.test:3001").path
    assert (
        login.path
        != RemoteSession(server=login.origin, account="alex", directory=tmp_path).path
    )
    login._private_directory()
    login.path.touch()
    login.path.chmod(0o644)
    login.request = Mock()
    with pytest.raises(ValueError, match="private"):
        login.resume()
    login.request.assert_not_called()
    with pytest.raises(ValueError):
        login._verify({"username": "alex", "is_active": True})


def test_named_chat_sends_only_message_and_explicit_destination(tmp_path):
    login = session(tmp_path)
    login.request = Mock(
        return_value={"sessionId": "server-selected", "message": "Ready for review."}
    )
    result = _post_chat(
        server=login.origin,
        api_key="UNUSED",
        message="Draft a note",
        session_id=None,
        login=login,
        staff_review=True,
        timeout_seconds=30,
    )
    assert result.session_id == "server-selected"
    login.request.assert_called_once_with(
        "support/chat",
        {"message": "Draft a note", "reviewDestination": "staff_slack"},
        timeout=30,
    )


def test_login_does_not_follow_redirects(tmp_path):
    login = session(tmp_path)
    login.http.request = Mock(return_value=Mock(status_code=307))
    with pytest.raises(ValueError, match="redirect"):
        login.request("auth/me")
    assert login.http.request.call_args.kwargs["allow_redirects"] is False
