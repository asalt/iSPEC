"""Named HTTPS sessions for the existing API client commands."""

from getpass import getpass
from hashlib import sha256
from http.cookiejar import MozillaCookieJar
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import requests


class RemoteSession:
    def __init__(self, *, server: str, account: str, directory: Path | None = None):
        url = urlsplit(server)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("Specify an HTTPS origin with no path, user or password.")
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}", account):
            raise ValueError("Invalid account name.")
        self.origin = f"https://{url.netloc.lower()}"
        self.account = account
        root = (
            directory
            or Path(os.getenv("XDG_STATE_HOME") or Path.home() / ".local/state")
            / "ispec/sessions"
        )
        self.directory = root / sha256(self.origin.encode()).hexdigest()[:24]
        self.path = self.directory / (account + ".cookies")
        self.jar = MozillaCookieJar()
        self.http = requests.Session()
        self.http.trust_env = False  # A named login never inherits netrc credentials.
        self.http.cookies = self.jar

    def request(self, path: str, data: dict | None = None, *, timeout: float = 30):
        response = self.http.request(
            "GET" if data is None else "POST",
            f"{self.origin}/api/{path}",
            json=data,
            allow_redirects=False,
            timeout=timeout,
        )
        if 300 <= response.status_code < 400:
            raise ValueError("Refusing a redirect from the configured iSPEC origin.")
        response.raise_for_status()
        return response.json()

    def _private_directory(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.directory.lstat()
        if (
            self.directory.is_symlink()
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError(
                "The session directory must be private and owned by this user."
            )

    def _verify(self, identity):
        if (
            identity.get("username") != self.account
            or identity.get("is_active") is not True
            or identity.get("must_change_password")
        ):
            raise ValueError(
                "Expected this active account with no pending password change."
            )

    def login(self):
        self._private_directory()
        self._verify(
            self.request(
                "auth/login", {"username": self.account, "password": getpass()}
            )
        )
        if not list(self.jar):
            raise ValueError("No session cookie was issued.")
        with NamedTemporaryFile(dir=self.directory, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            self.jar.save(str(temporary), ignore_discard=True)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def resume(self):
        self._private_directory()
        info = self.path.lstat()
        if (
            self.path.is_symlink()
            or not self.path.is_file()
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError(
                "The session cookie file must be private and owned by this user."
            )
        self.jar.load(str(self.path), ignore_discard=True)
        self._verify(self.request("auth/me"))
