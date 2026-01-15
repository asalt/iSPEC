"""API security helpers for the FastAPI service.

This module provides two layers of protection:

1) Optional API key (env: ``ISPEC_API_KEY``)
   - Accepted via ``X-API-Key`` header or ``Authorization: Bearer <key>``.

2) Optional cookie-backed user sessions (env: ``ISPEC_REQUIRE_LOGIN=1``)
   - Login issues an HttpOnly cookie containing a random session token.
   - The token is stored *hashed* in the database.

Notes on password storage:
- Do **not** store raw passwords or fast hashes (e.g. plain SHA256).
- iSPEC uses PBKDF2-HMAC-SHA256 with a per-user random salt and an optional
  server-side "pepper" from env (``ISPEC_PASSWORD_PEPPER``).
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
import hashlib
import os
import secrets

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ispec.db.connect import get_session_dep
from ispec.db.models import AuthSession, AuthUser, AuthUserProject, Project, UserRole

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_BEARER = HTTPBearer(auto_error=False)

_DEFAULT_SESSION_COOKIE = "ispec_session"


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _expected_api_key() -> str | None:
    key = os.getenv("ISPEC_API_KEY")
    if key is None:
        return None
    key = key.strip()
    return key or None


def _provided_api_key(
    x_api_key: str | None = Depends(_API_KEY_HEADER),
    bearer: HTTPAuthorizationCredentials | None = Depends(_BEARER),
) -> str | None:
    if x_api_key:
        return x_api_key
    if bearer and bearer.credentials:
        return bearer.credentials
    return None


def require_api_key(provided: str | None = Depends(_provided_api_key)) -> None:
    """FastAPI dependency enforcing ``ISPEC_API_KEY`` when configured."""

    expected = _expected_api_key()
    if expected is None:
        if _is_truthy(os.getenv("ISPEC_API_REQUIRE_KEY")):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API key enforcement is enabled but ISPEC_API_KEY is not set.",
            )
        return

    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )


def _password_pepper_bytes() -> bytes:
    """Return a server-side pepper derived from an env var (optional).

    This is **not** a substitute for a per-user random salt. It is an additional
    secret mixed into the password hashing step.
    """

    raw = (
        os.getenv("ISPEC_PASSWORD_PEPPER")
        or os.getenv("ISPEC_AUTH_PEPPER")
        or os.getenv("ISPEC_PASSWORD_SALT")
        or ""
    )
    if not raw.strip():
        return b""
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _password_iterations() -> int:
    raw = os.getenv("ISPEC_PASSWORD_ITERATIONS")
    if not raw:
        return 250_000
    try:
        return max(50_000, int(raw))
    except ValueError:
        return 250_000


def hash_password(
    password: str,
    *,
    salt: bytes | None = None,
    iterations: int | None = None,
) -> tuple[str, str, int]:
    """Return (salt_b64, hash_b64, iterations) for the supplied password."""

    if iterations is None:
        iterations = _password_iterations()
    if salt is None:
        salt = secrets.token_bytes(16)
    pepper = _password_pepper_bytes()
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8") + pepper,
        salt,
        iterations,
        dklen=32,
    )
    return (
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(derived).decode("ascii"),
        iterations,
    )


def verify_password(
    password: str,
    *,
    salt_b64: str,
    hash_b64: str,
    iterations: int,
) -> bool:
    """Verify ``password`` against stored PBKDF2 material."""

    try:
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
    except Exception:
        return False

    pepper = _password_pepper_bytes()
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8") + pepper,
        salt,
        iterations,
        dklen=len(expected),
    )
    return secrets.compare_digest(derived, expected)


def _session_cookie_name() -> str:
    name = (os.getenv("ISPEC_SESSION_COOKIE_NAME") or _DEFAULT_SESSION_COOKIE).strip()
    return name or _DEFAULT_SESSION_COOKIE


def session_cookie_name() -> str:
    return _session_cookie_name()


def _session_ttl_seconds() -> int:
    raw = os.getenv("ISPEC_SESSION_TTL_SECONDS")
    if not raw:
        return 60 * 60 * 12  # 12 hours
    try:
        return max(60, int(raw))
    except ValueError:
        return 60 * 60 * 12


def _cookie_samesite() -> str:
    raw = (os.getenv("ISPEC_SESSION_COOKIE_SAMESITE") or "lax").strip().lower()
    if raw in {"lax", "strict", "none"}:
        return raw
    return "lax"


def _cookie_secure() -> bool:
    return _is_truthy(os.getenv("ISPEC_SESSION_COOKIE_SECURE"))


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(db: Session, *, user: AuthUser) -> str:
    """Create a DB-backed session token and return the **raw** token."""

    ttl = _session_ttl_seconds()
    token = secrets.token_urlsafe(32)
    token_hash = _hash_session_token(token)
    now = datetime.now(UTC)
    session = AuthSession(
        user_id=user.id,
        token_hash=token_hash,
        created_at=now,
        expires_at=now + timedelta(seconds=ttl),
    )
    db.add(session)
    db.commit()
    return token


def delete_session(db: Session, *, token: str) -> None:
    token_hash = _hash_session_token(token)
    db.query(AuthSession).filter(AuthSession.token_hash == token_hash).delete()
    db.commit()


def set_session_cookie(response: Response, *, token: str) -> None:
    response.set_cookie(
        _session_cookie_name(),
        token,
        httponly=True,
        samesite=_cookie_samesite(),
        secure=_cookie_secure(),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(_session_cookie_name(), path="/")


def get_current_user(
    request: Request,
    db: Session = Depends(get_session_dep),
) -> AuthUser | None:
    token = request.cookies.get(_session_cookie_name())
    if not token:
        return None

    token_hash = _hash_session_token(token)
    now = datetime.now(UTC)
    row = (
        db.query(AuthSession)
        .join(AuthUser, AuthSession.user_id == AuthUser.id)
        .filter(AuthSession.token_hash == token_hash)
        .filter(AuthSession.expires_at > now)
        .first()
    )
    if row is None:
        return None
    if not row.user.is_active:
        return None
    return row.user


_PASSWORD_CHANGE_ALLOWED_PATHS = {
    "/api/auth/me",
    "/api/auth/logout",
    "/api/auth/change-password",
}


def require_user(
    request: Request,
    user: AuthUser | None = Depends(get_current_user),
) -> AuthUser:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    if getattr(user, "must_change_password", False):
        path = (request.url.path or "").rstrip("/")
        if path not in _PASSWORD_CHANGE_ALLOWED_PATHS:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Password change required.")

    return user


def require_admin(user: AuthUser = Depends(require_user)) -> AuthUser:
    if user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required."
        )
    return user


def require_staff(user: AuthUser = Depends(require_user)) -> AuthUser:
    if user.role not in {UserRole.admin, UserRole.editor}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Staff access required.",
        )
    return user


def get_project_or_404_for_user(
    db: Session,
    *,
    project_id: int,
    user: AuthUser | None,
) -> Project:
    """Fetch a project if it exists and the user can access it.

    Staff/admin bypass project scoping. Client users must be explicitly mapped
    via ``auth_user_project``.
    """

    if user is not None and user.role == UserRole.client:
        project = (
            db.query(Project)
            .join(AuthUserProject, AuthUserProject.project_id == Project.id)
            .filter(
                Project.id == project_id,
                AuthUserProject.user_id == user.id,
            )
            .first()
        )
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return project

    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _require_login() -> bool:
    return _is_truthy(os.getenv("ISPEC_REQUIRE_LOGIN"))


def require_access(
    request: Request,
    db: Session = Depends(get_session_dep),
    provided_api_key: str | None = Depends(_provided_api_key),
) -> AuthUser | None:
    """Dependency guarding protected API routes.

    Rules:
    - If ``ISPEC_API_KEY`` is set, the request must supply that key.
    - If ``ISPEC_REQUIRE_LOGIN`` is truthy, the request must also carry a valid
      session cookie (created via ``/api/auth/login``).
    - When logged in as a ``viewer``, write methods are rejected.
    """

    expected = _expected_api_key()
    if expected is not None and (
        provided_api_key is None or not secrets.compare_digest(provided_api_key, expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )

    if not _require_login():
        request.state.user = None
        return None

    user = get_current_user(request, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated."
        )

    request.state.user = user

    if getattr(user, "must_change_password", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Password change required.")

    if request.method in {"POST", "PUT", "DELETE"} and user.role in {UserRole.viewer, UserRole.client}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Read-only account.")

    return user


def require_assistant_access(
    request: Request,
    db: Session = Depends(get_session_dep),
    provided_api_key: str | None = Depends(_provided_api_key),
) -> AuthUser | None:
    """Access rules for assistant endpoints.

    Same as :func:`require_access` but does not block ``viewer`` users on POST
    requests (assistant chat and feedback still write, but not to core tables).
    """

    expected = _expected_api_key()
    if expected is not None and (
        provided_api_key is None or not secrets.compare_digest(provided_api_key, expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )

    if not _require_login():
        request.state.user = None
        return None

    user = get_current_user(request, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated."
        )

    request.state.user = user

    if getattr(user, "must_change_password", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Password change required.")

    return user
