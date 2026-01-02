from __future__ import annotations

# src/ispec/api/main.py
import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ispec.api.routes.auth import router as auth_router
from ispec.api.routes.routes import router
from ispec.api.security import require_access, require_api_key
from ispec.logging import get_logger

logger = get_logger(__name__)


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


_DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

cors_origins = _parse_csv_list(os.getenv("ISPEC_CORS_ORIGINS")) or _DEFAULT_CORS_ORIGINS
cors_allow_credentials = _is_truthy(os.getenv("ISPEC_CORS_ALLOW_CREDENTIALS"))
legacy_root_routes = _is_truthy(os.getenv("ISPEC_API_LEGACY_ROOT_ROUTES", "1"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/status")
def status():
    return {"ok": True}


# Auth endpoints live under /api/auth.
app.include_router(auth_router, prefix="/api", dependencies=[Depends(require_api_key)])

# Protected CRUD routes (both legacy root routes and /api/*).
if legacy_root_routes:
    app.include_router(router, dependencies=[Depends(require_access)])
app.include_router(router, prefix="/api", dependencies=[Depends(require_access)])


@app.on_event("startup")
def _bootstrap_dev_admin_user() -> None:
    """Optionally ensure a default dev admin user exists.

    This is intentionally opt-in via env vars, since it creates a known
    credential suitable only for local development environments.
    """

    if not _is_truthy(os.getenv("ISPEC_DEV_DEFAULT_ADMIN")):
        return

    username = (os.getenv("ISPEC_DEV_ADMIN_USERNAME") or "admin").strip()
    password = os.getenv("ISPEC_DEV_ADMIN_PASSWORD") or "admin"
    if not username:
        logger.warning("ISPEC_DEV_DEFAULT_ADMIN is enabled but username is empty; skipping.")
        return

    try:
        from ispec.api.security import hash_password
        from ispec.db.connect import get_session
        from ispec.db.models import AuthUser, UserRole

        with get_session() as db:
            if db.query(AuthUser).count() > 0:
                return
            salt_b64, hash_b64, iterations = hash_password(password)
            user = AuthUser(
                username=username,
                password_hash=hash_b64,
                password_salt=salt_b64,
                password_iterations=iterations,
                role=UserRole.admin,
                is_active=True,
            )
            db.add(user)
            db.commit()

        logger.warning(
            "Dev admin user created (username=%r). Change this for non-dev use.",
            username,
        )
    except Exception:
        logger.exception("Failed to bootstrap dev admin user.")
