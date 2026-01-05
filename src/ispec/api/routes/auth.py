from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ispec.api.security import (
    clear_session_cookie,
    create_session,
    delete_session,
    hash_password,
    require_admin,
    require_user,
    session_cookie_name,
    set_session_cookie,
    verify_password,
)
from ispec.db.connect import get_session_dep
from ispec.db.models import AuthSession, AuthUser, AuthUserProject, Project, UserRole


router = APIRouter(prefix="/auth", tags=["Auth"])


class UserOut(BaseModel):
    id: int
    username: str
    role: UserRole
    is_active: bool
    must_change_password: bool


class BootstrapRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=8, max_length=1024)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=8, max_length=1024)
    role: UserRole = UserRole.editor
    must_change_password: bool = False


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=8, max_length=1024)


class ResetPasswordRequest(BaseModel):
    password: str = Field(min_length=8, max_length=1024)
    must_change_password: bool = True


class UserProjectsOut(BaseModel):
    user_id: int
    project_ids: list[int]


class UserProjectsUpdate(BaseModel):
    project_ids: list[int] = Field(default_factory=list)


def _user_out(user: AuthUser) -> UserOut:
    return UserOut(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        must_change_password=bool(getattr(user, "must_change_password", False)),
    )


def _invalidate_user_sessions(db: Session, *, user_id: int) -> None:
    db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()


@router.post("/bootstrap", response_model=UserOut, status_code=201)
def bootstrap(payload: BootstrapRequest, response: Response, db: Session = Depends(get_session_dep)):
    """Create the first (admin) user if no users exist yet."""

    if db.query(AuthUser).count() > 0:
        raise HTTPException(status_code=409, detail="Users already exist.")

    salt_b64, hash_b64, iterations = hash_password(payload.password)
    user = AuthUser(
        username=payload.username,
        password_hash=hash_b64,
        password_salt=salt_b64,
        password_iterations=iterations,
        role=UserRole.admin,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_session(db, user=user)
    set_session_cookie(response, token=token)
    return _user_out(user)


@router.post("/login", response_model=UserOut)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_session_dep)):
    user = db.query(AuthUser).filter(AuthUser.username == payload.username).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    if not verify_password(
        payload.password,
        salt_b64=user.password_salt,
        hash_b64=user.password_hash,
        iterations=user.password_iterations,
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    user.last_login_at = datetime.now(UTC)
    token = create_session(db, user=user)
    set_session_cookie(response, token=token)
    return _user_out(user)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_session_dep)):
    token = request.cookies.get(session_cookie_name())
    if token:
        delete_session(db, token=token)
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/me", response_model=UserOut)
def me(user: AuthUser = Depends(require_user)):
    return _user_out(user)


@router.get("/users", response_model=list[UserOut])
def list_users(_: AuthUser = Depends(require_admin), db: Session = Depends(get_session_dep)):
    users = db.query(AuthUser).order_by(AuthUser.username.asc()).all()
    return [_user_out(u) for u in users]


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(payload: CreateUserRequest, _: AuthUser = Depends(require_admin), db: Session = Depends(get_session_dep)):
    existing = db.query(AuthUser).filter(AuthUser.username == payload.username).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username already exists.")

    salt_b64, hash_b64, iterations = hash_password(payload.password)
    user = AuthUser(
        username=payload.username,
        password_hash=hash_b64,
        password_salt=salt_b64,
        password_iterations=iterations,
        role=payload.role,
        is_active=True,
        must_change_password=bool(payload.must_change_password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_out(user)


@router.post("/change-password", response_model=UserOut)
def change_password(
    payload: ChangePasswordRequest,
    response: Response,
    user: AuthUser = Depends(require_user),
    db: Session = Depends(get_session_dep),
):
    if not verify_password(
        payload.current_password,
        salt_b64=user.password_salt,
        hash_b64=user.password_hash,
        iterations=user.password_iterations,
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    salt_b64, hash_b64, iterations = hash_password(payload.new_password)
    user.password_hash = hash_b64
    user.password_salt = salt_b64
    user.password_iterations = iterations
    user.must_change_password = False
    user.password_changed_at = datetime.now(UTC)
    _invalidate_user_sessions(db, user_id=user.id)
    db.add(user)
    db.commit()
    db.refresh(user)

    # Rotate session token after password change (defense-in-depth).
    token = create_session(db, user=user)
    set_session_cookie(response, token=token)
    return _user_out(user)


@router.post("/users/{user_id}/reset-password", response_model=UserOut)
def reset_password(
    user_id: int,
    payload: ResetPasswordRequest,
    _: AuthUser = Depends(require_admin),
    db: Session = Depends(get_session_dep),
):
    user = db.get(AuthUser, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    salt_b64, hash_b64, iterations = hash_password(payload.password)
    user.password_hash = hash_b64
    user.password_salt = salt_b64
    user.password_iterations = iterations
    user.must_change_password = bool(payload.must_change_password)
    user.password_changed_at = datetime.now(UTC)
    _invalidate_user_sessions(db, user_id=user.id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_out(user)


@router.get("/users/{user_id}/projects", response_model=UserProjectsOut)
def list_user_projects(
    user_id: int,
    _: AuthUser = Depends(require_admin),
    db: Session = Depends(get_session_dep),
):
    user = db.get(AuthUser, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    project_ids = [
        int(row[0])
        for row in (
            db.query(AuthUserProject.project_id)
            .filter(AuthUserProject.user_id == user_id)
            .order_by(AuthUserProject.project_id.asc())
            .all()
        )
    ]
    return UserProjectsOut(user_id=user_id, project_ids=project_ids)


@router.put("/users/{user_id}/projects", response_model=UserProjectsOut)
def update_user_projects(
    user_id: int,
    payload: UserProjectsUpdate,
    _: AuthUser = Depends(require_admin),
    db: Session = Depends(get_session_dep),
):
    user = db.get(AuthUser, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    desired: list[int] = []
    for value in payload.project_ids:
        if value not in desired:
            desired.append(int(value))

    if desired:
        existing_projects = {
            int(row[0]) for row in db.query(Project.id).filter(Project.id.in_(desired)).all()
        }
        missing = [pid for pid in desired if pid not in existing_projects]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown project ids: {', '.join(map(str, missing))}",
            )

    db.query(AuthUserProject).filter(AuthUserProject.user_id == user_id).delete()
    for project_id in desired:
        db.add(AuthUserProject(user_id=user_id, project_id=project_id))
    db.commit()

    return UserProjectsOut(user_id=user_id, project_ids=desired)
