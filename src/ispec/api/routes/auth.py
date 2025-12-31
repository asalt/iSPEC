from __future__ import annotations

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
from ispec.db.models import AuthUser, UserRole


router = APIRouter(prefix="/auth", tags=["Auth"])


class UserOut(BaseModel):
    id: int
    username: str
    role: UserRole
    is_active: bool


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


def _user_out(user: AuthUser) -> UserOut:
    return UserOut(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
    )


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
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_out(user)
