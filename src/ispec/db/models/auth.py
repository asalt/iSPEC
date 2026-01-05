from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, make_timestamp_mixin


AuthTimestamp = make_timestamp_mixin("auth")


class UserRole(str, enum.Enum):
    admin = "admin"
    editor = "editor"
    viewer = "viewer"
    client = "client"


class AuthUser(AuthTimestamp, Base):
    __tablename__ = "auth_user"

    id: Mapped[int] = mapped_column(primary_key=True)

    username: Mapped[str] = mapped_column(Text, unique=True, index=True)

    password_hash: Mapped[str] = mapped_column(Text)
    password_salt: Mapped[str] = mapped_column(Text)
    password_iterations: Mapped[int] = mapped_column(default=250_000)

    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, native_enum=True, validate_strings=True),
        default=UserRole.editor,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    sessions: Mapped[list["AuthSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    project_access: Mapped[list["AuthUserProject"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class AuthSession(Base):
    __tablename__ = "auth_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("auth_user.id"), index=True)

    token_hash: Mapped[str] = mapped_column(Text, unique=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)

    user: Mapped["AuthUser"] = relationship(back_populates="sessions")


class AuthUserProject(Base):
    __tablename__ = "auth_user_project"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("auth_user.id"),
        primary_key=True,
        index=True,
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("project.id"),
        primary_key=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    user: Mapped["AuthUser"] = relationship(back_populates="project_access")
