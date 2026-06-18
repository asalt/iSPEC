from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ispec.db.models import AuthUser, AuthUserProject, Project, ProjectAccessMode, UserRole


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip()
    return text or None


def effective_project_access_mode(user: AuthUser | None) -> ProjectAccessMode:
    """Return the effective project data scope for an account.

    The stored column is intentionally nullable so existing accounts can keep
    role-derived defaults:
    - admin/editor: all projects
    - viewer/client: explicit grants only
    """

    if user is None:
        return ProjectAccessMode.all_projects

    raw_mode = _enum_value(getattr(user, "project_access_mode", None))
    if raw_mode:
        try:
            return ProjectAccessMode(raw_mode)
        except ValueError:
            pass

    role = _enum_value(getattr(user, "role", None))
    if role in {UserRole.client.value, UserRole.viewer.value}:
        return ProjectAccessMode.explicit_projects
    return ProjectAccessMode.all_projects


def uses_explicit_project_access(user: AuthUser | None) -> bool:
    return effective_project_access_mode(user) == ProjectAccessMode.explicit_projects


def scope_project_query(query: Any, user: AuthUser | None) -> Any:
    """Apply project grant scoping to a SQLAlchemy query over Project rows."""

    if not uses_explicit_project_access(user):
        return query
    return query.join(
        AuthUserProject,
        AuthUserProject.project_id == Project.id,
    ).filter(AuthUserProject.user_id == user.id)  # type: ignore[union-attr]


def get_project_for_user(
    db: Session,
    *,
    user: AuthUser | None,
    project_id: int,
) -> Project | None:
    query = db.query(Project).filter(Project.id == int(project_id))
    return scope_project_query(query, user).first()


def can_access_project(
    db: Session,
    *,
    user: AuthUser | None,
    project_id: int,
) -> bool:
    return get_project_for_user(db, user=user, project_id=int(project_id)) is not None
