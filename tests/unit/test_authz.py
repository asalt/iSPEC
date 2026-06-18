from __future__ import annotations

import pytest

from ispec.authz import (
    can_access_project,
    effective_project_access_mode,
    get_project_for_user,
    scope_project_query,
    uses_explicit_project_access,
)
from ispec.db.models import AuthUser, AuthUserProject, Project, ProjectAccessMode, UserRole


def _user(username: str, *, role: UserRole) -> AuthUser:
    return AuthUser(
        username=username,
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=role,
        is_active=True,
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (UserRole.admin, ProjectAccessMode.all_projects),
        (UserRole.editor, ProjectAccessMode.all_projects),
        (UserRole.viewer, ProjectAccessMode.explicit_projects),
        (UserRole.client, ProjectAccessMode.explicit_projects),
    ],
)
def test_effective_project_access_mode_defaults_by_role(role, expected):
    user = _user(str(role.value), role=role)
    assert effective_project_access_mode(user) == expected
    assert uses_explicit_project_access(user) is (
        expected == ProjectAccessMode.explicit_projects
    )


def test_explicit_stored_project_access_mode_overrides_role_default():
    user = _user("staff-scoped", role=UserRole.editor)
    user.project_access_mode = ProjectAccessMode.explicit_projects

    assert effective_project_access_mode(user) == ProjectAccessMode.explicit_projects
    assert uses_explicit_project_access(user) is True


def test_scope_project_query_limits_explicit_user_to_grants(db_session):
    allowed = Project(id=101, prj_AddedBy="test", prj_ProjectTitle="Allowed")
    denied = Project(id=102, prj_AddedBy="test", prj_ProjectTitle="Denied")
    scoped = _user("viewer", role=UserRole.viewer)
    staff = _user("staff", role=UserRole.editor)
    db_session.add_all([allowed, denied, scoped, staff])
    db_session.commit()
    db_session.refresh(scoped)

    db_session.add(AuthUserProject(user_id=int(scoped.id), project_id=int(allowed.id)))
    db_session.commit()

    scoped_rows = scope_project_query(
        db_session.query(Project).order_by(Project.id.asc()),
        scoped,
    ).all()
    staff_rows = scope_project_query(
        db_session.query(Project).order_by(Project.id.asc()),
        staff,
    ).all()

    assert [int(project.id) for project in scoped_rows] == [101]
    assert [int(project.id) for project in staff_rows] == [101, 102]
    assert can_access_project(db_session, user=scoped, project_id=101) is True
    assert can_access_project(db_session, user=scoped, project_id=102) is False
    assert get_project_for_user(db_session, user=scoped, project_id=102) is None
