"""Synthetic retrieval and access-control cases; no production records."""

import pytest

from ispec.assistant.tools import run_tool
from ispec.db.models import (
    AuthUser,
    AuthUserProject,
    Person,
    Project,
    ProjectComment,
    UserRole,
)


def search(db, query, *, tool="search_projects", user=None, **options):
    return run_tool(
        name=tool,
        args={"query": query, **options},
        core_db=db,
        user=user,
        schedule_db=None,
        api_schema=None,
    )


@pytest.fixture
def catalog(db_session):
    db_session.add_all(
        [
            Project(
                id=81,
                prj_AddedBy="fixture",
                prj_ProjectTitle="Winter orchard survey",
                prj_Project_LabContact="Carmen Fenwick",
                prj_PI="Avery Silva",
                prj_ProjectDescription="SARS-CoV-2 archive",
                prj_Current_FLAG=True,
            ),
            Project(
                id=82,
                prj_AddedBy="fixture",
                prj_ProjectTitle="Orchard archive",
                prj_PI="Avery Blake",
                prj_ProjectBackground="Silver canopy",
            ),
            Person(
                id=19,
                ppl_AddedBy="fixture",
                ppl_Name_First="Avery",
                ppl_Name_Last="Silva",
            ),
        ]
    )
    db_session.flush()
    db_session.add(
        ProjectComment(
            project_id=81, person_id=19, com_Comment="The zircon marker arrived."
        )
    )
    db_session.commit()
    return db_session


@pytest.mark.parametrize(
    ("text", "expected", "field"),
    [
        ("orchard Fenwick Silva", 81, "title"),
        ("sIlVa", 81, "pi"),
        ("Carmen", 81, "lab_contact"),
        ("COVID-19", 81, "description"),
        ("winter orchard", 81, "title"),
        ("wint", 81, "title"),
        ("Fenwck", 81, "lab_contact"),
        ("Fenwick", 81, "lab_contact"),
        ("Fenwikc", 81, "lab_contact"),
        ("canopy", 82, "background"),
        ("zircon", 81, "comment"),
        ("MSPC000081", 81, "id"),
    ],
)
def test_ranked_evidence(catalog, text, expected, field):
    first = search(catalog, text)["result"]["matches"][0]
    assert first["id"] == expected
    assert any(hit["field"] == field and hit["snippet"] for hit in first["evidence"])


@pytest.mark.parametrize(
    "text", ["%_", "the project number", "quasar", "999", "carmenxylophone"]
)
def test_no_match(catalog, text):
    assert search(catalog, text)["result"]["matches"] == []


def test_scope_applied_before_ranking_pagination_and_snippets(catalog):
    client = AuthUser(
        username="synthetic",
        role=UserRole.client,
        password_hash="test",
        password_salt="test",
    )
    catalog.add(client)
    catalog.flush()
    catalog.add(AuthUserProject(user_id=client.id, project_id=81))
    catalog.flush()

    def scoped(query, **options):
        return search(catalog, query, tool="my_projects", user=client, **options)[
            "result"
        ]

    assert not search(catalog, "canopy", user=client)["ok"]
    assert scoped("canopy")["total"] == scoped("82")["total"] == 0
    assert (
        scoped("zircon")["total"] == 0
    )  # Internal comment on an otherwise granted project.
    result = scoped("orchard", current_only=True, limit=1)
    assert result["total"] == 1
    assert result["projects"][0]["id"] == 81
    staff = search(catalog, "orchard", tool="my_projects", limit=1)["result"]
    assert staff["total"] == 2 and staff["count"] == 1
