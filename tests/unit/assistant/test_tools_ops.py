from __future__ import annotations

from datetime import UTC, datetime

from ispec.assistant.tools import run_tool
from ispec.db.models import Experiment, ExperimentRun, Person, Project, ProjectComment


def test_project_status_counts(db_session):
    db_session.add_all(
        [
            Project(
                prj_AddedBy="test",
                prj_ProjectTitle="Current Inquiry",
                prj_Status="inquiry",
                prj_Current_FLAG=True,
            ),
            Project(
                prj_AddedBy="test",
                prj_ProjectTitle="Current Closed",
                prj_Status="closed",
                prj_Current_FLAG=True,
            ),
            Project(
                prj_AddedBy="test",
                prj_ProjectTitle="Not Current",
                prj_Status="closed",
                prj_Current_FLAG=False,
            ),
        ]
    )
    db_session.commit()

    payload = run_tool(
        name="project_status_counts",
        args={},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    items = payload["result"]["items"]
    assert any(item["status"] == "closed" and item["count"] == 2 for item in items)
    assert any(item["status"] == "inquiry" and item["count"] == 1 for item in items)

    payload_current = run_tool(
        name="project_status_counts",
        args={"current_only": True},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload_current["ok"] is True
    items_current = payload_current["result"]["items"]
    assert any(item["status"] == "closed" and item["count"] == 1 for item in items_current)
    assert any(item["status"] == "inquiry" and item["count"] == 1 for item in items_current)
    assert payload_current["result"]["total"] == 2


def test_latest_projects_sorting(db_session):
    t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    t2 = datetime(2024, 2, 1, 12, 0, 0, tzinfo=UTC)
    t3 = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)
    t4 = datetime(2024, 4, 1, 12, 0, 0, tzinfo=UTC)

    p1 = Project(
        prj_AddedBy="test",
        prj_ProjectTitle="Older Created",
        prj_Status="inquiry",
        prj_Current_FLAG=True,
        prj_CreationTS=t1,
        prj_ModificationTS=t4,
    )
    p2 = Project(
        prj_AddedBy="test",
        prj_ProjectTitle="Newer Created",
        prj_Status="closed",
        prj_Current_FLAG=False,
        prj_CreationTS=t3,
        prj_ModificationTS=t2,
    )
    db_session.add_all([p1, p2])
    db_session.commit()

    created = run_tool(
        name="latest_projects",
        args={"sort": "created", "limit": 2},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert created["ok"] is True
    assert [row["title"] for row in created["result"]["projects"]] == [
        "Newer Created",
        "Older Created",
    ]

    modified = run_tool(
        name="latest_projects",
        args={"sort": "modified", "limit": 2},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert modified["ok"] is True
    assert [row["title"] for row in modified["result"]["projects"]] == [
        "Older Created",
        "Newer Created",
    ]


def test_latest_project_comments(db_session):
    project = Project(prj_AddedBy="test", prj_ProjectTitle="Comment Project")
    person = Person(ppl_AddedBy="test", ppl_Name_First="A", ppl_Name_Last="B")
    db_session.add_all([project, person])
    db_session.commit()
    db_session.refresh(project)
    db_session.refresh(person)

    c1 = ProjectComment(
        project_id=project.id,
        person_id=person.id,
        com_Comment="first",
        com_CreationTS=datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC),
    )
    long_text = "x" * 500
    c2 = ProjectComment(
        project_id=project.id,
        person_id=person.id,
        com_Comment=long_text,
        com_CreationTS=datetime(2024, 2, 1, 1, 0, 0, tzinfo=UTC),
    )
    db_session.add_all([c1, c2])
    db_session.commit()

    payload = run_tool(
        name="latest_project_comments",
        args={"limit": 1, "project_id": project.id},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    assert payload["result"]["count"] == 1
    comment = payload["result"]["comments"][0]
    assert comment["project_id"] == project.id
    assert comment["project_title"] == "Comment Project"
    assert comment["comment"].endswith("…")


def test_experiments_for_project(db_session):
    project = Project(prj_AddedBy="test", prj_ProjectTitle="Exp Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    exp1 = Experiment(record_no="EXP-1", exp_Name="First", project_id=project.id)
    exp2 = Experiment(record_no="EXP-2", exp_Name="Second", project_id=project.id)
    exp_other = Experiment(record_no="EXP-X", exp_Name="Other", project_id=None)
    db_session.add_all([exp1, exp2, exp_other])
    db_session.commit()

    payload = run_tool(
        name="experiments_for_project",
        args={"project_id": project.id, "limit": 10},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    assert payload["result"]["project_id"] == project.id
    assert payload["result"]["count"] == 2
    names = [row["name"] for row in payload["result"]["experiments"]]
    assert set(names) == {"First", "Second"}


def test_project_counts_snapshot(db_session):
    db_session.add_all(
        [
            Project(
                prj_AddedBy="test",
                prj_ProjectTitle="A",
                prj_Status="inquiry",
                prj_Current_FLAG=True,
                prj_ProjectPriceLevel="internal",
                prj_Billing_ReadyToBill=True,
            ),
            Project(
                prj_AddedBy="test",
                prj_ProjectTitle="B",
                prj_Status="closed",
                prj_Current_FLAG=False,
                prj_ProjectPriceLevel="external",
                prj_Billing_ReadyToBill=False,
            ),
        ]
    )
    db_session.commit()

    payload = run_tool(
        name="project_counts_snapshot",
        args={"max_categories": 10},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    projects = payload["result"]["projects"]
    assert projects["total"] == 2
    assert projects["current"] == 1
    assert projects["billing_ready_to_bill"]["total"] == 1
    assert projects["billing_ready_to_bill"]["current"] == 1
    assert any(item["status"] == "inquiry" and item["count"] == 1 for item in projects["status_counts_total"])
    assert any(item["category"] == "internal" for item in projects["billing_categories"]["items"])


def test_latest_activity_includes_multiple_kinds(db_session):
    t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    t2 = datetime(2024, 2, 1, 12, 0, 0, tzinfo=UTC)
    t3 = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)

    project = Project(
        prj_AddedBy="test",
        prj_ProjectTitle="Activity Project",
        prj_Current_FLAG=True,
        prj_CreationTS=t1,
        prj_ModificationTS=t1,
    )
    person = Person(ppl_AddedBy="test", ppl_Name_First="A", ppl_Name_Last="B")
    db_session.add_all([project, person])
    db_session.commit()
    db_session.refresh(project)
    db_session.refresh(person)

    comment = ProjectComment(
        project_id=project.id,
        person_id=person.id,
        com_Comment="hello",
        com_CreationTS=t3,
    )
    experiment = Experiment(
        record_no="EXP-1",
        exp_Name="Exp",
        project_id=project.id,
        Experiment_CreationTS=t2,
        Experiment_ModificationTS=t2,
    )
    db_session.add_all([comment, experiment])
    db_session.commit()
    db_session.refresh(experiment)

    run = ExperimentRun(
        experiment_id=experiment.id,
        ExperimentRun_CreationTS=t2,
        ExperimentRun_ModificationTS=t2,
    )
    db_session.add(run)
    db_session.commit()

    payload = run_tool(
        name="latest_activity",
        args={"limit": 10},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    events = payload["result"]["events"]
    kinds = {row["kind"] for row in events}
    assert {"project", "project_comment", "experiment", "experiment_run"} <= kinds
    assert events[0]["kind"] == "project_comment"


def test_billing_category_counts(db_session):
    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="A", prj_ProjectPriceLevel="internal"),
            Project(prj_AddedBy="test", prj_ProjectTitle="B", prj_ProjectPriceLevel="internal"),
            Project(prj_AddedBy="test", prj_ProjectTitle="C", prj_ProjectPriceLevel="external"),
        ]
    )
    db_session.commit()

    payload = run_tool(
        name="billing_category_counts",
        args={"limit": 10},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    items = payload["result"]["items"]
    assert items[0]["category"] == "internal"
    assert items[0]["count"] == 2
    assert any(item["category"] == "external" and item["count"] == 1 for item in items)


def test_db_file_stats_reports_core_db_file_size(db_session):
    db_session.add(Project(prj_AddedBy="test", prj_ProjectTitle="A"))
    db_session.commit()

    payload = run_tool(
        name="db_file_stats",
        args={},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    core = payload["result"]["core_db"]
    assert core["exists"] is True
    assert isinstance(core["size_bytes"], int)
    assert core["size_bytes"] >= 0
    assert isinstance(core["size_human"], str)
