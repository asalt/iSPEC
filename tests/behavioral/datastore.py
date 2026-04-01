from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ispec.agent.connect import get_agent_session
from ispec.assistant.connect import get_assistant_session
from ispec.db.connect import get_session
from ispec.db.models import Project
from ispec.schedule.connect import get_schedule_session


@dataclass(frozen=True)
class BehavioralDatastore:
    root: Path
    core_db_path: Path
    assistant_db_path: Path
    schedule_db_path: Path
    agent_db_path: Path
    summary: dict[str, int]
    project_ids: tuple[int, ...]
    project_titles: dict[int, str]

    def env_overrides(self) -> dict[str, str]:
        return {
            "ISPEC_DB_PATH": str(self.core_db_path),
            "ISPEC_ASSISTANT_DB_PATH": str(self.assistant_db_path),
            "ISPEC_SCHEDULE_DB_PATH": str(self.schedule_db_path),
            "ISPEC_AGENT_DB_PATH": str(self.agent_db_path),
        }


def create_behavioral_datastore(
    root: str | Path,
) -> BehavioralDatastore:
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    core_db_path = root_path / "ispec-behavioral.db"
    assistant_db_path = root_path / "ispec-assistant-behavioral.db"
    schedule_db_path = root_path / "ispec-schedule-behavioral.db"
    agent_db_path = root_path / "ispec-agent-behavioral.db"

    summary = _seed_behavioral_core_db(core_db_path)

    with get_assistant_session(assistant_db_path):
        pass
    with get_schedule_session(schedule_db_path):
        pass
    with get_agent_session(agent_db_path):
        pass

    with get_session(core_db_path) as core_db:
        rows = core_db.query(Project.id, Project.prj_ProjectTitle).order_by(Project.id.asc()).all()
        project_ids = tuple(int(row[0]) for row in rows)
        project_titles = {
            int(row[0]): str(row[1] or "")
            for row in rows
        }

    return BehavioralDatastore(
        root=root_path,
        core_db_path=core_db_path,
        assistant_db_path=assistant_db_path,
        schedule_db_path=schedule_db_path,
        agent_db_path=agent_db_path,
        summary=dict(summary),
        project_ids=project_ids,
        project_titles=project_titles,
    )


def _seed_behavioral_core_db(core_db_path: Path) -> dict[str, int]:
    templates = [
        ("Behavioral Sandbox Alpha", "behavioral"),
        ("Behavioral Sandbox Beta", "behavioral"),
        ("Behavioral Sandbox Gamma", "behavioral"),
    ]
    with get_session(core_db_path) as core_db:
        existing = core_db.query(Project).count()
        if existing == 0:
            for title, added_by in templates:
                core_db.add(
                    Project(
                        prj_AddedBy=added_by,
                        prj_ProjectTitle=title,
                    )
                )
            core_db.flush()
        project_count = core_db.query(Project).count()
    return {"projects": int(project_count)}
