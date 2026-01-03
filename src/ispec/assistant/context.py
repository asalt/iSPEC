from __future__ import annotations

import re
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from ispec.db.models import Person, Project, ProjectComment


_PROJECT_ID_RE = re.compile(r"\b(?:project|prj)\s*#?\s*(\d{1,9})\b", re.IGNORECASE)
_PRJ_RE = re.compile(r"\bPRJ\s*#?\s*(\d{1,9})\b", re.IGNORECASE)
_PERSON_ID_RE = re.compile(r"\b(?:person|ppl)\s*#?\s*(\d{1,9})\b", re.IGNORECASE)

_PROJECT_STATUSES = [
    "inquiry",
    "consultation",
    "waiting",
    "processing",
    "analysis",
    "summary",
    "closed",
    "hibernate",
]


def _safe_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _truncate(value: str | None, limit: int = 280) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def extract_project_ids(text: str) -> list[int]:
    ids: set[int] = set()
    for regex in (_PROJECT_ID_RE, _PRJ_RE):
        for match in regex.finditer(text or ""):
            parsed = _safe_int(match.group(1))
            if parsed is not None:
                ids.add(parsed)
    return sorted(ids)


def extract_person_ids(text: str) -> list[int]:
    ids: set[int] = set()
    for match in _PERSON_ID_RE.finditer(text or ""):
        parsed = _safe_int(match.group(1))
        if parsed is not None:
            ids.add(parsed)
    return sorted(ids)


def project_summary(db: Session, project: Project) -> dict[str, Any]:
    comment_count = (
        db.query(func.count(ProjectComment.id))
        .filter(ProjectComment.project_id == project.id)
        .scalar()
    )
    latest_comment = (
        db.query(ProjectComment)
        .filter(ProjectComment.project_id == project.id)
        .order_by(ProjectComment.com_CreationTS.desc(), ProjectComment.id.desc())
        .first()
    )
    latest_comment_payload: dict[str, Any] | None = None
    if latest_comment is not None:
        latest_comment_payload = {
            "id": int(latest_comment.id),
            "type": latest_comment.com_CommentType,
            "added_by": latest_comment.com_AddedBy,
            "created": latest_comment.com_CreationTS.isoformat()
            if latest_comment.com_CreationTS
            else None,
            "comment": _truncate(latest_comment.com_Comment, 240),
        }

    return {
        "id": int(project.id),
        "title": project.prj_ProjectTitle,
        "status": project.prj_Status,
        "current": bool(project.prj_Current_FLAG),
        "to_be_billed": bool(project.prj_Billing_ReadyToBill),
        "pi": project.prj_PI,
        "lab_contact": project.prj_Project_LabContact,
        "created": project.prj_CreationTS.isoformat() if project.prj_CreationTS else None,
        "modified": project.prj_ModificationTS.isoformat()
        if project.prj_ModificationTS
        else None,
        "comments": {
            "count": int(comment_count or 0),
            "latest": latest_comment_payload,
        },
        "links": {
            "ui": f"/project/{project.id}",
            "api": f"/api/projects/{project.id}",
        },
    }


def person_summary(person: Person) -> dict[str, Any]:
    return {
        "id": int(person.id),
        "first": person.ppl_Name_First,
        "last": person.ppl_Name_Last,
        "email": person.ppl_Email,
        "institution": person.ppl_Institution,
        "status": person.ppl_Status,
        "links": {
            "api": f"/api/people/{person.id}",
        },
    }


def build_ispec_context(
    db: Session,
    *,
    message: str,
    state: dict[str, Any] | None = None,
    max_items: int = 20,
) -> dict[str, Any]:
    """Return a compact, LLM-friendly context payload from the iSPEC DB."""

    lowered = (message or "").lower()
    context: dict[str, Any] = {}
    state = state or {}

    project_ids = extract_project_ids(message)
    if not project_ids:
        focused = state.get("current_project_id")
        if isinstance(focused, int) and focused >= 0:
            project_ids = [focused]

    resolved_projects: list[dict[str, Any]] = []
    missing_project_ids: list[int] = []
    for project_id in project_ids[:5]:
        project = db.get(Project, project_id)
        if project is None:
            missing_project_ids.append(project_id)
            continue
        resolved_projects.append(project_summary(db, project))
    if resolved_projects:
        context["projects"] = resolved_projects
    if missing_project_ids:
        context["missing_projects"] = missing_project_ids

    if "current" in lowered and "project" in lowered:
        rows = (
            db.query(Project)
            .filter(Project.prj_Current_FLAG.is_(True))
            .order_by(Project.id.asc())
            .limit(max_items)
            .all()
        )
        context["current_projects"] = [
            {"id": int(p.id), "title": p.prj_ProjectTitle, "status": p.prj_Status}
            for p in rows
        ]

    if ("to be billed" in lowered) or ("to-bill" in lowered) or ("ready to bill" in lowered):
        rows = (
            db.query(Project)
            .filter(Project.prj_Billing_ReadyToBill.is_(True))
            .order_by(Project.id.asc())
            .limit(max_items)
            .all()
        )
        context["to_bill_projects"] = [
            {"id": int(p.id), "title": p.prj_ProjectTitle, "status": p.prj_Status}
            for p in rows
        ]

    for status in _PROJECT_STATUSES:
        if status in lowered and "project" in lowered:
            rows = (
                db.query(Project)
                .filter(Project.prj_Status == status)
                .order_by(Project.id.asc())
                .limit(max_items)
                .all()
            )
            context.setdefault("projects_by_status", {})[status] = [
                {"id": int(p.id), "title": p.prj_ProjectTitle}
                for p in rows
            ]
            break

    person_ids = extract_person_ids(message)
    if person_ids:
        people: list[dict[str, Any]] = []
        missing_people: list[int] = []
        for person_id in person_ids[:5]:
            person = db.get(Person, person_id)
            if person is None:
                missing_people.append(person_id)
                continue
            people.append(person_summary(person))
        if people:
            context["people"] = people
        if missing_people:
            context["missing_people"] = missing_people

    return context

