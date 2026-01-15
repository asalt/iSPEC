from __future__ import annotations

import re
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session, defer

from ispec.db.models import (
    AuthUser,
    AuthUserProject,
    Person,
    Project,
    ProjectComment,
    ProjectFile,
    UserRole,
)


_PROJECT_ID_RE = re.compile(r"\b(?:project|prj)\s*#?\s*(\d{1,9})\b", re.IGNORECASE)
_PRJ_RE = re.compile(r"\bPRJ\s*#?\s*(\d{1,9})\b", re.IGNORECASE)
_PERSON_ID_RE = re.compile(r"\b(?:person|ppl)\s*#?\s*(\d{1,9})\b", re.IGNORECASE)
_SINGULAR_PROJECT_RE = re.compile(r"\bproject\b", re.IGNORECASE)
_PLURAL_PROJECTS_RE = re.compile(r"\bprojects\b", re.IGNORECASE)
_THIS_PROJECT_RE = re.compile(r"\b(this|current|that|the)\s+project\b", re.IGNORECASE)
_FILE_HINT_RE = re.compile(
    r"\b(files?|plots?|pcaplots?|pca|biplot|cluster(?:plot)?s?|heatmap|pdfs?|images?)\b",
    re.IGNORECASE,
)
_PCA_HINT_RE = re.compile(r"\bpca\b|pc1|pc2|biplot|pcaplot", re.IGNORECASE)

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


def project_summary(
    db: Session,
    project: Project,
    *,
    include_comments: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
        "links": {
            "ui": f"/project/{project.id}",
            "api": f"/api/projects/{project.id}",
        },
    }

    if include_comments:
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

        payload["comments"] = {
            "count": int(comment_count or 0),
            "latest": latest_comment_payload,
        }

    return payload


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


def _file_display_path(filename: str | None) -> str | None:
    raw = (filename or "").strip()
    if not raw:
        return None
    return raw.replace("__", "/")


def _file_analysis_key(filename: str | None) -> str:
    display = _file_display_path(filename)
    if not display:
        return "Ungrouped"
    parts = [part for part in display.split("/") if part]
    return parts[0] if parts else "Ungrouped"


def _file_relative_path(filename: str | None) -> str | None:
    display = _file_display_path(filename)
    if not display:
        return None
    parts = [part for part in display.split("/") if part]
    if len(parts) <= 1:
        return display
    return "/".join(parts[1:])


def project_file_summary(file: ProjectFile) -> dict[str, Any]:
    filename_raw = getattr(file, "prjfile_FileName", None)
    display = _file_display_path(filename_raw)
    analysis = _file_analysis_key(filename_raw)
    relative = _file_relative_path(filename_raw)

    return {
        "id": int(file.id),
        "project_id": int(file.project_id),
        "analysis": analysis,
        "path": display,
        "name": relative,
        "filename_raw": filename_raw,
        "content_type": getattr(file, "prjfile_ContentType", None),
        "size_bytes": int(getattr(file, "prjfile_SizeBytes", 0) or 0),
        "sha256": getattr(file, "prjfile_Sha256", None),
        "added_by": getattr(file, "prjfile_AddedBy", None),
        "created": file.prjfile_CreationTS.isoformat() if file.prjfile_CreationTS else None,
        "modified": file.prjfile_ModificationTS.isoformat() if file.prjfile_ModificationTS else None,
        "links": {
            "download": f"/api/projects/{file.project_id}/files/{file.id}",
            "preview": f"/api/projects/{file.project_id}/files/{file.id}/preview",
        },
    }


def project_note_summary(comment: ProjectComment, *, max_chars: int = 400) -> dict[str, Any]:
    return {
        "id": int(comment.id),
        "type": comment.com_CommentType,
        "added_by": comment.com_AddedBy,
        "created": comment.com_CreationTS.isoformat() if comment.com_CreationTS else None,
        "comment": _truncate(comment.com_Comment, max_chars),
    }


def build_ispec_context(
    db: Session,
    *,
    message: str,
    state: dict[str, Any] | None = None,
    user: AuthUser | None = None,
    max_items: int = 20,
) -> dict[str, Any]:
    """Return a compact, LLM-friendly context payload from the iSPEC DB."""

    lowered = (message or "").lower()
    context: dict[str, Any] = {}
    state = state or {}

    is_client = bool(user is not None and user.role == UserRole.client)
    include_project_comments = not is_client

    def get_project(project_id: int) -> Project | None:
        if not is_client:
            return db.get(Project, project_id)
        return (
            db.query(Project)
            .join(AuthUserProject, AuthUserProject.project_id == Project.id)
            .filter(
                Project.id == project_id,
                AuthUserProject.user_id == user.id,  # type: ignore[union-attr]
            )
            .first()
        )

    if is_client:
        accessible_query = (
            db.query(Project)
            .join(AuthUserProject, AuthUserProject.project_id == Project.id)
            .filter(AuthUserProject.user_id == user.id)  # type: ignore[union-attr]
        )
        try:
            accessible_total = int(accessible_query.order_by(None).count())
        except Exception:
            accessible_total = int(accessible_query.count())
        accessible_rows = (
            accessible_query.order_by(Project.id.asc()).limit(max_items).all()
        )
        context["accessible_projects"] = [
            {"id": int(p.id), "title": p.prj_ProjectTitle, "status": p.prj_Status}
            for p in accessible_rows
        ]
        context["accessible_projects_total"] = accessible_total
        if accessible_total > len(accessible_rows):
            context["accessible_projects_truncated"] = True

    project_ids = extract_project_ids(message)
    focused_project_id = state.get("current_project_id")
    if not isinstance(focused_project_id, int) or focused_project_id < 0:
        focused_project_id = None

    resolved_projects: list[dict[str, Any]] = []
    missing_project_ids: list[int] = []
    for project_id in project_ids[:5]:
        project = get_project(project_id)
        if project is None:
            missing_project_ids.append(project_id)
            continue
        resolved_projects.append(
            project_summary(db, project, include_comments=include_project_comments)
        )
    if resolved_projects:
        context["projects"] = resolved_projects
    if missing_project_ids:
        context["missing_projects"] = missing_project_ids

    if not project_ids and focused_project_id is not None:
        wants_this_project = bool(_THIS_PROJECT_RE.search(message or ""))
        mentions_singular = bool(_SINGULAR_PROJECT_RE.search(message or "")) and not bool(
            _PLURAL_PROJECTS_RE.search(message or "")
        )
        if wants_this_project or mentions_singular:
            project = get_project(focused_project_id)
            if project is None:
                context["missing_current_project"] = focused_project_id
            else:
                context["current_project"] = project_summary(
                    db, project, include_comments=include_project_comments
                )

    if "current" in lowered and "project" in lowered:
        query = db.query(Project).filter(Project.prj_Current_FLAG.is_(True))
        if is_client:
            query = query.join(
                AuthUserProject, AuthUserProject.project_id == Project.id
            ).filter(AuthUserProject.user_id == user.id)  # type: ignore[union-attr]
        rows = query.order_by(Project.id.asc()).limit(max_items).all()
        context["current_projects"] = [
            {"id": int(p.id), "title": p.prj_ProjectTitle, "status": p.prj_Status}
            for p in rows
        ]

    if ("to be billed" in lowered) or ("to-bill" in lowered) or ("ready to bill" in lowered):
        query = db.query(Project).filter(Project.prj_Billing_ReadyToBill.is_(True))
        if is_client:
            query = query.join(
                AuthUserProject, AuthUserProject.project_id == Project.id
            ).filter(AuthUserProject.user_id == user.id)  # type: ignore[union-attr]
        rows = query.order_by(Project.id.asc()).limit(max_items).all()
        context["to_bill_projects"] = [
            {"id": int(p.id), "title": p.prj_ProjectTitle, "status": p.prj_Status}
            for p in rows
        ]

    for status in _PROJECT_STATUSES:
        if status in lowered and "project" in lowered:
            query = db.query(Project).filter(Project.prj_Status == status)
            if is_client:
                query = query.join(
                    AuthUserProject, AuthUserProject.project_id == Project.id
                ).filter(AuthUserProject.user_id == user.id)  # type: ignore[union-attr]
            rows = query.order_by(Project.id.asc()).limit(max_items).all()
            context.setdefault("projects_by_status", {})[status] = [
                {"id": int(p.id), "title": p.prj_ProjectTitle}
                for p in rows
            ]
            break

    wants_files = bool(_FILE_HINT_RE.search(message or ""))
    if wants_files and focused_project_id is not None:
        project = get_project(focused_project_id)
        if project is not None:
            file_query = (
                db.query(ProjectFile)
                .options(defer(ProjectFile.prjfile_Data))
                .filter(ProjectFile.project_id == focused_project_id)
            )
            try:
                file_total = int(file_query.order_by(None).count())
            except Exception:
                file_total = int(file_query.count())

            file_rows = file_query.order_by(ProjectFile.id.asc()).limit(50).all()
            file_items = [project_file_summary(row) for row in file_rows]

            by_analysis: dict[str, int] = {}
            for item in file_items:
                analysis = str(item.get("analysis") or "Ungrouped")
                by_analysis[analysis] = by_analysis.get(analysis, 0) + 1

            highlights = [
                item
                for item in file_items
                if _PCA_HINT_RE.search(str(item.get("path") or ""))
                or _PCA_HINT_RE.search(str(item.get("filename_raw") or ""))
            ][:10]

            context["current_project_files"] = {
                "project_id": int(project.id),
                "total": file_total,
                "count": len(file_items),
                "by_analysis": by_analysis,
                "highlights": highlights,
                "items": file_items[:20],
            }

    if is_client and focused_project_id is not None:
        project = get_project(focused_project_id)
        if project is not None:
            note_query = (
                db.query(ProjectComment)
                .filter(ProjectComment.project_id == focused_project_id)
                .filter(ProjectComment.com_CommentType == "client_note")
                .order_by(ProjectComment.com_CreationTS.desc(), ProjectComment.id.desc())
            )
            try:
                note_total = int(note_query.order_by(None).count())
            except Exception:
                note_total = int(note_query.count())

            note_limit = max(1, min(10, max_items))
            note_rows = note_query.limit(note_limit).all()
            context["current_project_notes"] = {
                "project_id": int(project.id),
                "total": note_total,
                "count": len(note_rows),
                "items": [project_note_summary(row) for row in note_rows],
            }
            if note_total > len(note_rows):
                context["current_project_notes_truncated"] = True

    if is_client:
        return context

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
