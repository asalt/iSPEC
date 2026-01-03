from __future__ import annotations

import hashlib
import os
from pathlib import PurePath

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ispec.api.security import require_access
from ispec.db.connect import get_session_dep
from ispec.db.models import AuthUser, Project, ProjectFile

router = APIRouter(prefix="/projects/{project_id}/files", tags=["ProjectFiles"])


def _max_upload_bytes() -> int:
    raw = os.getenv("ISPEC_PROJECT_FILE_MAX_BYTES") or "5242880"  # 5MB default
    try:
        return max(1, int(raw))
    except ValueError:
        return 5_242_880


def _safe_filename(name: str | None) -> str:
    if not name:
        return "upload"
    filename = PurePath(name).name.strip()
    return filename or "upload"


class ProjectFileOut(BaseModel):
    id: int
    project_id: int
    prjfile_FileName: str
    prjfile_ContentType: str | None = None
    prjfile_SizeBytes: int
    prjfile_Sha256: str | None = None
    prjfile_AddedBy: str | None = None

    model_config = {"from_attributes": True}


def _get_project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _get_project_file_or_404(db: Session, *, project_id: int, file_id: int) -> ProjectFile:
    row = (
        db.query(ProjectFile)
        .filter(ProjectFile.id == file_id)
        .filter(ProjectFile.project_id == project_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="file not found")
    return row


@router.get("", response_model=list[ProjectFileOut])
@router.get("/", response_model=list[ProjectFileOut])
def list_project_files(
    project_id: int,
    db: Session = Depends(get_session_dep),
    _user: AuthUser | None = Depends(require_access),
):
    _get_project_or_404(db, project_id)
    rows = (
        db.query(ProjectFile)
        .filter(ProjectFile.project_id == project_id)
        .order_by(ProjectFile.id.asc())
        .all()
    )
    return [ProjectFileOut.model_validate(row) for row in rows]


@router.post("", response_model=ProjectFileOut, status_code=201)
@router.post("/", response_model=ProjectFileOut, status_code=201)
async def upload_project_file(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_session_dep),
    user: AuthUser | None = Depends(require_access),
):
    _get_project_or_404(db, project_id)

    data = await file.read()
    max_bytes = _max_upload_bytes()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"file too large (max {max_bytes} bytes)")

    filename = _safe_filename(file.filename)
    content_type = (file.content_type or "").strip() or None
    sha256 = hashlib.sha256(data).hexdigest()
    added_by = user.username if user is not None else None

    record = ProjectFile(
        project_id=project_id,
        prjfile_FileName=filename,
        prjfile_ContentType=content_type,
        prjfile_SizeBytes=len(data),
        prjfile_Sha256=sha256,
        prjfile_AddedBy=added_by,
        prjfile_Data=data,
    )
    db.add(record)
    db.flush()
    return ProjectFileOut.model_validate(record)


@router.get("/{file_id}")
def download_project_file(
    project_id: int,
    file_id: int,
    db: Session = Depends(get_session_dep),
    _user: AuthUser | None = Depends(require_access),
):
    row = _get_project_file_or_404(db, project_id=project_id, file_id=file_id)
    filename = _safe_filename(row.prjfile_FileName)
    content_type = row.prjfile_ContentType or "application/octet-stream"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=row.prjfile_Data, media_type=content_type, headers=headers)


@router.delete("/{file_id}", status_code=204)
def delete_project_file(
    project_id: int,
    file_id: int,
    db: Session = Depends(get_session_dep),
    _user: AuthUser | None = Depends(require_access),
):
    row = _get_project_file_or_404(db, project_id=project_id, file_id=file_id)
    db.delete(row)
    return Response(status_code=204)

