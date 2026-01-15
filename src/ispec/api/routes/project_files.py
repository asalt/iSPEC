from __future__ import annotations

import hashlib
import mimetypes
import os
from datetime import datetime
from pathlib import PurePath

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ispec.api.security import get_project_or_404_for_user, require_access
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


def _infer_content_type(filename: str, provided: str | None) -> str | None:
    content_type = (provided or "").strip()
    normalized = content_type.lower()
    is_generic = normalized in {"application/octet-stream", "binary/octet-stream"}
    guessed, _ = mimetypes.guess_type(filename)

    if guessed:
        return guessed
    if content_type and not is_generic:
        return content_type
    return None


def _normalize_content_type(value: str | None) -> str:
    if not value:
        return ""
    return value.split(";", 1)[0].strip().lower()


_INLINE_ALLOWED_TYPES = {"application/pdf"}
_INLINE_BLOCKED_IMAGE_TYPES = {"image/svg+xml"}


def _is_inline_allowed(content_type: str | None) -> bool:
    normalized = _normalize_content_type(content_type)
    if not normalized:
        return False
    if normalized.startswith("image/"):
        return normalized not in _INLINE_BLOCKED_IMAGE_TYPES
    return normalized in _INLINE_ALLOWED_TYPES


async def _read_upload_file(file: UploadFile, *, max_bytes: int) -> tuple[bytes, str, int]:
    sha256 = hashlib.sha256()
    parts: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"file too large (max {max_bytes} bytes)")
        sha256.update(chunk)
        parts.append(chunk)

    data = b"".join(parts)
    return data, sha256.hexdigest(), total


class ProjectFileOut(BaseModel):
    id: int
    project_id: int
    prjfile_FileName: str
    prjfile_ContentType: str | None = None
    prjfile_SizeBytes: int
    prjfile_Sha256: str | None = None
    prjfile_AddedBy: str | None = None
    prjfile_CreationTS: datetime | None = None
    prjfile_ModificationTS: datetime | None = None

    model_config = {"from_attributes": True}


def _get_project_or_404(db: Session, *, project_id: int, user: AuthUser | None) -> Project:
    return get_project_or_404_for_user(db, project_id=project_id, user=user)


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
    user: AuthUser | None = Depends(require_access),
):
    _get_project_or_404(db, project_id=project_id, user=user)
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
    _get_project_or_404(db, project_id=project_id, user=user)

    max_bytes = _max_upload_bytes()
    data, sha256, size_bytes = await _read_upload_file(file, max_bytes=max_bytes)

    filename = _safe_filename(file.filename)
    content_type = _infer_content_type(filename, file.content_type)
    added_by = user.username if user is not None else None

    record = ProjectFile(
        project_id=project_id,
        prjfile_FileName=filename,
        prjfile_ContentType=content_type,
        prjfile_SizeBytes=size_bytes,
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
    user: AuthUser | None = Depends(require_access),
):
    _get_project_or_404(db, project_id=project_id, user=user)
    row = _get_project_file_or_404(db, project_id=project_id, file_id=file_id)
    filename = _safe_filename(row.prjfile_FileName)
    content_type = (
        _infer_content_type(filename, row.prjfile_ContentType)
        or row.prjfile_ContentType
        or "application/octet-stream"
    )
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    if row.prjfile_Sha256:
        headers["ETag"] = f"\"{row.prjfile_Sha256}\""
    return Response(content=row.prjfile_Data, media_type=content_type, headers=headers)


@router.get("/{file_id}/preview")
def preview_project_file(
    project_id: int,
    file_id: int,
    db: Session = Depends(get_session_dep),
    user: AuthUser | None = Depends(require_access),
):
    _get_project_or_404(db, project_id=project_id, user=user)
    row = _get_project_file_or_404(db, project_id=project_id, file_id=file_id)
    filename = _safe_filename(row.prjfile_FileName)
    content_type = _infer_content_type(filename, row.prjfile_ContentType) or row.prjfile_ContentType
    if not _is_inline_allowed(content_type):
        raise HTTPException(
            status_code=415,
            detail="Inline preview is only supported for images and PDFs.",
        )

    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    if row.prjfile_Sha256:
        headers["ETag"] = f"\"{row.prjfile_Sha256}\""
    return Response(
        content=row.prjfile_Data,
        media_type=content_type or "application/octet-stream",
        headers=headers,
    )


@router.delete("/{file_id}", status_code=204)
def delete_project_file(
    project_id: int,
    file_id: int,
    db: Session = Depends(get_session_dep),
    user: AuthUser | None = Depends(require_access),
):
    _get_project_or_404(db, project_id=project_id, user=user)
    row = _get_project_file_or_404(db, project_id=project_id, file_id=file_id)
    db.delete(row)
    return Response(status_code=204)
