"""Local comment reversal with an explicit target/result boundary for future adapters."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from ispec.assistant.project_comment_approval import stable_hash
from ispec.db.models import ProjectComment, ProjectCommentDraft


@dataclass(frozen=True, kw_only=True)
class CommentUndoTarget:
    comment_id: int
    project_id: int
    content_hash: str
    created_at: str
    modified_at: str
    person_id: int
    added_by: str

    @classmethod
    def from_comment(cls, *, comment: ProjectComment) -> "CommentUndoTarget":
        return cls(
            comment_id=comment.id,
            project_id=comment.project_id,
            content_hash=stable_hash(comment.com_Comment),
            created_at=comment.com_CreationTS.replace(tzinfo=None).isoformat(),
            modified_at=comment.com_ModificationTS.replace(tzinfo=None).isoformat(),
            person_id=comment.person_id,
            added_by=comment.com_AddedBy,
        )


@dataclass(frozen=True, kw_only=True)
class CommentUndoResult:
    ok: bool
    backend: str = "local_sqlite"
    error_code: str | None = None
    error: str | None = None


def undo_local_comment(
    *, db: Session, draft: ProjectCommentDraft, target: CommentUndoTarget
) -> CommentUndoResult:
    """Remove the unchanged draft-created row inside the caller's transaction."""
    comment = db.get(ProjectComment, target.comment_id)
    if (
        comment is None
        or draft.comment_id != target.comment_id
        or target.project_id != draft.project_id
        or target.content_hash != draft.content_hash
        or comment.com_Comment != draft.comment
        or CommentUndoTarget.from_comment(comment=comment) != target
    ):
        return CommentUndoResult(
            ok=False,
            error_code="comment_changed",
            error="The saved comment changed or is missing. Review it before undoing.",
        )
    if comment.com_LegacyImportTS is not None:
        return CommentUndoResult(
            ok=False,
            error_code="legacy_undo_unavailable",
            error="This comment has legacy provenance; local-only Undo is unavailable.",
        )

    draft.comment_id = None
    db.flush([draft])
    db.delete(comment)
    db.flush()
    return CommentUndoResult(ok=True)
