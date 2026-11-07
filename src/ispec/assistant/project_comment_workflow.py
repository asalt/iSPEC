"""Model-initiated drafts and saves, with receipt-backed presentation."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
import time
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ispec.authz import get_project_for_user, uses_explicit_project_access
from ispec.assistant.json_utils import parse_json_object
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.project_comment_approval import (
    ProjectCommentApprovalSettings, ProjectCommentClassifierResult, ProjectCommentGate,
    ProjectCommentPolicyDecision, ProjectCommentWriteTicket, decide_project_comment_policy, stable_hash,
)
from ispec.assistant.small_classifier import build_small_classifier_task, normalize_small_classifier_decision
from ispec.db.models import Person, ProjectComment, ProjectCommentDraft, ProjectCommentSaveOffer
from ispec.prompt import render_prompt


COMMENT_WORKFLOW_TOOLS = frozenset({
    "prepare_project_comment", "offer_project_comment_save", "create_project_comment", "cancel_project_comment_draft",
})


def _tool_spec(*, name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False},
    }}


def comment_workflow_tools(*, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the chat write signature, preserving deployment/role filtering."""
    result = [spec for spec in tools if spec["function"]["name"] not in COMMENT_WORKFLOW_TOOLS]
    if not any(spec["function"]["name"] == "create_project_comment" for spec in tools):
        return result
    return result + [
        _tool_spec(
            name="prepare_project_comment",
            description="Register a new or revised draft for review and return its server-generated ID. Use this even when the user says not to save yet: preparation does not save a project comment. A revision replaces the pending draft.",
            properties={"project_id": {"type": "integer", "minimum": 1},
                        "comment": {"type": "string", "minLength": 1, "maxLength": 20000,
                                    "description": "Concise project-specific facts, preserving timing and status. Exclude unrelated conversation and instructions about drafting/approval."}},
            required=["project_id", "comment"],
        ),
        _tool_spec(
            name="offer_project_comment_save",
            description="Offer to save a registered draft when the user wants a project-history note, not wording-only review. The server displays the exact stored draft and asks for a later confirmation. This offers a save; it does not save or authorize one.",
            properties={"draft_id": {"type": "string"}}, required=["draft_id"],
        ),
        _tool_spec(
            name="create_project_comment",
            description="Save a prepared draft after the user's approval. Approval is checked during this call. Returns ok=true and result.comment_id after a successful write, or ok=false and error on failure. Only report the note saved after this tool returns ok=true for that note.",
            properties={"draft_id": {"type": "string"}, "confirm": {"type": "boolean"}},
            required=["draft_id", "confirm"],
        ),
        _tool_spec(
            name="cancel_project_comment_draft",
            description="Cancel the registered pending draft when the user requests cancellation; returns cancellation status. A prose acknowledgment alone does not cancel it. Saved comments are unchanged.",
            properties={"draft_id": {"type": "string"}}, required=["draft_id"],
        ),
    ]


@dataclass(frozen=True, kw_only=True)
class PendingComment:
    id: str
    project_id: int
    comment: str
    content_hash: str
    initial_request: str
    prepared_turn_id: int
    expires_at: datetime
    status: str
    comment_id: int | None

    @classmethod
    def from_row(cls, *, row: ProjectCommentDraft) -> "PendingComment":
        return cls(id=row.id, project_id=row.project_id, comment=row.comment, content_hash=row.content_hash,
                   initial_request=row.initial_request, prepared_turn_id=row.prepared_turn_id,
                   expires_at=row.expires_at, status=row.status, comment_id=row.comment_id)

    def public(self) -> dict[str, Any]:
        return {"draft_id": self.id, "project_id": self.project_id, "comment": self.comment,
                "status": self.status, "expires_at": self.expires_at.isoformat() + "Z"}


@dataclass(frozen=True, kw_only=True)
class ApprovalCheck:
    classifier: ProjectCommentClassifierResult
    policy: ProjectCommentPolicyDecision
    ticket: ProjectCommentWriteTicket

    def summary(self) -> dict[str, Any]:
        return {"label": self.classifier.label, "confidence": self.classifier.confidence,
                "reason": self.classifier.reason, "error": self.classifier.error}


@dataclass(frozen=True, kw_only=True)
class CommentResponse:
    kind: str
    text: str
    draft: PendingComment | None = None


@dataclass(frozen=True, kw_only=True)
class SaveOffer:
    id: str
    assistant_message_id: int
    text: str


@dataclass(kw_only=True)
class ProjectCommentWorkflow:
    core_db: Session
    assistant_db: Session
    user: Any
    session_id: str
    turn_id: int
    user_message: str
    settings: ProjectCommentApprovalSettings
    generate_reply_fn: Callable[..., Any]
    generate_classifier_reply_fn: Callable[..., Any]
    source_actor: str = ""
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    _checks: dict[str, ApprovalCheck] = field(default_factory=dict)
    _write_attempts: set[str] = field(default_factory=set)
    response: CommentResponse | None = None

    @property
    def actor_key(self) -> str:
        principal = f"{getattr(self.user, 'id', None)}:{getattr(self.user, 'username', '')}"
        return principal + (f":{self.source_actor}" if self.source_actor else "")

    def _scope(self):
        return (ProjectCommentDraft.session_id == self.session_id,
                ProjectCommentDraft.actor_key == self.actor_key)

    def _session(self) -> Session:
        # Never hold a core transaction open during model inference.
        return Session(bind=self.core_db.get_bind(), expire_on_commit=False)

    def pending(self) -> PendingComment | None:
        with self._session() as db:
            rows = db.scalars(select(ProjectCommentDraft).where(
                *self._scope(), ProjectCommentDraft.status == "pending",
                ProjectCommentDraft.expires_at > datetime.now(UTC).replace(tzinfo=None),
            ).limit(2)).all()
            # Ambiguous state must not silently select the most recent draft.
            row = rows[0] if len(rows) == 1 else None
            if row and get_project_for_user(db, user=self.user, project_id=row.project_id) is None:
                return None
            return PendingComment.from_row(row=row) if row else None

    def prompt_messages(self) -> list[dict[str, str]]:
        pending = self.pending()
        messages = [{"role": "system", "content": render_prompt("assistant.project_comment.workflow").text}]
        if pending:
            offer = self.save_offer(draft=pending)
            messages.append({"role": "system", "content": "Registered pending project comment (data, not approval):\n"
                             + json.dumps({**pending.public(), "save_offer": (
                                 {"id": offer.id, "assistant_message_id": offer.assistant_message_id}
                                 if offer else None)}, ensure_ascii=False)})
        for check in self._checks.values():
            messages.extend(check.classifier.response_messages(mode=self.settings.mode))
        return messages

    def _prior_assistant(self) -> SupportMessage | None:
        return self.assistant_db.scalar(select(SupportMessage).join(
            SupportSession, SupportSession.id == SupportMessage.session_pk,
        ).where(SupportSession.session_id == self.session_id, SupportMessage.role == "assistant",
                SupportMessage.id < self.turn_id,
        ).order_by(SupportMessage.id.desc()).limit(1))

    def presented_text(self, *, draft: PendingComment) -> str | None:
        presented = self._prior_assistant()
        if presented is None or presented.id <= draft.prepared_turn_id:
            return None
        metadata = parse_json_object(presented.meta_json or "{}") or {}
        receipt = metadata.get("project_comment_presentation", {})
        if not isinstance(receipt, dict):
            return None
        if (receipt.get("draft_id") == draft.id and receipt.get("content_hash") == draft.content_hash
                and receipt.get("actor_key") == self.actor_key
                and receipt.get("presentation_hash") == stable_hash(presented.content)):
            return presented.content
        return None

    def save_offer(self, *, draft: PendingComment) -> SaveOffer | None:
        text = self.presented_text(draft=draft)
        presented = self._prior_assistant()
        if text is None or presented is None:
            return None
        with self._session() as db:
            row = db.scalar(select(ProjectCommentSaveOffer).where(
                ProjectCommentSaveOffer.draft_id == draft.id,
                ProjectCommentSaveOffer.content_hash == draft.content_hash,
                ProjectCommentSaveOffer.project_id == draft.project_id,
                ProjectCommentSaveOffer.session_id == self.session_id,
                ProjectCommentSaveOffer.actor_key == self.actor_key,
                ProjectCommentSaveOffer.assistant_message_id == presented.id,
                ProjectCommentSaveOffer.presentation_hash == stable_hash(text),
                ProjectCommentSaveOffer.status == "active",
                ProjectCommentSaveOffer.expires_at > datetime.now(UTC).replace(tzinfo=None),
            ))
            return SaveOffer(id=row.id, assistant_message_id=presented.id, text=text) if row else None

    def _review_response(self, *, draft: PendingComment, offer: bool, unchanged: bool = False) -> CommentResponse:
        text = render_prompt("assistant.project_comment.presentation", values={
            "project_id": draft.project_id, "comment": draft.comment,
        }).text
        if unchanged:
            text = "The stored draft is unchanged; no revision or cancellation was registered this turn.\n\n" + text
        text += ("\n\nShall I save this exact note to the project history?" if offer
                 else "\n\nThis is a draft review, not a save offer.")
        return CommentResponse(kind="offered" if offer else "prepared", text=text, draft=draft)

    def delivery_response(self) -> CommentResponse | None:
        if self.response is not None:
            return self.response
        pending = self.pending()
        if pending:
            self.response = self._review_response(draft=pending, offer=False, unchanged=True)
        return self.response

    def record_delivery(self, *, message: SupportMessage) -> dict[str, Any] | None:
        """Bind only our exact response, after the assistant row has an ID.

        The assistant DB and core DB are separate. An orphaned core receipt is
        harmless: lookup also requires the committed matching assistant row.
        """
        response = self.response
        if response is None or response.draft is None or message.content != response.text:
            return None
        draft = response.draft
        receipt = {"draft_id": draft.id, "content_hash": draft.content_hash, "project_id": draft.project_id,
                   "actor_key": self.actor_key, "kind": response.kind,
                   "presentation_hash": stable_hash(message.content)}
        if response.kind == "offered":
            now = datetime.now(UTC).replace(tzinfo=None)
            with self._session() as db, db.begin():
                # Take a write lock before checking the draft, as saves/revisions
                # use this same database and can race with response persistence.
                valid = db.execute(update(ProjectCommentDraft).where(
                    *self._scope(), ProjectCommentDraft.id == draft.id,
                    ProjectCommentDraft.content_hash == draft.content_hash,
                    ProjectCommentDraft.status == "pending", ProjectCommentDraft.expires_at > now,
                ).values(status="pending")).rowcount
                if not valid or get_project_for_user(db, user=self.user, project_id=draft.project_id) is None:
                    return None
                self._invalidate_offers(db=db)
                offer = ProjectCommentSaveOffer(
                    id=uuid4().hex, draft_id=draft.id, content_hash=draft.content_hash,
                    project_id=draft.project_id, session_id=self.session_id, actor_key=self.actor_key,
                    assistant_message_id=message.id, presentation_hash=stable_hash(message.content),
                    created_at=now, expires_at=draft.expires_at, status="active",
                )
                db.add(offer)
                receipt["offer_id"] = offer.id
        return receipt

    def _invalidate_offers(self, *, db: Session) -> None:
        db.execute(update(ProjectCommentSaveOffer).where(
            ProjectCommentSaveOffer.session_id == self.session_id,
            ProjectCommentSaveOffer.actor_key == self.actor_key,
            ProjectCommentSaveOffer.status == "active",
        ).values(status="invalidated"))

    def approval_check(self, *, draft: PendingComment, presented: str) -> ApprovalCheck:
        if draft.id not in self._checks:
            check = self.check(draft=draft, presented=presented)
            self._checks[draft.id] = check
            offer = self.save_offer(draft=draft)
            self.evaluations.append({
                "schema_version": 2, "draft_id": draft.id, "project_id": draft.project_id,
                "mode": self.settings.mode, "classifier": check.classifier.to_dict(),
                "policy": check.policy.to_dict(), "write_ticket": check.ticket.to_dict(),
                "save_offer": ({"id": offer.id, "assistant_message_id": offer.assistant_message_id}
                               if offer else None),
            })
        return self._checks[draft.id]

    def _error(self, *, name: str, code: str, message: str) -> dict[str, Any]:
        # A failed follow-up call must not erase a successful write or the exact
        # newly prepared text that still needs to be presented for review.
        if self.response is None or self.response.kind == "error":
            self.response = CommentResponse(kind="error", text=f"No change was made by this request. {message}")
        return {"ok": False, "tool": name, "error_code": code, "error": message, "saved": False}

    def run(self, *, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.user is None or getattr(self.user, "can_write_project_comments", True) is not True:
            return self._error(name=name, code="permission_denied", message="Project-comment access is unavailable.")
        if name == "prepare_project_comment":
            return self.prepare(args=args)
        if set(args) - ({"draft_id", "confirm"} if name == "create_project_comment" else {"draft_id"}):
            return self._error(name=name, code="invalid_arguments", message="Use draft_id; changing the note requires a new preparation.")
        draft_id = args.get("draft_id")
        if not isinstance(draft_id, str) or not draft_id:
            return self._error(name=name, code="draft_required", message="Prepare and show a draft before requesting its save.")
        with self._session() as db:
            row = db.scalar(select(ProjectCommentDraft).where(*self._scope(), ProjectCommentDraft.id == draft_id))
            if row is None:
                return self._error(name=name, code="draft_not_found", message="That draft ID does not match a draft in this conversation. No other draft was substituted.")
            draft = PendingComment.from_row(row=row)
            if get_project_for_user(db, user=self.user, project_id=draft.project_id) is None:
                return self._error(name=name, code="project_unavailable", message="Project not found or not accessible.")
        if name == "cancel_project_comment_draft":
            with self._session() as db, db.begin():
                changed = db.execute(update(ProjectCommentDraft).where(
                    *self._scope(), ProjectCommentDraft.id == draft.id, ProjectCommentDraft.status == "pending",
                ).values(status="cancelled")).rowcount
                if changed:
                    self._invalidate_offers(db=db)
            if not changed:
                return self._error(name=name, code="draft_inactive", message="Only pending drafts can be cancelled; saved comments are unchanged.")
            self.response = CommentResponse(kind="cancelled", text=f"The pending draft for project {draft.project_id} was cancelled. Saved comments are unchanged.")
            return {"ok": True, "tool": name, "result": {"draft_id": draft.id, "cancelled": True}}
        if name == "offer_project_comment_save":
            if draft.status != "pending" or draft.expires_at <= datetime.now(UTC).replace(tzinfo=None):
                return self._error(name=name, code="draft_inactive", message="Only an active pending draft can be offered.")
            self.response = self._review_response(draft=draft, offer=True)
            return {"ok": True, "tool": name, "saved": False, "result": {
                **draft.public(), "save_offer": "awaiting_server_presentation",
            }}
        if args.get("confirm") is not True:
            return self._error(name=name, code="confirm_required", message="confirm=true is required for a save request.")
        if draft.status == "saved":
            return self._saved(draft=draft, comment_id=draft.comment_id, already_saved=True)
        if draft.id in self._write_attempts:
            return self._error(name=name, code="write_already_attempted", message="A write was already attempted this turn; no automatic retry.")
        if draft.status != "pending" or draft.expires_at <= datetime.now(UTC).replace(tzinfo=None):
            return self._error(name=name, code="draft_inactive", message="Draft expired, cancelled, or superseded; prepare a new draft.")
        offer = self.save_offer(draft=draft)
        if offer is None:
            self.response = self._review_response(draft=draft, offer=True)
            return {"ok": False, "tool": name, "saved": False, "error_code": "save_offer_required",
                    "error": "No delivered save offer. The server will show the exact stored draft and ask for a later confirmation."}
        check = self.approval_check(draft=draft, presented=offer.text)
        ticket = check.ticket
        if not ticket.issued or ticket.shadow_only or self.settings.mode != "own":
            return {**self._error(name=name, code="approval_not_granted", message="Not saved. Approval check did not permit this draft; clarify or revise it."),
                    "approval": check.summary()}
        if not ticket.expires_at or datetime.now(UTC) >= datetime.fromisoformat(ticket.expires_at):
            return self._error(name=name, code="approval_expired", message="Approval check expired; no write performed.")
        self._write_attempts.add(draft.id)
        with self._session() as db, db.begin():
            # Claim and insert share one transaction. A racing save/revision cannot
            # create a second comment or commit an obsolete version. Claim before
            # reading to avoid SQLite's deferred read-to-write upgrade race.
            changed = db.execute(update(ProjectCommentDraft).where(
                *self._scope(), ProjectCommentDraft.id == draft.id, ProjectCommentDraft.status == "pending",
                ProjectCommentDraft.content_hash == draft.content_hash,
                ProjectCommentDraft.expires_at > datetime.now(UTC).replace(tzinfo=None),
            ).values(status="saving")).rowcount
            if not changed:
                return self._error(name=name, code="draft_changed", message="Draft changed or was already processed; no additional write performed.")
            consumed = db.execute(update(ProjectCommentSaveOffer).where(
                ProjectCommentSaveOffer.id == offer.id, ProjectCommentSaveOffer.status == "active",
                ProjectCommentSaveOffer.expires_at > datetime.now(UTC).replace(tzinfo=None),
            ).values(status="consumed")).rowcount
            if not consumed:
                db.rollback()
                return self._error(name=name, code="save_offer_changed", message="Save offer changed or expired; not saved.")
            if get_project_for_user(db, user=self.user, project_id=draft.project_id) is None:
                db.rollback()
                return self._error(name=name, code="project_unavailable", message="Project access changed; not saved.")
            person = db.scalar(select(Person).where(Person.ppl_Name_First == "iSPEC", Person.ppl_Name_Last == "Assistant"))
            if person is None:
                person = Person(ppl_AddedBy=self.user.username, ppl_Name_First="iSPEC", ppl_Name_Last="Assistant")
                db.add(person)
                db.flush()
            comment = ProjectComment(project_id=draft.project_id, person_id=person.id,
                                     com_Comment=draft.comment, com_AddedBy=self.user.username,
                                     com_CommentType="client_note" if uses_explicit_project_access(self.user) else "assistant_note")
            db.add(comment)
            db.flush()
            comment_id = int(comment.id)
            db.execute(update(ProjectCommentDraft).where(ProjectCommentDraft.id == draft.id).values(
                status="saved", comment_id=comment_id))
        evidence = next(item for item in self.evaluations if item["draft_id"] == draft.id)
        evidence["write_outcome"] = {"status": "succeeded", "comment_id": comment_id}
        return {**self._saved(draft=draft, comment_id=comment_id, already_saved=False), "approval": check.summary()}

    def prepare(self, *, args: dict[str, Any]) -> dict[str, Any]:
        name = "prepare_project_comment"
        project_id, comment = args.get("project_id"), args.get("comment")
        if (set(args) != {"project_id", "comment"} or not isinstance(project_id, int)
                or isinstance(project_id, bool) or project_id <= 0
                or not isinstance(comment, str) or not comment.strip() or len(comment) > 20_000):
            return self._error(name=name, code="invalid_arguments", message="Provide a positive project_id and a non-empty comment of at most 20000 characters.")
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._session() as db, db.begin():
            if get_project_for_user(db, user=self.user, project_id=project_id) is None:
                return self._error(name=name, code="project_unavailable", message="Project not found or not accessible.")
            db.execute(update(ProjectCommentDraft).where(*self._scope(), ProjectCommentDraft.status == "pending").values(status="superseded"))
            self._invalidate_offers(db=db)
            row = ProjectCommentDraft(id=uuid4().hex, session_id=self.session_id, actor_key=self.actor_key,
                                      project_id=project_id, comment=comment, content_hash=stable_hash(comment),
                                      initial_request=self.user_message, prepared_turn_id=self.turn_id,
                                      created_at=now, expires_at=now + timedelta(hours=24), status="pending")
            db.add(row)
            db.flush()
            draft = PendingComment.from_row(row=row)
        self.response = self._review_response(draft=draft, offer=False)
        return {"ok": True, "tool": name, "saved": False, "result": draft.public()}

    def _saved(self, *, draft: PendingComment, comment_id: int | None, already_saved: bool) -> dict[str, Any]:
        verb = "was already saved" if already_saved else "was saved"
        self.response = CommentResponse(kind="saved", text=(
            f"The note for project {draft.project_id} {verb} (comment {comment_id}):\n\n{draft.comment}"
        ))
        return {"ok": True, "tool": "create_project_comment", "result": {
            "draft_id": draft.id, "project_id": draft.project_id, "comment_id": comment_id,
            "snippet": draft.comment[:240], "already_saved": already_saved,
            "links": {"project_ui": f"/project/{draft.project_id}"},
        }}

    def check(self, *, draft: PendingComment, presented: str) -> ApprovalCheck:
        offer = self.save_offer(draft=draft)
        eligible = bool(offer and offer.text == presented)
        gate = ProjectCommentGate(eligible=eligible, kind="pending_save_confirmation" if eligible else "none",
                                  reason="recorded_save_offer" if eligible else "save_offer_required",
                                  project_id=draft.project_id, pending_tool="create_project_comment", session_id=self.session_id,
                                  thread_key=self.session_id, current_turn_id=str(self.turn_id),
                                  prior_assistant_message_id=offer.assistant_message_id if offer else None,
                                  pending_draft_hash=draft.content_hash, source_message_hash=stable_hash(self.user_message))
        classifier = ProjectCommentClassifierResult.empty(timeout_ms=self.settings.timeout_ms)
        if eligible and self.settings.mode != "off":
            started = time.monotonic()
            try:
                prepared = build_small_classifier_task("staged_comment_approval", payload={
                    "initial_request": draft.initial_request, "presented_draft": presented,
                    "pending_comment": draft.public(), "latest_user_reply": self.user_message,
                })
                reply = self.generate_classifier_reply_fn(
                    base_generate_reply_fn=self.generate_reply_fn, messages=prepared.messages,
                    vllm_extra_body={"structured_outputs": {"json": prepared.schema},
                                     "max_tokens": prepared.task.max_tokens, "temperature": prepared.task.temperature},
                    timeout_seconds=max(.5, self.settings.timeout_ms / 1000),
                    observability_context=prepared.observability_context(extra={"surface": "support_chat", "stage": "staged_comment_backstop"}),
                )
                decision = normalize_small_classifier_decision("staged_comment_approval", parse_json_object(reply.content))
                latency = int((time.monotonic() - started) * 1000)
                valid = bool(reply.ok and decision and latency <= self.settings.timeout_ms)
                classifier = ProjectCommentClassifierResult(ran=True, ok=valid,
                    label=decision.label if valid else None, confidence=decision.confidence if valid else 0,
                    reason=decision.reason if valid else "", provider=reply.provider, model=reply.model,
                    latency_ms=latency, timeout_ms=self.settings.timeout_ms,
                    error=None if valid else reply.error or "classifier_timeout_or_invalid_result")
            except Exception as exc:
                classifier = ProjectCommentClassifierResult(ran=True, ok=False, label=None, confidence=0, reason="",
                    provider=None, model=None, latency_ms=int((time.monotonic() - started) * 1000),
                    timeout_ms=self.settings.timeout_ms, error=type(exc).__name__)
        policy, ticket = decide_project_comment_policy(
            settings=self.settings, gate=gate, trigger=None, classifier=classifier,
            candidate_comment_hash=draft.content_hash,
        )
        return ApprovalCheck(classifier=classifier, policy=policy, ticket=ticket)
