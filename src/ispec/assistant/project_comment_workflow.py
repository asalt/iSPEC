"""Model-initiated drafts with human-click confirmation cards."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
import time
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ispec.authz import get_project_for_user
from ispec.assistant.project_comment_reviews import request_review
from ispec.assistant.project_comment_cards import staff_review_channel
from ispec.assistant.json_utils import parse_json_object
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.project_comment_cards import (
    CardOrigin,
    ConfirmationCard,
    TextSaveContext,
    cards_for_conversation,
    new_card,
    save_from_text,
    staff_review_context,
)
from ispec.assistant.project_comment_approval import (
    ProjectCommentApprovalSettings,
    ProjectCommentAuthorization,
    ProjectCommentClassifierResult,
    ProjectCommentGate,
    ProjectCommentPolicyDecision,
    ProjectCommentWriteTicket,
    decide_project_comment_policy,
    stable_hash,
)
from ispec.assistant.small_classifier import (
    build_small_classifier_task,
    normalize_small_classifier_decision,
)
from ispec.db.models import (
    Project,
    ProjectCommentCard,
    ProjectCommentDraft,
    ProjectCommentSaveOffer,
)
from ispec.prompt import render_prompt


COMMENT_WORKFLOW_TOOLS = frozenset(
    {
        "prepare_project_comment",
        "request_staff_review",
        "create_project_comment",
        "show_project_comment_card",
        "cancel_project_comment_draft",
    }
)


def _tool_spec(
    *, name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


PREPARE_ONLY_TOOLS = {"prepare_project_comment", "show_project_comment_card"}


def comment_workflow_tools(
    *, tools: list[dict[str, Any]], prepare_only: bool = False
) -> list[dict[str, Any]]:
    """Replace the chat write signature, preserving deployment/role filtering."""
    result = [
        spec for spec in tools if spec["function"]["name"] not in COMMENT_WORKFLOW_TOOLS
    ]
    if not prepare_only and not any(
        spec["function"]["name"] == "create_project_comment" for spec in tools
    ):
        return result
    workflow = [
        _tool_spec(
            name="prepare_project_comment",
            description="Register a new or revised project-history draft. The server shows the exact stored text in a confirmation card; the person clicks Save or Cancel. Use prose for wording-only help. A revision requires a new preparation and supersedes the old card.",
            properties={
                "project_id": {"type": "integer", "minimum": 1},
                "comment": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 20000,
                    "description": "Concise project-specific facts, preserving timing and status. Exclude unrelated conversation and instructions about drafting/approval.",
                },
            },
            required=["project_id", "comment"],
        ),
        _tool_spec(
            name="create_project_comment",
            description="Save a registered, previously presented draft after the user's text approval. Use its returned draft_id and confirm=true. The classifier and ticket checks run during this call. Only report saved from a successful result with comment_id. An already saved draft returns its existing comment ID.",
            properties={"draft_id": {"type": "string"}, "confirm": {"type": "boolean"}},
            required=["draft_id", "confirm"],
        ),
        _tool_spec(
            name="show_project_comment_card",
            description="Show the registered draft's current card state. A person may Save, Cancel or Undo using its buttons. This tool only displays state.",
            properties={"draft_id": {"type": "string"}},
            required=["draft_id"],
        ),
        _tool_spec(
            name="cancel_project_comment_draft",
            description="Cancel the registered pending draft when the user requests cancellation; returns cancellation status. A prose acknowledgment alone does not cancel it. Saved comments are unchanged.",
            properties={"draft_id": {"type": "string"}},
            required=["draft_id"],
        ),
    ]

    if prepare_only:
        from ispec.assistant.tools import assistant_write_error

        if assistant_write_error("prepare_project_comment"):
            return result
        workflow = [
            spec for spec in workflow if spec["function"]["name"] in PREPARE_ONLY_TOOLS
        ]
        if staff_review_channel():
            workflow.append(
                _tool_spec(
                    name="request_staff_review",
                    description="When asked to share a note with staff on Slack, queue its existing draft_id for review. Prepare first if needed. Only a posted delivery result confirms posting.",
                    properties={"draft_id": {"type": "string"}},
                    required=["draft_id"],
                )
            )
    return result + workflow


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
        return cls(
            id=row.id,
            project_id=row.project_id,
            comment=row.comment,
            content_hash=row.content_hash,
            initial_request=row.initial_request,
            prepared_turn_id=row.prepared_turn_id,
            expires_at=row.expires_at,
            status=row.status,
            comment_id=row.comment_id,
        )

    def public(self) -> dict[str, Any]:
        return {
            "draft_id": self.id,
            "project_id": self.project_id,
            "comment": self.comment,
            "status": self.status,
            "expires_at": self.expires_at.isoformat() + "Z",
        }


@dataclass(frozen=True, kw_only=True)
class ApprovalCheck:
    gate: ProjectCommentGate
    classifier: ProjectCommentClassifierResult
    policy: ProjectCommentPolicyDecision
    ticket: ProjectCommentWriteTicket

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.classifier.label,
            "confidence": self.classifier.confidence,
            "reason": self.classifier.reason,
            "error": self.classifier.error,
        }


@dataclass(frozen=True, kw_only=True)
class DraftPresentation:
    """Receipt tying a stored draft to the assistant message that displayed it."""

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
    card_origin: CardOrigin | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    _checks: dict[str, ApprovalCheck] = field(default_factory=dict)
    _authorizations: dict[str, ProjectCommentAuthorization] = field(
        default_factory=dict
    )
    _prepared_draft: PendingComment | None = None

    @property
    def actor_key(self) -> str:
        principal = (
            f"{getattr(self.user, 'id', None)}:{getattr(self.user, 'username', '')}"
        )
        return principal + (f":{self.source_actor}" if self.source_actor else "")

    def _scope(self):
        return (
            ProjectCommentDraft.session_id == self.session_id,
            ProjectCommentDraft.actor_key == self.actor_key,
        )

    def _session(self) -> Session:
        # Never hold a core transaction open during model inference.
        return Session(bind=self.core_db.get_bind(), expire_on_commit=False)

    def _can_reference_project(self, db: Session, project_id: int) -> bool:
        """A comment agent can propose notes across projects without reading project records."""
        if getattr(self.user, "is_comment_agent", False):
            return (
                db.scalar(select(Project.id).where(Project.id == project_id))
                is not None
            )
        return (
            get_project_for_user(db, user=self.user, project_id=project_id) is not None
        )

    def pending(self) -> PendingComment | None:
        with self._session() as db:
            rows = db.scalars(
                select(ProjectCommentDraft)
                .where(
                    *self._scope(),
                    ProjectCommentDraft.status == "pending",
                    ProjectCommentDraft.expires_at
                    > datetime.now(UTC).replace(tzinfo=None),
                )
                .limit(2)
            ).all()
            # Ambiguous state must not silently select the most recent draft.
            row = rows[0] if len(rows) == 1 else None
            if row and not self._can_reference_project(db, row.project_id):
                return None
            return PendingComment.from_row(row=row) if row else None

    def prompt_messages(self) -> list[dict[str, str]]:
        pending = self.pending()
        messages = [
            {
                "role": "system",
                "content": (
                    "Prepare or revise the requested project note with prepare_project_comment. "
                    "This account can request a draft; only an authorized reviewer can save it."
                    if getattr(self.user, "is_comment_agent", False)
                    else render_prompt("assistant.project_comment.workflow").text
                ),
            }
        ]
        if pending:
            presentation = self.presentation(draft=pending)
            messages.append(
                {
                    "role": "system",
                    "content": "Registered pending project comment (data, not approval):\n"
                    + json.dumps(
                        {
                            **pending.public(),
                            "presentation": (
                                {
                                    "assistant_message_id": presentation.assistant_message_id
                                }
                                if presentation
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        for check in self._checks.values():
            messages.append(
                {
                    "role": "system",
                    "content": "Project approval classifier: "
                    + json.dumps(check.summary())
                    + ". Report the actual tool result and current draft state.",
                }
            )
        messages.extend(self.state_messages())
        return messages

    def state_messages(self) -> list[dict[str, str]]:
        """Supply authoritative conversation state even when comment tools are absent."""
        with self._session() as db:
            cards = cards_for_conversation(
                db=db, session_id=self.session_id, actor_key=self.actor_key
            )
            handoff = staff_review_context(db=db, origin=self.card_origin)
        messages = []
        if handoff:
            messages.append(
                {
                    "role": "system",
                    "content": "Staff review card (server data): "
                    + json.dumps(handoff),
                }
            )
        if cards:
            fields = {"draft_id", "project_id", "status", "comment_id", "can_undo"}
            states = [card.model_dump(include=fields) for card in cards]
            if getattr(self.user, "is_comment_agent", False):
                for state in states:
                    state["requested_by"] = self.user.username
            messages.append(
                {
                    "role": "system",
                    "content": "Current server confirmation-card state:\n"
                    + json.dumps(states),
                }
            )
        return messages

    def response_cards(self) -> list[ConfirmationCard]:
        """Return authoritative card states separately from model-authored text."""
        with self._session() as db:
            return cards_for_conversation(
                db=db, session_id=self.session_id, actor_key=self.actor_key
            )

    def _prior_assistant(self) -> SupportMessage | None:
        return self.assistant_db.scalar(
            select(SupportMessage)
            .join(
                SupportSession,
                SupportSession.id == SupportMessage.session_pk,
            )
            .where(
                SupportSession.session_id == self.session_id,
                SupportMessage.role == "assistant",
                SupportMessage.id < self.turn_id,
            )
            .order_by(SupportMessage.id.desc())
            .limit(1)
        )

    def presented_text(self, *, draft: PendingComment) -> str | None:
        presented = self._prior_assistant()
        if presented is None or presented.id <= draft.prepared_turn_id:
            return None
        metadata = parse_json_object(presented.meta_json or "{}") or {}
        receipt = metadata.get("project_comment_presentation", {})
        if not isinstance(receipt, dict):
            return None
        if receipt.get("status", "recorded") != "recorded":
            return None
        if (
            receipt.get("draft_id") == draft.id
            and receipt.get("content_hash") == draft.content_hash
            and receipt.get("actor_key") == self.actor_key
            and receipt.get("presentation_hash") == stable_hash(presented.content)
        ):
            return presented.content
        return None

    def presentation(self, *, draft: PendingComment) -> DraftPresentation | None:
        """Find the active receipt for the prior reply showing this exact draft."""
        text = self.presented_text(draft=draft)
        presented = self._prior_assistant()
        if text is None or presented is None:
            return None
        with self._session() as db:
            row = db.scalar(
                select(ProjectCommentSaveOffer).where(
                    ProjectCommentSaveOffer.draft_id == draft.id,
                    ProjectCommentSaveOffer.content_hash == draft.content_hash,
                    ProjectCommentSaveOffer.project_id == draft.project_id,
                    ProjectCommentSaveOffer.session_id == self.session_id,
                    ProjectCommentSaveOffer.actor_key == self.actor_key,
                    ProjectCommentSaveOffer.assistant_message_id == presented.id,
                    ProjectCommentSaveOffer.presentation_hash == stable_hash(text),
                    ProjectCommentSaveOffer.status == "active",
                    ProjectCommentSaveOffer.expires_at
                    > datetime.now(UTC).replace(tzinfo=None),
                )
            )
            return (
                DraftPresentation(
                    id=row.id, assistant_message_id=presented.id, text=text
                )
                if row
                else None
            )

    def record_delivery(self, *, message: SupportMessage) -> dict[str, Any] | None:
        """Bind the separate server card to its accompanying assistant reply.

        The assistant DB and core DB are separate. An orphaned core receipt is
        harmless: lookup also requires the committed matching assistant row.
        """
        draft = self._prepared_draft or self.pending()
        if draft is None:
            return None
        receipt = {
            "draft_id": draft.id,
            "content_hash": draft.content_hash,
            "project_id": draft.project_id,
            "actor_key": self.actor_key,
            "kind": "presented",
            "source": "server_confirmation_card",
            "status": "recorded",
            "presentation_hash": stable_hash(message.content),
        }
        session = self.assistant_db.get(SupportSession, message.session_pk)
        if (
            message.role != "assistant"
            or message.id is None
            or message.id <= self.turn_id
            or session is None
            or session.session_id != self.session_id
        ):
            return {**receipt, "status": "message_scope_mismatch"}
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._session() as db, db.begin():
            valid = db.execute(
                update(ProjectCommentDraft)
                .where(
                    *self._scope(),
                    ProjectCommentDraft.id == draft.id,
                    ProjectCommentDraft.content_hash == draft.content_hash,
                    ProjectCommentDraft.status == "pending",
                    ProjectCommentDraft.expires_at > now,
                )
                .values(status="pending")
            ).rowcount
            if not valid or not self._can_reference_project(db, draft.project_id):
                return {**receipt, "status": "draft_unavailable"}
            db.execute(
                update(ProjectCommentCard)
                .where(
                    ProjectCommentCard.draft_id == draft.id,
                    ProjectCommentCard.assistant_message_id.is_(None),
                )
                .values(assistant_message_id=message.id)
            )
            self._invalidate_presentations(db=db)
            # Preserve conversation evidence for text approval evaluation. Click
            # authorization uses the separate confirmation-card receipt.
            presentation = ProjectCommentSaveOffer(
                id=uuid4().hex,
                draft_id=draft.id,
                content_hash=draft.content_hash,
                project_id=draft.project_id,
                session_id=self.session_id,
                actor_key=self.actor_key,
                assistant_message_id=message.id,
                presentation_hash=stable_hash(message.content),
                created_at=now,
                expires_at=draft.expires_at,
                status="active",
            )
            db.add(presentation)
            receipt["receipt_id"] = presentation.id
        return receipt

    def _invalidate_presentations(self, *, db: Session) -> None:
        db.execute(
            update(ProjectCommentSaveOffer)
            .where(
                ProjectCommentSaveOffer.session_id == self.session_id,
                ProjectCommentSaveOffer.actor_key == self.actor_key,
                ProjectCommentSaveOffer.status == "active",
            )
            .values(status="invalidated")
        )

    def approval_check(self, *, draft: PendingComment, presented: str) -> ApprovalCheck:
        if draft.id not in self._checks:
            check = self.check(draft=draft, presented=presented)
            self._checks[draft.id] = check
            presentation = self.presentation(draft=draft)
            self.evaluations.append(
                {
                    "schema_version": 3,
                    "draft_id": draft.id,
                    "project_id": draft.project_id,
                    "mode": self.settings.mode,
                    "classifier": check.classifier.to_dict(),
                    "policy": check.policy.to_dict(),
                    "write_ticket": check.ticket.to_dict(),
                    "presentation": (
                        {
                            "id": presentation.id,
                            "assistant_message_id": presentation.assistant_message_id,
                        }
                        if presentation
                        else None
                    ),
                }
            )
        return self._checks[draft.id]

    def _error(self, *, name: str, code: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": name,
            "error_code": code,
            "error": message,
            "saved": False,
        }

    def run(self, *, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in COMMENT_WORKFLOW_TOOLS:
            return self._error(
                name=name,
                code="unknown_tool",
                message="Unknown project-comment workflow tool.",
            )
        if name == "request_staff_review":
            if set(args) != {"draft_id"} or not isinstance(args["draft_id"], str):
                return self._error(
                    name=name,
                    code="invalid_arguments",
                    message="Provide only the registered draft_id.",
                )
            with self._session() as db, db.begin():
                return request_review(
                    db=db,
                    user=self.user,
                    session_id=self.session_id,
                    actor_key=self.actor_key,
                    draft_id=args["draft_id"],
                )
        if self.user is None or (
            getattr(self.user, "can_write_project_comments", True) is not True
            and not (
                getattr(self.user, "is_comment_agent", False)
                and name in PREPARE_ONLY_TOOLS
            )
        ):
            return self._error(
                name=name,
                code="permission_denied",
                message="Project-comment access is unavailable.",
            )
        if name == "prepare_project_comment":
            return self.prepare(args=args)
        if set(args) - (
            {"draft_id", "confirm"}
            if name == "create_project_comment"
            else {"draft_id"}
        ):
            return self._error(
                name=name,
                code="invalid_arguments",
                message="Use draft_id; changing the note requires a new preparation.",
            )
        draft_id = args.get("draft_id")
        if not isinstance(draft_id, str) or not draft_id:
            return self._error(
                name=name,
                code="draft_required",
                message="Prepare and show a draft before requesting its save.",
            )
        with self._session() as db:
            row = db.scalar(
                select(ProjectCommentDraft).where(
                    *self._scope(), ProjectCommentDraft.id == draft_id
                )
            )
            if row is None:
                return self._error(
                    name=name,
                    code="draft_not_found",
                    message="That draft ID does not match a draft in this conversation. No other draft was substituted.",
                )
            draft = PendingComment.from_row(row=row)
            if not self._can_reference_project(db, draft.project_id):
                return self._error(
                    name=name,
                    code="project_unavailable",
                    message="Project not found or not accessible.",
                )
        if name == "cancel_project_comment_draft":
            with self._session() as db, db.begin():
                changed = db.execute(
                    update(ProjectCommentDraft)
                    .where(
                        *self._scope(),
                        ProjectCommentDraft.id == draft.id,
                        ProjectCommentDraft.status == "pending",
                    )
                    .values(status="cancelled")
                ).rowcount
                if changed:
                    self._invalidate_presentations(db=db)
            if not changed:
                return self._error(
                    name=name,
                    code="draft_inactive",
                    message="Only pending drafts can be cancelled; saved comments are unchanged.",
                )
            self._prepared_draft = None
            return {
                "ok": True,
                "tool": name,
                "result": {"draft_id": draft.id, "cancelled": True},
            }
        if name == "create_project_comment":
            if args.get("confirm") is not True:
                return self._error(
                    name=name,
                    code="confirm_required",
                    message="confirm=true is required for a text save request.",
                )
            return self.save(draft=draft)

        return {
            "ok": True,
            "tool": name,
            "saved": False,
            "result": draft.public(),
            "message": "Click Save on the confirmation card to save this draft.",
        }

    def save(self, *, draft: PendingComment) -> dict[str, Any]:
        """Classify a requested text save and execute it against current stored state."""
        authorization = None
        check = None
        if draft.status == "pending":
            presentation = self.presentation(draft=draft)
            if presentation is None:
                return self._error(
                    name="create_project_comment",
                    code="draft_not_presented",
                    message="This draft needs a committed presentation before text approval can save it.",
                )
            check = self.approval_check(draft=draft, presented=presentation.text)
            if draft.id not in self._authorizations:
                self._authorizations[draft.id] = ProjectCommentAuthorization(
                    gate=check.gate, ticket=check.ticket
                )
            authorization = self._authorizations[draft.id]

        outcome = save_from_text(
            core_db=self.core_db,
            assistant_db=self.assistant_db,
            draft_id=draft.id,
            content_hash=draft.content_hash,
            user=self.user,
            context=TextSaveContext(
                session_id=self.session_id,
                actor_key=self.actor_key,
                turn_id=self.turn_id,
                user_message=self.user_message,
                origin=self.card_origin or CardOrigin.for_request(user=self.user),
                authorization=authorization,
            ),
        )
        saved = bool(outcome.ok and outcome.card and outcome.card.status == "saved")
        if check is not None:
            evidence = next(
                item for item in self.evaluations if item["draft_id"] == draft.id
            )
            evidence["write_outcome"] = {
                "status": "succeeded" if saved else "failed",
                "comment_id": outcome.card.comment_id if outcome.card else None,
                "source": "text_approval",
                "already_completed": outcome.already_completed,
            }

        return {
            "ok": outcome.ok,
            "tool": "create_project_comment",
            "saved": saved,
            "error_code": outcome.error_code,
            "error": outcome.error,
            "result": {
                **(outcome.card.model_dump() if outcome.card else draft.public()),
                "already_saved": outcome.already_completed and saved,
            },
            "approval": check.summary() if check else None,
        }

    def prepare(self, *, args: dict[str, Any]) -> dict[str, Any]:
        name = "prepare_project_comment"
        project_id, comment = args.get("project_id"), args.get("comment")
        if (
            set(args) != {"project_id", "comment"}
            or not isinstance(project_id, int)
            or isinstance(project_id, bool)
            or project_id <= 0
            or not isinstance(comment, str)
            or not comment.strip()
            or len(comment) > 20_000
        ):
            return self._error(
                name=name,
                code="invalid_arguments",
                message="Provide project_id and comment.",
            )
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._session() as db, db.begin():
            if not self._can_reference_project(db, project_id):
                return self._error(
                    name=name,
                    code="project_unavailable",
                    message="Project not found or not accessible.",
                )
            db.execute(
                update(ProjectCommentDraft)
                .where(*self._scope(), ProjectCommentDraft.status == "pending")
                .values(status="superseded")
            )
            self._invalidate_presentations(db=db)
            row = ProjectCommentDraft(
                id=uuid4().hex,
                session_id=self.session_id,
                actor_key=self.actor_key,
                project_id=project_id,
                comment=comment,
                content_hash=stable_hash(comment),
                initial_request=self.user_message,
                prepared_turn_id=self.turn_id,
                created_at=now,
                expires_at=now + timedelta(hours=24),
                status="pending",
            )
            db.add(row)
            db.flush()
            db.add(
                new_card(
                    draft=row,
                    origin=self.card_origin or CardOrigin.for_request(user=self.user),
                )
            )
            db.flush()
            draft = PendingComment.from_row(row=row)
        self._prepared_draft = draft
        return {"ok": True, "tool": name, "saved": False, "result": draft.public()}

    def check(self, *, draft: PendingComment, presented: str) -> ApprovalCheck:
        presentation = self.presentation(draft=draft)
        eligible = bool(presentation and presentation.text == presented)
        gate = ProjectCommentGate(
            eligible=eligible,
            kind="pending_save_confirmation" if eligible else "none",
            reason="presented_pending_draft" if eligible else "draft_not_presented",
            project_id=draft.project_id,
            pending_tool="create_project_comment",
            session_id=self.session_id,
            thread_key=self.session_id,
            current_turn_id=str(self.turn_id),
            prior_assistant_message_id=(
                presentation.assistant_message_id if presentation else None
            ),
            pending_draft_hash=draft.content_hash,
            source_message_hash=stable_hash(self.user_message),
        )
        classifier = ProjectCommentClassifierResult.empty(
            timeout_ms=self.settings.timeout_ms
        )
        if eligible and self.settings.mode != "off":
            started = time.monotonic()
            try:
                prepared = build_small_classifier_task(
                    "staged_comment_approval",
                    payload={
                        "initial_request": draft.initial_request,
                        "presented_draft": presented,
                        "pending_comment": draft.public(),
                        "latest_user_reply": self.user_message,
                    },
                )
                reply = self.generate_classifier_reply_fn(
                    base_generate_reply_fn=self.generate_reply_fn,
                    messages=prepared.messages,
                    vllm_extra_body={
                        "structured_outputs": {"json": prepared.schema},
                        "max_tokens": prepared.task.max_tokens,
                        "temperature": prepared.task.temperature,
                    },
                    timeout_seconds=max(0.5, self.settings.timeout_ms / 1000),
                    observability_context=prepared.observability_context(
                        extra={
                            "surface": "support_chat",
                            "stage": "staged_comment_backstop",
                        }
                    ),
                )
                decision = normalize_small_classifier_decision(
                    "staged_comment_approval", parse_json_object(reply.content)
                )
                latency = int((time.monotonic() - started) * 1000)
                valid = bool(
                    reply.ok and decision and latency <= self.settings.timeout_ms
                )
                classifier = ProjectCommentClassifierResult(
                    ran=True,
                    ok=valid,
                    label=decision.label if valid else None,
                    confidence=decision.confidence if valid else 0,
                    reason=decision.reason if valid else "",
                    provider=reply.provider,
                    model=reply.model,
                    latency_ms=latency,
                    timeout_ms=self.settings.timeout_ms,
                    error=(
                        None
                        if valid
                        else reply.error or "classifier_timeout_or_invalid_result"
                    ),
                )
            except Exception as exc:
                classifier = ProjectCommentClassifierResult(
                    ran=True,
                    ok=False,
                    label=None,
                    confidence=0,
                    reason="",
                    provider=None,
                    model=None,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    timeout_ms=self.settings.timeout_ms,
                    error=type(exc).__name__,
                )

        policy, ticket = decide_project_comment_policy(
            settings=self.settings,
            gate=gate,
            trigger=None,
            classifier=classifier,
            candidate_comment_hash=draft.content_hash,
        )

        return ApprovalCheck(
            gate=gate, classifier=classifier, policy=policy, ticket=ticket
        )
