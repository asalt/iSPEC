"""Shared constants for the local relay dispatcher."""

from __future__ import annotations

RELAY_AGENT_ID = "local-relay"
RELAY_SCHEMA_VERSION = 1

EVENT_RELAY_REQUEST_ENQUEUED = "local_relay_request_enqueued_v1"
EVENT_RELAY_RECEIPT = "local_relay_receipt_v1"

FAILURE_INVALID_REQUEST = "invalid_request"
FAILURE_UNSUPPORTED_KIND = "unsupported_kind"
FAILURE_MISSING_BODY = "missing_body"
FAILURE_MISSING_TARGET = "missing_target"
FAILURE_TARGET_NOT_ALLOWED = "target_not_allowed"
FAILURE_TARGET_BLOCKED = "target_blocked"
FAILURE_CONFIRMATION_REQUIRED = "confirmation_required"
FAILURE_LIVE_SEND_DISABLED = "live_send_disabled"
FAILURE_MISSING_TOKEN = "missing_token"
FAILURE_PROVIDER_ERROR = "provider_error"
FAILURE_TMUX_SEND_FAILED = "tmux_send_failed"
FAILURE_SOURCE_NOT_ALLOWED = "source_not_allowed"
FAILURE_ATTACHMENT_UNSUPPORTED = "attachment_upload_unsupported"
FAILURE_ATTACHMENT_MISSING = "attachment_missing"
FAILURE_ATTACHMENT_TOO_LARGE = "attachment_too_large"
FAILURE_ATTACHMENT_UPLOAD_FAILED = "attachment_upload_failed"

KIND_SLACK_MESSAGE = "slack_message"
KIND_TMUX_SEND = "tmux_send"
KIND_STATUS_RECORD = "status_record"
SUPPORTED_KINDS = {KIND_SLACK_MESSAGE, KIND_TMUX_SEND, KIND_STATUS_RECORD}
