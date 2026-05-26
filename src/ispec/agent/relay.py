"""Public facade for local relay inbox helpers.

The relay implementation is split by concern so config probing, request
normalization, persistence, dispatch policy, Slack, and tmux delivery stay
reviewable. This module keeps the original import surface stable for API,
CLI, supervisor, and tests.
"""

from __future__ import annotations

from ispec.agent.relay_config import (
    CanonicalEnv,
    load_canonical_env,
    relay_config_probe,
    resolve_slack_destination,
)
from ispec.agent.relay_constants import (
    EVENT_RELAY_RECEIPT,
    EVENT_RELAY_REQUEST_ENQUEUED,
    FAILURE_ATTACHMENT_MISSING,
    FAILURE_ATTACHMENT_TOO_LARGE,
    FAILURE_ATTACHMENT_UNSUPPORTED,
    FAILURE_ATTACHMENT_UPLOAD_FAILED,
    FAILURE_CONFIRMATION_REQUIRED,
    FAILURE_INVALID_REQUEST,
    FAILURE_LIVE_SEND_DISABLED,
    FAILURE_MISSING_BODY,
    FAILURE_MISSING_TARGET,
    FAILURE_MISSING_TOKEN,
    FAILURE_PROVIDER_ERROR,
    FAILURE_SOURCE_NOT_ALLOWED,
    FAILURE_TARGET_BLOCKED,
    FAILURE_TARGET_NOT_ALLOWED,
    FAILURE_TMUX_SEND_FAILED,
    FAILURE_UNSUPPORTED_KIND,
    KIND_SLACK_MESSAGE,
    KIND_STATUS_RECORD,
    KIND_TMUX_SEND,
    RELAY_AGENT_ID,
    RELAY_SCHEMA_VERSION,
    SUPPORTED_KINDS,
)
from ispec.agent.relay_dispatcher import dispatch_relay_request
from ispec.agent.relay_normalize import normalize_relay_request
from ispec.agent.relay_store import enqueue_relay_request

__all__ = [
    "CanonicalEnv",
    "EVENT_RELAY_RECEIPT",
    "EVENT_RELAY_REQUEST_ENQUEUED",
    "FAILURE_ATTACHMENT_MISSING",
    "FAILURE_ATTACHMENT_TOO_LARGE",
    "FAILURE_ATTACHMENT_UNSUPPORTED",
    "FAILURE_ATTACHMENT_UPLOAD_FAILED",
    "FAILURE_CONFIRMATION_REQUIRED",
    "FAILURE_INVALID_REQUEST",
    "FAILURE_LIVE_SEND_DISABLED",
    "FAILURE_MISSING_BODY",
    "FAILURE_MISSING_TARGET",
    "FAILURE_MISSING_TOKEN",
    "FAILURE_PROVIDER_ERROR",
    "FAILURE_SOURCE_NOT_ALLOWED",
    "FAILURE_TARGET_BLOCKED",
    "FAILURE_TARGET_NOT_ALLOWED",
    "FAILURE_TMUX_SEND_FAILED",
    "FAILURE_UNSUPPORTED_KIND",
    "KIND_SLACK_MESSAGE",
    "KIND_STATUS_RECORD",
    "KIND_TMUX_SEND",
    "RELAY_AGENT_ID",
    "RELAY_SCHEMA_VERSION",
    "SUPPORTED_KINDS",
    "dispatch_relay_request",
    "enqueue_relay_request",
    "load_canonical_env",
    "normalize_relay_request",
    "relay_config_probe",
    "resolve_slack_destination",
]
