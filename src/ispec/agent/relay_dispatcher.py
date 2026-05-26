"""Dispatch policy for local relay requests."""

from __future__ import annotations

from typing import Any, Callable

from ispec.agent.relay_config import (
    load_canonical_env,
    relay_config_probe,
    relay_live_enabled,
    resolve_slack_destination,
    source_policy,
)
from ispec.agent.relay_constants import (
    FAILURE_CONFIRMATION_REQUIRED,
    FAILURE_INVALID_REQUEST,
    FAILURE_LIVE_SEND_DISABLED,
    FAILURE_MISSING_BODY,
    FAILURE_MISSING_TARGET,
    FAILURE_SOURCE_NOT_ALLOWED,
    FAILURE_TARGET_NOT_ALLOWED,
    FAILURE_UNSUPPORTED_KIND,
    KIND_SLACK_MESSAGE,
    KIND_STATUS_RECORD,
    KIND_TMUX_SEND,
    RELAY_SCHEMA_VERSION,
)
from ispec.agent.relay_normalize import normalize_relay_request
from ispec.agent.relay_slack import execute_slack_send, execute_slack_uploads
from ispec.agent.relay_store import record_relay_receipt
from ispec.agent.relay_tmux import execute_tmux_send, validate_tmux_target


def _finish(command_id: int | None, request: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    record_relay_receipt(command_id=command_id, request=request, receipt=receipt)
    return receipt


def dispatch_relay_request(
    payload: dict[str, Any],
    *,
    command_id: int | None = None,
    slack_post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    request, normalize_error = normalize_relay_request(payload)
    if request is None:
        receipt = {
            "ok": False,
            "schema_version": RELAY_SCHEMA_VERSION,
            "delivery_outcome": "failed",
            "sent": False,
            "error_type": normalize_error or FAILURE_INVALID_REQUEST,
        }
        return _finish(command_id, {"request_id": None}, receipt)

    env = load_canonical_env()
    probe = relay_config_probe(
        target_alias=(request.get("target") or {}).get("alias"),
        message_type=request.get("message_type"),
    )
    kind = str(request.get("kind") or "")
    mode = str(request.get("mode") or "stage")
    body = str(request.get("body") or "").strip()

    receipt: dict[str, Any] = {
        "ok": True,
        "schema_version": RELAY_SCHEMA_VERSION,
        "request_id": request.get("request_id"),
        "kind": kind,
        "mode": mode,
        "delivery_outcome": "staged",
        "sent": False,
        "config_probe": probe,
        "policy": {
            "live_send_enabled": relay_live_enabled(env),
            "confirm": bool(request.get("confirm") is True),
        },
        "target": request.get("target"),
        "provenance": request.get("provenance") if isinstance(request.get("provenance"), dict) else {},
        "metadata": request.get("metadata") if isinstance(request.get("metadata"), dict) else {},
    }

    source_allowed, source_detail = source_policy(request, env=env)
    receipt["policy"]["source"] = source_detail
    if not source_allowed:
        receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_SOURCE_NOT_ALLOWED)
        return _finish(command_id, request, receipt)

    if kind in {KIND_SLACK_MESSAGE, KIND_TMUX_SEND} and not body:
        receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_MISSING_BODY)
        return _finish(command_id, request, receipt)

    if kind == KIND_STATUS_RECORD:
        receipt["delivery_outcome"] = "recorded"
        return _finish(command_id, request, receipt)

    if kind == KIND_SLACK_MESSAGE:
        target_alias = (request.get("target") or {}).get("alias")
        destination, destination_error = resolve_slack_destination(
            env,
            str(target_alias or ""),
            message_type=request.get("message_type"),
        )
        if destination_error or destination is None:
            receipt.update(ok=False, delivery_outcome="failed", error_type=destination_error or FAILURE_TARGET_NOT_ALLOWED)
            return _finish(command_id, request, receipt)
        receipt["resolved_target"] = destination
        if mode != "send":
            return _finish(command_id, request, receipt)
        if request.get("confirm") is not True:
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_CONFIRMATION_REQUIRED)
            return _finish(command_id, request, receipt)
        if not relay_live_enabled(env):
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_LIVE_SEND_DISABLED)
            return _finish(command_id, request, receipt)
        if request.get("attachments"):
            ok, result, error = execute_slack_uploads(
                request=request,
                env=env,
                destination=destination,
                post=slack_post,
            )
        else:
            ok, result, error = execute_slack_send(request=request, env=env, destination=destination, post=slack_post)
        receipt.update(result)
        receipt["ok"] = bool(ok)
        receipt["delivery_outcome"] = "sent" if ok else "failed"
        if error:
            receipt["error_type"] = error
        return _finish(command_id, request, receipt)

    if kind == KIND_TMUX_SEND:
        target = str((request.get("target") or {}).get("target") or "").strip()
        if not target:
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_MISSING_TARGET)
            return _finish(command_id, request, receipt)
        allowed, target_error, policy_detail = validate_tmux_target(target)
        receipt["policy"]["tmux"] = policy_detail
        if not allowed:
            receipt.update(ok=False, delivery_outcome="failed", error_type=target_error or FAILURE_TARGET_NOT_ALLOWED)
            return _finish(command_id, request, receipt)
        if mode != "send":
            return _finish(command_id, request, receipt)
        if request.get("confirm") is not True:
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_CONFIRMATION_REQUIRED)
            return _finish(command_id, request, receipt)
        if not relay_live_enabled(env):
            receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_LIVE_SEND_DISABLED)
            return _finish(command_id, request, receipt)
        ok, result, error = execute_tmux_send(request)
        receipt.update(result)
        receipt["ok"] = bool(ok)
        receipt["delivery_outcome"] = "sent" if ok else "failed"
        if error:
            receipt["error_type"] = error
        return _finish(command_id, request, receipt)

    receipt.update(ok=False, delivery_outcome="failed", error_type=FAILURE_UNSUPPORTED_KIND)
    return _finish(command_id, request, receipt)
