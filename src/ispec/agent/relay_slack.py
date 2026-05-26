"""Slack delivery helpers for the local relay dispatcher."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Callable

import requests

from ispec.agent.relay_config import CanonicalEnv, slack_timeout_seconds, slack_upload_max_bytes, token_info
from ispec.agent.relay_constants import (
    FAILURE_ATTACHMENT_MISSING,
    FAILURE_ATTACHMENT_TOO_LARGE,
    FAILURE_ATTACHMENT_UNSUPPORTED,
    FAILURE_ATTACHMENT_UPLOAD_FAILED,
    FAILURE_MISSING_TARGET,
    FAILURE_MISSING_TOKEN,
    FAILURE_PROVIDER_ERROR,
)
from ispec.agent.relay_utils import truncate


def slack_api_call(
    *,
    token: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    post: Callable[..., Any] | None = None,
    as_form: bool = False,
) -> dict[str, Any]:
    post_fn = post or requests.post
    if as_form:
        request_kwargs = {"data": payload}
        content_type = "application/x-www-form-urlencoded"
    else:
        request_kwargs = {"json": payload}
        content_type = "application/json; charset=utf-8"
    resp = post_fn(
        f"https://slack.com/api/{endpoint.lstrip('/')}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        },
        **request_kwargs,
        timeout=max(1.0, float(timeout_seconds)),
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def resolve_slack_channel(
    *,
    token: str,
    destination: dict[str, Any],
    timeout_seconds: float,
    post: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    channel = str(destination.get("channel") or "").strip()
    user_id = str(destination.get("user_id") or "").strip()
    email = str(destination.get("email") or "").strip()
    if not channel and email:
        lookup = slack_api_call(
            token=token,
            endpoint="users.lookupByEmail",
            payload={"email": email},
            timeout_seconds=timeout_seconds,
            post=post,
        )
        if lookup.get("ok") is not True:
            error = str(lookup.get("error") or "unknown_error")
            return {"ok": False, "error": error, "slack": lookup}, FAILURE_PROVIDER_ERROR
        user = lookup.get("user")
        if isinstance(user, dict):
            user_id = str(user.get("id") or "").strip()

    if not channel and user_id:
        opened = slack_api_call(
            token=token,
            endpoint="conversations.open",
            payload={"users": user_id},
            timeout_seconds=timeout_seconds,
            post=post,
        )
        if opened.get("ok") is not True:
            error = str(opened.get("error") or "unknown_error")
            return {"ok": False, "error": error, "slack": opened}, FAILURE_PROVIDER_ERROR
        channel_obj = opened.get("channel")
        if isinstance(channel_obj, dict):
            channel = str(channel_obj.get("id") or "").strip()

    if not channel:
        return {"ok": False, "error": "Missing channel after destination resolution."}, FAILURE_MISSING_TARGET

    return {"ok": True, "channel": channel, "user_id": user_id or None, "email": email or None}, None


def execute_slack_send(
    *,
    request: dict[str, Any],
    env: CanonicalEnv,
    destination: dict[str, Any],
    post: Callable[..., Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    token, token_meta = token_info(env)
    if not token:
        return False, {"ok": False, "error_type": FAILURE_MISSING_TOKEN, "token": token_meta}, FAILURE_MISSING_TOKEN
    timeout_seconds = slack_timeout_seconds(env)

    channel_result, channel_error = resolve_slack_channel(
        token=token,
        destination=destination,
        timeout_seconds=timeout_seconds,
        post=post,
    )
    if channel_error:
        return False, channel_result, channel_error

    channel = str(channel_result.get("channel") or "").strip()
    user_id = str(channel_result.get("user_id") or "").strip()
    email = str(channel_result.get("email") or "").strip()
    if not channel:
        return False, {"ok": False, "error": "Missing channel after destination resolution."}, FAILURE_MISSING_TARGET

    message_payload: dict[str, Any] = {"channel": channel, "text": str(request.get("body") or "")}
    if request.get("thread_ts"):
        message_payload["thread_ts"] = str(request["thread_ts"])
    posted = slack_api_call(
        token=token,
        endpoint="chat.postMessage",
        payload=message_payload,
        timeout_seconds=timeout_seconds,
        post=post,
    )
    if posted.get("ok") is not True:
        error = str(posted.get("error") or "unknown_error")
        return False, {"ok": False, "error": error, "slack": posted}, FAILURE_PROVIDER_ERROR
    return True, {
        "ok": True,
        "sent": True,
        "channel": channel,
        "user_id": user_id or None,
        "email": email or None,
        "thread_ts": request.get("thread_ts"),
        "slack": posted,
    }, None


def _resolve_pdf_attachment(item: dict[str, Any], *, env: CanonicalEnv) -> tuple[dict[str, Any] | None, str | None]:
    raw_path = str(item.get("path") or "").strip()
    if not raw_path:
        return None, FAILURE_ATTACHMENT_MISSING
    try:
        file_path = Path(raw_path).expanduser().resolve()
    except Exception:
        file_path = Path(raw_path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return None, FAILURE_ATTACHMENT_MISSING
    if file_path.name.startswith(".env"):
        return None, FAILURE_ATTACHMENT_UNSUPPORTED
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    if file_path.suffix.lower() != ".pdf" and mime_type != "application/pdf":
        return None, FAILURE_ATTACHMENT_UNSUPPORTED
    size_bytes = int(file_path.stat().st_size)
    max_bytes = slack_upload_max_bytes(env)
    if size_bytes > max_bytes:
        return None, FAILURE_ATTACHMENT_TOO_LARGE
    title = truncate(item.get("title"), limit=240) or file_path.name
    return {
        "path": str(file_path),
        "filename": file_path.name,
        "title": title,
        "size_bytes": size_bytes,
        "mime_type": mime_type,
    }, None


def upload_slack_file_external(
    *,
    token: str,
    channel: str,
    attachment: dict[str, Any],
    timeout_seconds: float,
    text: str | None,
    thread_ts: str | None,
    post: Callable[..., Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    get_url_response = slack_api_call(
        token=token,
        endpoint="files.getUploadURLExternal",
        payload={
            "filename": attachment["filename"],
            "length": attachment["size_bytes"],
        },
        timeout_seconds=timeout_seconds,
        post=post,
        as_form=True,
    )
    if get_url_response.get("ok") is not True:
        return False, {
            "ok": False,
            "stage": "get_upload_url",
            "error": str(get_url_response.get("error") or "unknown_error"),
            "file": {key: attachment[key] for key in ("path", "filename", "size_bytes", "mime_type")},
            "slack": get_url_response,
        }, FAILURE_ATTACHMENT_UPLOAD_FAILED

    upload_url = str(get_url_response.get("upload_url") or "").strip()
    file_id = str(get_url_response.get("file_id") or "").strip()
    if not upload_url or not file_id:
        return False, {
            "ok": False,
            "stage": "get_upload_url",
            "error": "missing_upload_url_or_file_id",
            "file": {key: attachment[key] for key in ("path", "filename", "size_bytes", "mime_type")},
        }, FAILURE_ATTACHMENT_UPLOAD_FAILED

    post_fn = post or requests.post
    with Path(attachment["path"]).open("rb") as handle:
        upload_response = post_fn(
            upload_url,
            files={"file": (attachment["filename"], handle, attachment["mime_type"])},
            timeout=max(1.0, float(timeout_seconds)),
        )
    status_code = int(getattr(upload_response, "status_code", 200) or 200)
    if status_code < 200 or status_code >= 300:
        return False, {
            "ok": False,
            "stage": "upload_bytes",
            "error": f"upload_http_{status_code}",
            "file": {key: attachment[key] for key in ("path", "filename", "size_bytes", "mime_type")},
            "body": str(getattr(upload_response, "text", ""))[:500],
        }, FAILURE_ATTACHMENT_UPLOAD_FAILED

    complete_payload: dict[str, Any] = {
        "files": [{"id": file_id, "title": attachment["title"]}],
        "channel_id": channel,
    }
    if thread_ts:
        complete_payload["thread_ts"] = str(thread_ts).strip()
    if text:
        complete_payload["initial_comment"] = str(text).strip()
    complete_response = slack_api_call(
        token=token,
        endpoint="files.completeUploadExternal",
        payload=complete_payload,
        timeout_seconds=timeout_seconds,
        post=post,
    )
    if complete_response.get("ok") is not True:
        return False, {
            "ok": False,
            "stage": "complete_upload",
            "error": str(complete_response.get("error") or "unknown_error"),
            "file": {key: attachment[key] for key in ("path", "filename", "size_bytes", "mime_type")},
            "slack": complete_response,
        }, FAILURE_ATTACHMENT_UPLOAD_FAILED

    return True, {
        "ok": True,
        "file_id": file_id,
        "file": {key: attachment[key] for key in ("path", "filename", "title", "size_bytes", "mime_type")},
        "slack": complete_response,
    }, None


def execute_slack_uploads(
    *,
    request: dict[str, Any],
    env: CanonicalEnv,
    destination: dict[str, Any],
    post: Callable[..., Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    token, token_meta = token_info(env)
    if not token:
        return False, {"ok": False, "error_type": FAILURE_MISSING_TOKEN, "token": token_meta}, FAILURE_MISSING_TOKEN
    timeout_seconds = slack_timeout_seconds(env)
    channel_result, channel_error = resolve_slack_channel(
        token=token,
        destination=destination,
        timeout_seconds=timeout_seconds,
        post=post,
    )
    if channel_error:
        return False, channel_result, channel_error

    channel = str(channel_result.get("channel") or "").strip()
    attachments: list[dict[str, Any]] = []
    for item in request.get("attachments") or []:
        if not isinstance(item, dict):
            return False, {"ok": False, "error": "Invalid attachment entry."}, FAILURE_ATTACHMENT_UNSUPPORTED
        attachment, attachment_error = _resolve_pdf_attachment(item, env=env)
        if attachment_error or attachment is None:
            return False, {"ok": False, "error_type": attachment_error, "attachment": item}, attachment_error
        attachments.append(attachment)

    uploaded: list[dict[str, Any]] = []
    for idx, attachment in enumerate(attachments):
        ok, result, error = upload_slack_file_external(
            token=token,
            channel=channel,
            attachment=attachment,
            timeout_seconds=timeout_seconds,
            text=str(request.get("body") or "").strip() if idx == 0 else None,
            thread_ts=request.get("thread_ts"),
            post=post,
        )
        if not ok:
            return False, {
                "ok": False,
                "sent": bool(uploaded),
                "channel": channel,
                "uploaded": uploaded,
                "failed_upload": result,
            }, error or FAILURE_ATTACHMENT_UPLOAD_FAILED
        uploaded.append(result)

    return True, {
        "ok": True,
        "sent": True,
        "channel": channel,
        "user_id": channel_result.get("user_id"),
        "email": channel_result.get("email"),
        "thread_ts": request.get("thread_ts"),
        "attachments_uploaded": uploaded,
    }, None
