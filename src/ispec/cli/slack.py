"""Slack bot helpers (Socket Mode).

This module is intentionally lightweight so it can run on instrument-adjacent
machines without additional infrastructure. The bot forwards messages to the
iSPEC assistant API (/api/support/chat) and posts replies back to Slack.
"""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any

import requests

from ispec.logging import get_logger

logger = get_logger(__name__)

_MENTION_RE = re.compile(r"<@[^>]+>")
_CHOOSE_RE = re.compile(r"^\s*choose\s+([01])\s*$", re.IGNORECASE)


def register_subcommands(subparsers) -> None:
    run_parser = subparsers.add_parser("run", help="Run the Slack bot (Socket Mode)")
    run_parser.add_argument(
        "--ispec-server",
        default=os.getenv("ISPEC_SLACK_ISPEC_SERVER") or os.getenv("ISPEC_API_URL") or "",
        help="iSPEC API base URL (default: $ISPEC_SLACK_ISPEC_SERVER or $ISPEC_API_URL)",
    )
    run_parser.add_argument(
        "--api-key",
        default=os.getenv("ISPEC_SLACK_API_KEY") or os.getenv("ISPEC_API_KEY") or "",
        help="Optional iSPEC API key (default: $ISPEC_SLACK_API_KEY or $ISPEC_API_KEY)",
    )
    run_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout for iSPEC calls (default: 60)",
    )


def dispatch(args) -> None:
    if args.subcommand == "run":
        _run_socket_mode(args)
        return
    raise SystemExit(f"Unknown slack subcommand: {args.subcommand}")


@dataclass
class _SlackConfig:
    bot_token: str
    app_token: str
    ispec_server: str
    api_key: str
    timeout_seconds: int


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _clean_slack_text(text: str, *, bot_user_id: str | None) -> str:
    raw = _unescape_slack_text(text)
    if not raw:
        return ""
    if bot_user_id:
        raw = raw.replace(f"<@{bot_user_id}>", "")
    raw = _MENTION_RE.sub("", raw)
    return raw.strip()


def _unescape_slack_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    return raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


def _session_id(*, team_id: str | None, channel: str, thread_ts: str) -> str:
    safe_team = team_id or "unknown"
    return f"slack:{safe_team}:{channel}:{thread_ts}"


def _session_id_for_dm(*, team_id: str | None, channel: str) -> str:
    safe_team = team_id or "unknown"
    return f"slack:{safe_team}:{channel}"


def _pending_key(*, channel: str, thread_ts: str | None) -> str:
    return f"{channel}:{thread_ts or 'root'}"


def _reply_thread_ts(*, channel_type: str | None, thread_ts: str | None) -> str | None:
    if thread_ts:
        return thread_ts
    if channel_type in {"im", "mpim"}:
        return None
    return None


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _post_ispec_chat(
    *,
    server: str,
    api_key: str,
    session_id: str,
    message: str,
    meta: dict[str, Any] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = server.rstrip("/") + "/api/support/chat"
    resp = requests.post(
        url,
        json={"sessionId": session_id, "message": message, "history": [], "meta": meta},
        headers=_headers(api_key),
        timeout=max(1, int(timeout_seconds)),
    )
    resp.raise_for_status()
    return resp.json()


def _post_ispec_choose(
    *,
    server: str,
    api_key: str,
    session_id: str,
    user_message_id: int,
    choice_index: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = server.rstrip("/") + "/api/support/choose"
    resp = requests.post(
        url,
        json={
            "sessionId": session_id,
            "userMessageId": int(user_message_id),
            "choiceIndex": int(choice_index),
        },
        headers=_headers(api_key),
        timeout=max(1, int(timeout_seconds)),
    )
    resp.raise_for_status()
    return resp.json()


def _format_compare_choices(compare: dict[str, Any]) -> str:
    choices = compare.get("choices") if isinstance(compare.get("choices"), list) else []
    lines = [
        "I have two draft replies. Reply in this thread with `choose 0` or `choose 1`.\n"
    ]
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        idx = choice.get("index")
        msg = str(choice.get("message") or "").strip()
        if msg:
            lines.append(f"*Option {idx}*\n{msg}\n")
    return "\n".join(lines).strip()


def _run_socket_mode(args) -> None:
    try:
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "slack-bolt is required for `ispec slack run`. Install with: pip install slack-bolt"
        ) from exc

    try:
        from slack_sdk.errors import SlackApiError
    except Exception:  # pragma: no cover
        SlackApiError = Exception  # type: ignore[misc,assignment]

    cfg = _SlackConfig(
        bot_token=_require_env("SLACK_BOT_TOKEN"),
        app_token=_require_env("SLACK_APP_TOKEN"),
        ispec_server=str(args.ispec_server or "").strip(),
        api_key=str(args.api_key or "").strip(),
        timeout_seconds=max(1, int(args.timeout_seconds)),
    )
    if not cfg.ispec_server:
        raise SystemExit("--ispec-server is required (or set ISPEC_SLACK_ISPEC_SERVER)")

    pending_compare: dict[str, dict[str, Any]] = {}
    user_cache: dict[str, dict[str, str]] = {}

    app = App(token=cfg.bot_token)

    api_key_fingerprint = ""
    if cfg.api_key:
        api_key_fingerprint = hashlib.sha256(cfg.api_key.encode("utf-8")).hexdigest()[:10]
    logger.info(
        "Slack bot config ispec_server=%s api_key_present=%s api_key_fingerprint=%s timeout_seconds=%s",
        cfg.ispec_server,
        bool(cfg.api_key),
        api_key_fingerprint or "none",
        cfg.timeout_seconds,
    )

    # Best-effort token sanity check so failures show up immediately in logs.
    try:
        auth = app.client.auth_test()
        logger.info(
            "Slack auth_test ok team=%s team_id=%s bot_user_id=%s",
            auth.get("team"),
            auth.get("team_id"),
            auth.get("user_id"),
        )
    except Exception:
        logger.exception("Slack auth_test failed (check SLACK_BOT_TOKEN)")

    def _safe_say(*, say, client, channel: str, thread_ts: str | None, text: str) -> None:
        try:
            if thread_ts:
                say(text=text, thread_ts=thread_ts)
            else:
                say(text=text)
            return
        except SlackApiError as exc:
            error = ""
            try:
                error = str(exc.response.get("error") or "")
            except Exception:
                error = ""
            logger.exception("Slack say failed (%s) channel=%s thread_ts=%s", error, channel, thread_ts)

            if error == "not_in_channel":
                try:
                    client.conversations_join(channel=channel)
                except Exception:
                    logger.exception("Slack join failed channel=%s", channel)
                    return
                try:
                    if thread_ts:
                        say(text=text, thread_ts=thread_ts)
                    else:
                        say(text=text)
                except Exception:
                    logger.exception("Slack say retry failed channel=%s thread_ts=%s", channel, thread_ts)

    @app.error
    def _on_slack_error(error, body, logger):  # type: ignore[no-untyped-def]
        try:
            logger.exception("Slack handler error: %s body=%s", error, body)
        except Exception:
            logger.exception("Slack handler error: %s", error)

    @app.event("app_home_opened")
    def _on_app_home_opened(event, body, logger, ack):  # type: ignore[no-untyped-def]
        # Slack fires this when a user opens the bot's App Home. We don't
        # currently publish a Home tab view, but we still register a handler so
        # Bolt doesn't warn about unhandled events.
        try:
            ack()
        except Exception:
            pass
        try:
            logger.info(
                "Slack app_home_opened user=%s tab=%s",
                event.get("user"),
                event.get("tab"),
            )
        except Exception:
            logger.info("Slack app_home_opened")

    @app.event("app_mention")
    def _on_mention(event, say, context, client, ack):  # type: ignore[no-untyped-def]
        try:
            ack()
        except Exception:
            pass
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        channel = str(event.get("channel") or "")
        if not channel:
            return

        thread_ts = str(event.get("thread_ts") or event.get("ts") or "")
        if not thread_ts:
            return

        team_id = context.get("team_id")
        session_id = _session_id(team_id=team_id, channel=channel, thread_ts=thread_ts)

        text = _clean_slack_text(str(event.get("text") or ""), bot_user_id=context.get("bot_user_id"))
        if not text:
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                text="Say something after mentioning me.",
            )
            return

        slack_user_id = str(event.get("user") or "").strip()

        def _slack_user_summary(user_id: str) -> dict[str, str]:
            user_id = (user_id or "").strip()
            if not user_id:
                return {}
            cached = user_cache.get(user_id)
            if isinstance(cached, dict) and cached:
                return cached
            summary: dict[str, str] = {"user_id": user_id}
            try:
                info = client.users_info(user=user_id)
                user_obj = info.get("user") if isinstance(info, dict) else None
                if isinstance(user_obj, dict):
                    name = str(user_obj.get("name") or "").strip()
                    if name:
                        summary["user_name"] = name
                    profile = user_obj.get("profile") if isinstance(user_obj.get("profile"), dict) else {}
                    display = str(profile.get("display_name") or "").strip()
                    if not display:
                        display = str(profile.get("real_name") or "").strip()
                    if display:
                        summary["user_display_name"] = display
            except Exception:
                pass
            user_cache[user_id] = summary
            return summary

        slack_user = _slack_user_summary(slack_user_id)
        speaker = (
            str(slack_user.get("user_display_name") or "").strip()
            or str(slack_user.get("user_name") or "").strip()
            or slack_user_id
        )
        speaker = speaker.replace("\n", " ").replace("\r", " ").strip()
        if speaker:
            speaker = speaker[:64]
        message_for_ispec = f"[{speaker}] {text}" if speaker else text
        meta: dict[str, Any] = {
            "source": "slack",
            "slack": {
                "event": "app_mention",
                "team_id": team_id,
                "channel": channel,
                "channel_type": event.get("channel_type"),
                "thread_ts": thread_ts,
                "message_ts": event.get("ts"),
                "user_id": slack_user_id or None,
                "user_name": slack_user.get("user_name"),
                "user_display_name": slack_user.get("user_display_name"),
            },
        }

        logger.info(
            "Slack app_mention channel=%s thread_ts=%s user=%s text=%r",
            channel,
            thread_ts,
            slack_user_id,
            message_for_ispec,
        )

        match = _CHOOSE_RE.match(text)
        if match:
            key = _pending_key(channel=channel, thread_ts=thread_ts)
            pending = pending_compare.get(key)
            if isinstance(pending, dict):
                try:
                    choice_index = int(match.group(1))
                except Exception:
                    choice_index = -1
                session_id_pending = str(pending.get("session_id") or "")
                user_message_id = int(pending.get("user_message_id") or 0)
                if session_id_pending and user_message_id > 0 and choice_index in {0, 1}:
                    try:
                        payload = _post_ispec_choose(
                            server=cfg.ispec_server,
                            api_key=cfg.api_key,
                            session_id=session_id_pending,
                            user_message_id=user_message_id,
                            choice_index=choice_index,
                            timeout_seconds=cfg.timeout_seconds,
                        )
                    except requests.HTTPError as exc:
                        body = ""
                        try:
                            body = exc.response.text if exc.response is not None else ""
                        except Exception:
                            body = ""
                        _safe_say(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=thread_ts,
                            text=f"iSPEC API error: {exc}\n{body}".strip(),
                        )
                        return
                    except Exception as exc:
                        _safe_say(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=thread_ts,
                            text=f"iSPEC API call failed: {exc}",
                        )
                        return
                    pending_compare.pop(key, None)
                    message = str(payload.get("message") or "").strip()
                    if message:
                        _safe_say(say=say, client=client, channel=channel, thread_ts=thread_ts, text=message)
                    return
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                text="No pending compare choices in this thread.",
            )
            return

        try:
            payload = _post_ispec_chat(
                server=cfg.ispec_server,
                api_key=cfg.api_key,
                session_id=session_id,
                message=message_for_ispec,
                meta=meta,
                timeout_seconds=cfg.timeout_seconds,
            )
        except requests.HTTPError as exc:
            body = ""
            try:
                body = exc.response.text if exc.response is not None else ""
            except Exception:
                body = ""
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                text=f"iSPEC API error: {exc}\n{body}".strip(),
            )
            return
        except Exception as exc:
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                text=f"iSPEC API call failed: {exc}",
            )
            return

        compare = payload.get("compare") if isinstance(payload.get("compare"), dict) else None
        message = str(payload.get("message") or "").strip()
        if compare is not None:
            key = _pending_key(channel=channel, thread_ts=thread_ts)
            pending_compare[key] = {
                "session_id": session_id,
                "user_message_id": int(compare.get("userMessageId") or 0),
            }
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                text=_format_compare_choices(compare),
            )
            return

        if not message:
            _safe_say(say=say, client=client, channel=channel, thread_ts=thread_ts, text="(no response)")
            return
        _safe_say(say=say, client=client, channel=channel, thread_ts=thread_ts, text=message)

    @app.event("message")
    def _on_message(event, say, client, context, ack):  # type: ignore[no-untyped-def]
        """Handle DMs and thread followups (choose 0/1)."""

        try:
            ack()
        except Exception:
            pass
        if event.get("bot_id") or event.get("subtype"):
            return

        channel = str(event.get("channel") or "")
        if not channel:
            return

        channel_type = event.get("channel_type")
        team_id = context.get("team_id")

        raw_text = _unescape_slack_text(str(event.get("text") or ""))
        if not raw_text:
            return

        thread_ts_raw = str(event.get("thread_ts") or "")
        reply_thread = _reply_thread_ts(channel_type=channel_type, thread_ts=thread_ts_raw or None)
        pending_key = _pending_key(channel=channel, thread_ts=thread_ts_raw or None)

        choose_match = _CHOOSE_RE.match(raw_text)
        if choose_match:
            pending = pending_compare.get(pending_key)
            if isinstance(pending, dict):
                try:
                    choice_index = int(choose_match.group(1))
                except Exception:
                    return

                session_id = str(pending.get("session_id") or "")
                user_message_id = int(pending.get("user_message_id") or 0)
                if session_id and user_message_id > 0:
                    try:
                        payload = _post_ispec_choose(
                            server=cfg.ispec_server,
                            api_key=cfg.api_key,
                            session_id=session_id,
                            user_message_id=user_message_id,
                            choice_index=choice_index,
                            timeout_seconds=cfg.timeout_seconds,
                        )
                    except requests.HTTPError as exc:
                        body = ""
                        try:
                            body = exc.response.text if exc.response is not None else ""
                        except Exception:
                            body = ""
                        _safe_say(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=reply_thread,
                            text=f"iSPEC API error: {exc}\n{body}".strip(),
                        )
                        return
                    except Exception as exc:
                        _safe_say(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=reply_thread,
                            text=f"iSPEC API call failed: {exc}",
                        )
                        return

                    pending_compare.pop(pending_key, None)
                    message = str(payload.get("message") or "").strip()
                    if message:
                        _safe_say(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=reply_thread,
                            text=message,
                        )
            return

        # Only respond to free-form messages in DMs; keep channels quiet unless mentioned.
        if channel_type not in {"im", "mpim"}:
            return

        session_id = _session_id_for_dm(team_id=team_id, channel=channel)
        logger.info(
            "Slack DM message channel=%s user=%s text=%r",
            channel,
            event.get("user"),
            raw_text,
        )

        slack_user_id = str(event.get("user") or "").strip()
        if slack_user_id:
            cached = user_cache.get(slack_user_id) or {}
            if not cached:
                try:
                    info = client.users_info(user=slack_user_id)
                    user_obj = info.get("user") if isinstance(info, dict) else None
                    if isinstance(user_obj, dict):
                        name = str(user_obj.get("name") or "").strip()
                        profile = user_obj.get("profile") if isinstance(user_obj.get("profile"), dict) else {}
                        display = str(profile.get("display_name") or "").strip()
                        if not display:
                            display = str(profile.get("real_name") or "").strip()
                        cached = {
                            "user_id": slack_user_id,
                            "user_name": name,
                            "user_display_name": display,
                        }
                except Exception:
                    cached = {"user_id": slack_user_id}
                user_cache[slack_user_id] = cached
            speaker = (
                str(cached.get("user_display_name") or "").strip()
                or str(cached.get("user_name") or "").strip()
                or slack_user_id
            )
        else:
            cached = {}
            speaker = ""

        speaker = speaker.replace("\n", " ").replace("\r", " ").strip()
        if speaker:
            speaker = speaker[:64]
        message_for_ispec = f"[{speaker}] {raw_text}" if speaker else raw_text
        meta: dict[str, Any] = {
            "source": "slack",
            "slack": {
                "event": "dm",
                "team_id": team_id,
                "channel": channel,
                "channel_type": channel_type,
                "thread_ts": thread_ts_raw or None,
                "message_ts": event.get("ts"),
                "user_id": slack_user_id or None,
                "user_name": cached.get("user_name"),
                "user_display_name": cached.get("user_display_name"),
            },
        }

        try:
            payload = _post_ispec_chat(
                server=cfg.ispec_server,
                api_key=cfg.api_key,
                session_id=session_id,
                message=message_for_ispec,
                meta=meta,
                timeout_seconds=cfg.timeout_seconds,
            )
        except requests.HTTPError as exc:
            body = ""
            try:
                body = exc.response.text if exc.response is not None else ""
            except Exception:
                body = ""
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=reply_thread,
                text=f"iSPEC API error: {exc}\n{body}".strip(),
            )
            return
        except Exception as exc:
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=reply_thread,
                text=f"iSPEC API call failed: {exc}",
            )
            return

        compare = payload.get("compare") if isinstance(payload.get("compare"), dict) else None
        message = str(payload.get("message") or "").strip()
        if compare is not None:
            pending_compare[pending_key] = {
                "session_id": session_id,
                "user_message_id": int(compare.get("userMessageId") or 0),
            }
            _safe_say(
                say=say,
                client=client,
                channel=channel,
                thread_ts=reply_thread,
                text=_format_compare_choices(compare),
            )
            return

        if not message:
            _safe_say(say=say, client=client, channel=channel, thread_ts=reply_thread, text="(no response)")
            return
        _safe_say(say=say, client=client, channel=channel, thread_ts=reply_thread, text=message)

    logger.info("Starting Slack bot (Socket Mode) -> %s", cfg.ispec_server)
    SocketModeHandler(app, cfg.app_token).start()
