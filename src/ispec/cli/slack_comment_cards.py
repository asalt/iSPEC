"""Socket Mode confirmation-card delivery and button callbacks."""

from dataclasses import dataclass
import json
import logging
from typing import Any

import requests

from ispec.assistant.project_comment_cards import ConfirmationCard


logger = logging.getLogger(__name__)
SAVE_ACTION = "ispec_project_comment_save"
CANCEL_ACTION = "ispec_project_comment_cancel"
UNDO_ACTION = "ispec_project_comment_undo"
_ACTION_NAMES = {SAVE_ACTION: "save", CANCEL_ACTION: "cancel", UNDO_ACTION: "undo"}


def card_message(*, card: ConfirmationCard) -> dict[str, Any]:
    """Render exact stored text as plain text, keeping Slack markup inert."""
    state = (
        f"Saved as comment #{card.comment_id}"
        if card.status == "saved"
        else (
            f"Undone - comment #{card.comment_id}"
            if card.status == "undone"
            else card.status.capitalize()
        )
    )
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Project {card.project_id} note"},
        }
    ]
    for start in range(0, len(card.comment), 2500):
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": card.comment[start : start + 2500],
                    "emoji": False,
                },
            }
        )
    blocks.append(
        {
            "type": "context",
            "elements": [
                {"type": "plain_text", "text": f"iSPEC confirmation card: {state}"}
            ],
        }
    )
    if card.status == "pending" or card.can_undo:
        value = json.dumps(
            {
                "card_id": card.card_id,
                "draft_id": card.draft_id,
                "content_hash": card.content_hash,
            }
        )
        buttons = (
            [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Undo"},
                    "action_id": UNDO_ACTION,
                    "value": value,
                }
            ]
            if card.can_undo
            else [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Save to project history"},
                    "action_id": SAVE_ACTION,
                    "value": value,
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Cancel"},
                    "action_id": CANCEL_ACTION,
                    "value": value,
                },
            ]
        )
        blocks.append(
            {
                "type": "actions",
                "elements": buttons,
            }
        )
    return {
        "text": f"Project {card.project_id} note ({state}): {card.comment}",
        "blocks": blocks,
    }


@dataclass(frozen=True, kw_only=True)
class SlackCommentCards:
    server: str
    api_key: str
    timeout_seconds: int = 10

    def _post(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            self.server.rstrip("/") + "/api/support/project-comment-cards/" + path,
            json=payload,
            headers={"X-API-Key": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def publish(
        self,
        *,
        payload: dict[str, Any],
        client: Any,
        team_id: str,
        channel_id: str,
        thread_ts: str,
    ) -> None:
        """Post new cards and refresh previous cards after revision or cancellation."""
        for raw in payload.get("commentCards", []):
            try:
                card = ConfirmationCard.model_validate(raw)
                if card.message_ts:
                    client.chat_update(
                        channel=channel_id,
                        ts=card.message_ts,
                        **card_message(card=card),
                    )
                    continue
                if card.status not in {"pending", "saved"}:
                    continue
                response = client.chat_postMessage(
                    channel=channel_id,
                    **({"thread_ts": thread_ts} if thread_ts else {}),
                    **card_message(card=card),
                )
                self._post(
                    path=f"{card.card_id}/delivery",
                    payload={
                        "team_id": team_id,
                        "channel_id": channel_id,
                        "thread_ts": thread_ts,
                        "message_ts": str(response["ts"]),
                    },
                )
            except Exception as exc:
                # Delivery failure cannot authorize a click; the backend requires
                # the persisted message binding. Never log tokens or response URLs.
                logger.warning(
                    "Confirmation card delivery failed error_type=%s",
                    type(exc).__name__,
                )

    def handle_action(self, *, ack: Any, body: dict[str, Any], client: Any) -> None:
        """Acknowledge immediately, then validate through the shared backend endpoint."""
        ack()
        channel = str((body.get("channel") or {}).get("id") or "")
        user_id = str((body.get("user") or {}).get("id") or "")
        message_ts = str((body.get("container") or {}).get("message_ts") or "")
        try:
            actions = body.get("actions") or []
            if body.get("type") != "block_actions" or len(actions) != 1:
                raise ValueError("Invalid action envelope")
            action = actions[0]
            if action.get("action_id") not in _ACTION_NAMES:
                raise ValueError("Unknown card action")
            value = json.loads(action["value"])
            if set(value) != {"card_id", "draft_id", "content_hash"}:
                raise ValueError("Invalid button value")
            # IDs originate in our card payload; constrain the URL path as well.
            card_id = str(value.pop("card_id"))
            if len(card_id) != 32 or any(c not in "0123456789abcdef" for c in card_id):
                raise ValueError("Invalid card ID")
            result = self._post(
                path=f"{card_id}/actions",
                payload={
                    **value,
                    "action": _ACTION_NAMES[action["action_id"]],
                    "slack": {
                        "team_id": str((body.get("team") or {}).get("id") or ""),
                        "channel_id": channel,
                        "message_ts": message_ts,
                        "user_id": user_id,
                    },
                },
            )
            if result.get("card"):
                card = ConfirmationCard.model_validate(result["card"])
                client.chat_update(
                    channel=channel, ts=message_ts, **card_message(card=card)
                )
            if not result.get("ok"):
                client.chat_postEphemeral(
                    channel=channel,
                    user=user_id,
                    text=result.get("error")
                    or "This card is no longer available. Use the latest draft.",
                )
        except Exception as exc:
            logger.warning(
                "Confirmation card action failed error_type=%s", type(exc).__name__
            )
            if channel and user_id:
                client.chat_postEphemeral(
                    channel=channel,
                    user=user_id,
                    text="Could not confirm the card action. Its saved state will be checked if you click again.",
                )

    def register(self, *, app: Any) -> None:
        app.action(SAVE_ACTION)(self.handle_action)
        app.action(CANCEL_ACTION)(self.handle_action)
        app.action(UNDO_ACTION)(self.handle_action)
