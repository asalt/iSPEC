from ispec.assistant.slack_participants import SlackParticipant, render_participant_mentions
from ispec.assistant.slack_thread_context import SlackThreadContext, render_thread_context
from ispec.cli.slack import _clean_slack_text, _slack_participant_map, _slack_user_summary


def test_real_mentions_survive_cleaning():
    assert _clean_slack_text("<@UBOT> ask <@U123> and <@U456>", bot_user_id="UBOT") == "ask <@U123> and <@U456>"


def test_mentions_use_unique_names_without_changing_plain_names_or_code():
    participants = (SlackParticipant(user_id="U1", user_name="shiro", user_display_name="Shiro"),
                    SlackParticipant(user_id="U2", user_name="antrix", user_real_name="Antrix Example"))
    text = "Ask @Shiro and @Antrix Example. Plain shiro, <@U2>, `@shiro`, a@shiro and https://x/@shiro stay."
    assert render_participant_mentions(text, participants) == (
        "Ask <@U1> and <@U2>. Plain shiro, <@U2>, `@shiro`, a@shiro and https://x/@shiro stay.")
    ambiguous = participants + (SlackParticipant(user_id="U3", user_display_name="shiro"),)
    assert render_participant_mentions("@shiro @unknown", ambiguous) == "@shiro @unknown"


def test_participant_lookup_is_bounded_cached_and_supports_slack_response():
    class Response:
        def get(self, key):
            return {"name": "shiro", "profile": {"display_name": "Shiro", "real_name": "Example Name"}}
    class Client:
        def __init__(self):
            self.calls = []
        def users_info(self, *, user):
            self.calls.append(user)
            return Response()
    client, cache = Client(), {}
    _slack_user_summary(client=client, user_cache=cache, user_id="U1")
    meta = {"slack": {"user_id": "U1"}, "slack_thread_context": {"messages": [
        {"user_id": f"U{n}", "text": "ask <@U2>"} for n in range(2, 40)
    ]}}
    result = _slack_participant_map(client=client, user_cache=cache, meta=meta, text="ask shiro <@U2>")
    assert len(result) == 16 and len(client.calls) == 16
    assert result[0]["user_display_name"] == "Shiro"
    _slack_participant_map(client=client, user_cache=cache, meta=meta, text="ask shiro <@U2>")
    assert len(client.calls) == 16


def test_participants_render_for_dm_without_historical_thread_or_authority():
    meta = {"source": "slack", "slack": {"team_id": "T1", "channel": "D1", "message_ts": "10.0"},
            "slack_participants": [{"user_id": "U1", "user_name": "shiro", "user_display_name": "Shiro"}]}
    context = SlackThreadContext.from_meta(meta)
    assert context and not context.messages
    prompt = render_thread_context(context).messages[0]["content"]
    assert '"user_id": "U1"' in prompt and "Shiro" in prompt
    assert "not permissions" in prompt
    assert SlackThreadContext.from_meta({**meta, "source": "frontend"}) is None
