from __future__ import annotations

from ispec.api.routes.support import _is_affirmative_reply, _is_confirmation_reply


def test_confirmation_reply_accepts_compound_confirmation_phrase():
    assert _is_confirmation_reply("Confirm yes commit it") is True
    assert _is_affirmative_reply("Confirm yes commit it") is True


def test_confirmation_reply_rejects_negative_affirmative_mix():
    assert _is_confirmation_reply("no dont commit it") is True
    assert _is_affirmative_reply("no dont commit it") is False

