"""Slack names are contextual evidence, never permissions or actor identity."""

import re

from pydantic import BaseModel, ConfigDict, Field

MAX_PARTICIPANTS = 16
USER_MENTION = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>]+)?>")


class SlackParticipant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: str = Field(pattern=r"^[UW][A-Z0-9]+$", max_length=80)
    user_name: str = Field(default="", max_length=100)
    user_display_name: str = Field(default="", max_length=100)
    user_real_name: str = Field(default="", max_length=100)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name in (
            self.user_name, self.user_display_name, self.user_real_name,
        ) if name)


def render_participant_mentions(text: str, participants: tuple[SlackParticipant, ...]) -> str:
    """Resolve only explicit @names; leave plain names, ambiguous names and code alone."""
    names: dict[str, set[str]] = {}
    for person in participants:
        for name in person.names:
            names.setdefault(name.casefold(), set()).add(person.user_id)
    unique = {name: next(iter(ids)) for name, ids in names.items() if len(ids) == 1}
    if not unique:
        return text
    choices = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    pattern = re.compile(r"(?<![\w@])@(" + choices + r")(?![\w-]|\.[\w])", re.IGNORECASE)

    def replace(match):
        user_id = unique.get(match[1].casefold())
        return f"<@{user_id}>" if user_id else match[0]

    # Protect Slack markup, URLs, and inline/fenced code from accidental pings.
    parts = re.split(r"(```[\s\S]*?```|`[^`\n]*`|<[^>\n]+>|https?://\S+)", text)
    return "".join(part if index % 2 else pattern.sub(replace, part)
                   for index, part in enumerate(parts))
