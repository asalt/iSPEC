"""Deterministic review leads, deliberately disconnected from authorization.

Presence in source is not entailment. This cannot establish which project a fact
belongs to, resolve pronouns, distinguish a retracted count, or detect chatter.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True, kw_only=True)
class Fact:
    kind: str
    value: str
    text: str


@dataclass(frozen=True, kw_only=True)
class FactReview:
    unsupported: tuple[Fact, ...]
    omitted: tuple[Fact, ...]
    advisory_only: bool = True


_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    map(str, range(21)), strict=True,
))
_PATTERNS = (
    ("project_id", re.compile(r"\b(?:project\s*(?:id\s*)?[:#]?\s*|MSPC0*)(\d+)\b", re.I)),
    ("sample_id", re.compile(r"\b\d{4,7}_\d+_\d+(?:_\d+)?\b")),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("date", re.compile(r"\b(?:today|yesterday|tomorrow|this afternoon|(?:next|last|this)\s+(?:week|month|year)|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.I)),
    ("sample_id", re.compile(r"\b[A-Za-z]{1,6}[-_]?[0-9]+(?:[-_][A-Za-z0-9]+)*\b")),
    ("number", re.compile(r"\b\d+(?:\.\d+)?\b|\b(?:" + "|".join(_WORDS) + r")\b", re.I)),
)


def extract_facts(*, text: str) -> tuple[Fact, ...]:
    occupied: list[tuple[int, int]] = []
    facts: dict[tuple[str, str], Fact] = {}
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            occupied.append(match.span())
            surface = match.group()
            value = str(int(match.group(1))) if kind == "project_id" else surface.casefold()
            value = _WORDS.get(value, value)
            facts[(kind, value)] = Fact(kind=kind, value=value, text=surface)
    return tuple(facts.values())


def review_draft_facts(*, draft: str, sources: tuple[str, ...]) -> FactReview:
    source_facts = extract_facts(text="\n".join(sources))
    draft_facts = extract_facts(text=draft)
    source_keys = {(fact.kind, fact.value) for fact in source_facts}
    draft_keys = {(fact.kind, fact.value) for fact in draft_facts}
    return FactReview(
        unsupported=tuple(fact for fact in draft_facts if (fact.kind, fact.value) not in source_keys),
        # The project is often in the server-rendered heading rather than the note.
        omitted=tuple(fact for fact in source_facts if fact.kind != "project_id"
                      and (fact.kind, fact.value) not in draft_keys),
    )
