"""Rank remembered project terms over an already access-scoped SQL query."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import re

from sqlalchemy.orm import Query

from ispec.db.models import Project, ProjectComment


STOP = frozenset(
    "a an and are at by can could did do does find for from get give had have i id in is it "
    "its lab look looking me my name named number of on or our pi please project projects "
    "recall remember search show that the their there this to up was we what which who with "
    "work you contact title topic about called involving associated tell".split()
)
FIELDS = {
    "title": (Project.prj_ProjectTitle, 6),
    "pi": (Project.prj_PI, 6),
    "lab_contact": (Project.prj_Project_LabContact, 6),
    "description": (Project.prj_ProjectDescription, 3),
    "background": (Project.prj_ProjectBackground, 2),
    "samples": (Project.prj_ProjectSamples, 2),
    "questions": (Project.prj_ProjectQuestions, 2),
}
WORD = re.compile(r"[^\W_]+", re.UNICODE)


@lru_cache(maxsize=1)
def vocabulary():
    groups = json.loads(files(__package__).joinpath("project_terms.json").read_text())
    aliases = {alias: tuple(group) for group in groups for alias in group}
    alternatives = sorted(aliases, key=len, reverse=True)
    lex = re.compile(
        r"(?<!\w)(?:" + "|".join(map(re.escape, alternatives)) + r")(?!\w)|[^\W_]+",
        re.I,
    )
    return aliases, lex


@lru_cache(maxsize=4096)
def near_name(left: str, right: str) -> bool:
    """Allow one insertion, deletion, substitution or adjacent swap in a name."""
    if min(len(left), len(right)) < 5 or abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    i = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), len(left))
    if len(left) < len(right):
        return left[i:] == right[i + 1 :]
    return left[i + 1 :] == right[i + 1 :] or (
        left[i : i + 2] == right[i : i + 2][::-1] and left[i + 2 :] == right[i + 2 :]
    )


@dataclass(frozen=True)
class Term:
    label: str
    pattern: re.Pattern
    title_prefix: re.Pattern | None


def query_terms(text: str) -> list[Term]:
    aliases, lex = vocabulary()
    terms = {}
    for token in lex.findall(text.casefold()):
        if token in STOP:
            continue
        variants = aliases.get(token, (token,))
        if variants in terms:
            continue
        pattern = (
            r"(?<![^\W_])(?:" + "|".join(map(re.escape, variants)) + r")(?![^\W_])"
        )
        prefix = None
        if len(token) >= 4 and token.isalpha():
            prefix = re.compile(r"(?<![^\W_])" + re.escape(token) + r"[^\W_]+", re.I)
        terms[variants] = Term(token, re.compile(pattern, re.I), prefix)
    return list(terms.values())


@dataclass
class Hit:
    weight: int
    field: str
    snippet: str


def collect(
    hits: dict[str, Hit], *, terms: list[Term], field: str, value: str, weight: int
) -> None:
    for term in terms:
        if term.label in hits and hits[term.label].weight >= weight:
            continue
        match = term.pattern.search(value)
        score = weight
        if not match and field == "title" and term.title_prefix:
            match = term.title_prefix.search(value)
            score -= 2
        if not match and field in {"pi", "lab_contact"}:
            match = next(
                (
                    m
                    for m in WORD.finditer(value)
                    if near_name(term.label, m[0].casefold())
                ),
                None,
            )
            score = weight - 2
        if match and (term.label not in hits or score > hits[term.label].weight):
            start = max(0, match.start() - 35)
            hits[term.label] = Hit(score, field, value[start : start + 110].strip())


def ranked_projects(query: Query, *, text: str, limit: int, include_comments: bool):
    """Return the ranked, bounded page and total before pagination. No index to sync."""
    terms = query_terms(text)
    if not terms:
        return [], 0
    id_match = re.fullmatch(
        r"\s*(?:(?:mspc|project)\s*#?\s*|#)?(\d{1,12})\s*", text, re.I
    )
    if id_match:
        query = query.filter(Project.id == int(id_match[1]))
    columns = [column.label(field) for field, (column, _) in FIELDS.items()]
    rows = query.with_entities(
        Project.id,
        Project.prj_Status,
        Project.prj_Current_FLAG,
        Project.prj_Billing_ReadyToBill,
        *columns,
    )
    candidates = {}
    for row in rows.yield_per(256):
        hits = {}
        candidates[row.id] = (row, hits)
        if id_match:
            hits["id"] = Hit(10, "id", str(row.id))
            continue
        for field, (_, weight) in FIELDS.items():
            value = getattr(row, field)
            if value:
                collect(hits, terms=terms, field=field, value=value, weight=weight)

    if include_comments and not id_match:
        comments = (
            query.join(ProjectComment, ProjectComment.project_id == Project.id)
            .with_entities(Project.id, ProjectComment.com_Comment)
            .order_by(ProjectComment.id.desc())
        )
        for project_id, value in comments.yield_per(256):
            if value and project_id in candidates:
                collect(
                    candidates[project_id][1],
                    terms=terms,
                    field="comment",
                    value=value,
                    weight=1,
                )

    ranked = [(row, hits) for row, hits in candidates.values() if hits]
    ranked.sort(
        key=lambda pair: (
            -len(pair[1]),
            -sum(h.weight for h in pair[1].values()),
            pair[0].id,
        )
    )
    matches = []
    for row, hits in ranked[:limit]:
        matches.append(
            {
                "id": row.id,
                "title": row.title,
                "status": row.prj_Status,
                "current": bool(row.prj_Current_FLAG),
                "to_be_billed": bool(row.prj_Billing_ReadyToBill),
                "links": {"ui": f"/project/{row.id}", "api": f"/api/projects/{row.id}"},
                "evidence": [
                    {"term": term, "field": h.field, "snippet": h.snippet}
                    for term, h in hits.items()
                ],
            }
        )
    return matches, len(ranked)
