"""Offline category budgets and directory estimates for already-authorized schemas."""

from dataclasses import dataclass
import json
import math

from ispec.assistant.project_comment_workflow import COMMENT_WORKFLOW_TOOLS
from ispec.assistant.tool_routing import _GROUP_DEFS
from ispec.assistant.tools import ToolScope, _TOOL_SCOPES


STAFF_CATEGORIES = frozenset({"assistant", "devops", "tmux", "bridge", "repo", "automation"})
DIRECTORY_LABELS = {
    "projects": "Projects, status and metadata.",
    "notes": "Drafting, saving or cancelling project notes.",
    "files": "Project files and attachments.",
    "experiments": "Experiments, runs and searches.",
    "omics": "Project gene quantities and QC.",
    "people": "People and institutions.",
    "schedule": "Scheduling slots and requests.",
    "repo": "Source repository inspection.",
    "devops": "Service operations and staff notifications.",
    "automation": "Recurring assistant jobs.",
    "tmux": "Terminal sessions and pane contents.",
    "bridge": "Slack-to-terminal communication.",
    "assistant": "Conversation history and agent records.",
    "misc": "API discovery and other utilities.",
}


@dataclass(frozen=True, kw_only=True)
class ToolCatalog:
    schemas: dict[str, dict]
    groups: dict[str, frozenset[str]]

    @classmethod
    def from_authorized_schemas(cls, *, schemas: list[dict], staff: bool) -> "ToolCatalog":
        available = {
            spec["function"]["name"]: spec for spec in schemas
            if spec["function"]["name"] != "assistant_list_tools"
            and (staff or _TOOL_SCOPES.get(spec["function"]["name"], ToolScope.staff) in {ToolScope.public, ToolScope.user})
        }
        groups = {}
        for group in _GROUP_DEFS:
            if not staff and group.name in STAFF_CATEGORIES:
                continue
            names = set(group.tool_names)
            if group.name == "projects":
                names.update(COMMENT_WORKFLOW_TOOLS)
            authorized = frozenset(names & available.keys())
            if authorized:
                groups[group.name] = authorized
        return cls(schemas=available, groups=groups)

    def open_category(self, *, name: str) -> tuple[dict, ...]:
        """Resolve only an advertised category from the supplied authorized catalog."""
        if name not in self.groups:
            raise ValueError(f"Unavailable tool category: {name}")
        return tuple(self.schemas[tool] for tool in sorted(self.groups[name]))

    def directory_schema(self) -> dict:
        listing = "\n".join(f"{name}: {DIRECTORY_LABELS[name]}" for name in self.groups)
        return {
            "type": "function",
            "function": {
                "name": "open_tool_category",
                "description": (
                    "Reveal an available category's tool schemas for the next step. "
                    "Choose a category only when its tools are needed.\n" + listing
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": list(self.groups)}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }


@dataclass(frozen=True, kw_only=True)
class SelectionBudget:
    cumulative_probability: float
    max_categories: int
    max_tools: int

    def __post_init__(self) -> None:
        if not 0 < self.cumulative_probability <= 1 or self.max_categories < 1 or self.max_tools < 1:
            raise ValueError("Use probability in (0, 1] and positive category/tool limits")


@dataclass(frozen=True, kw_only=True)
class CategorySelection:
    categories: tuple[str, ...]
    probability_mass: float
    tool_names: tuple[str, ...]
    stop_reason: str

    @property
    def tool_count(self) -> int:
        return len(self.tool_names) + 1  # The directory is always present.


def select_categories(
    *, scores: dict[str, float], catalog: ToolCatalog, budget: SelectionBudget,
    always_tool_names: frozenset[str] = frozenset(),
) -> CategorySelection:
    """Accumulate ranked categories until mass or a catalog budget stops selection."""
    if not scores or any(not math.isfinite(value) or value < 0 for value in scores.values()):
        raise ValueError("Expected finite nonnegative probabilities")
    if not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-5):
        raise ValueError("Category probabilities must sum to one")
    if "none" not in scores:
        raise ValueError("The none choice is required")
    if set(scores) - (catalog.groups.keys() | {"none"}):
        raise ValueError("Filter unavailable categories before scoring")
    if always_tool_names - catalog.schemas.keys():
        raise ValueError("Always-present tools must already be authorized")
    ranked = sorted(scores, key=scores.get, reverse=True)

    categories, tools, mass = [], set(always_tool_names), 0.0
    reason = "available_categories_exhausted"
    for name in ranked:
        if name != "none":
            if len(categories) >= budget.max_categories:
                reason = "category_cap"
                break
            proposed_tools = tools | catalog.groups[name]
            # Always retain the top eligible category; cap only additional groups.
            if categories and len(proposed_tools) + 1 > budget.max_tools:
                reason = "tool_cap"
                break
            categories.append(name)
            tools = proposed_tools
        mass += scores[name]
        if mass >= budget.cumulative_probability or math.isclose(mass, budget.cumulative_probability, abs_tol=1e-9):
            reason = "probability_target"
            break
    return CategorySelection(
        categories=tuple(categories), probability_mass=mass,
        tool_names=tuple(sorted(tools)), stop_reason=reason,
    )


def selection_tokens(*, catalog: ToolCatalog, selection: CategorySelection, tokenizer) -> int:
    """Count compact schema JSON tokens, including the directory, with the real tokenizer."""
    schemas = [catalog.directory_schema(), *(catalog.schemas[name] for name in selection.tool_names)]
    text = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    return len(tokenizer.encode(text, add_special_tokens=False).ids)
