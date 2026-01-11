from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Literal


Profile = Literal["dev", "prod"]
VarKind = Literal["string", "bool", "int", "path", "url", "csv"]


@dataclass(frozen=True)
class GeneratorSpec:
    kind: Literal["token_urlsafe"]
    nbytes: int = 32


@dataclass(frozen=True)
class RequiredIf:
    key: str
    equals: str


@dataclass(frozen=True)
class VarSpec:
    key: str
    kind: VarKind
    group: str
    description: str
    required_in: frozenset[Profile] = field(default_factory=frozenset)
    recommended_in: frozenset[Profile] = field(default_factory=frozenset)
    required_if: RequiredIf | None = None
    default_by_profile: dict[Profile, str] = field(default_factory=dict)
    choices: tuple[str, ...] = ()
    secret: bool = False
    generator: GeneratorSpec | None = None
    min_value: int | None = None
    max_value: int | None = None
    min_length: int | None = None
    forbid_values: tuple[str, ...] = ()

    def default_for(self, profile: Profile) -> str | None:
        raw = (self.default_by_profile.get(profile) or "").strip()
        return raw or None


def spec_to_dict(spec: VarSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "key": spec.key,
        "kind": spec.kind,
        "group": spec.group,
        "description": spec.description,
        "required_in": sorted(spec.required_in),
        "recommended_in": sorted(spec.recommended_in),
        "required_if": None,
        "default_by_profile": dict(spec.default_by_profile),
        "choices": list(spec.choices),
        "secret": bool(spec.secret),
        "generator": None,
        "min_value": spec.min_value,
        "max_value": spec.max_value,
        "min_length": spec.min_length,
        "forbid_values": list(spec.forbid_values),
    }
    if spec.required_if is not None:
        payload["required_if"] = {"key": spec.required_if.key, "equals": spec.required_if.equals}
    if spec.generator is not None:
        payload["generator"] = {"kind": spec.generator.kind, "nbytes": spec.generator.nbytes}
    return payload


def default_contract() -> tuple[VarSpec, ...]:
    """Contract for env vars used by the full-stack wrapper deploy."""

    return (
        VarSpec(
            key="ISPEC_HOST",
            kind="string",
            group="Deploy",
            description="Host interface for the backend API server (Makefile).",
            default_by_profile={"dev": "0.0.0.0", "prod": "0.0.0.0"},
        ),
        VarSpec(
            key="ISPEC_PORT",
            kind="int",
            group="Deploy",
            description="Port for the backend API server (Makefile).",
            default_by_profile={"dev": "3001", "prod": "3001"},
            min_value=1,
            max_value=65535,
        ),
        VarSpec(
            key="UI_HOST",
            kind="string",
            group="Deploy",
            description="Host interface for the frontend dev server (Makefile).",
            default_by_profile={"dev": "0.0.0.0", "prod": "0.0.0.0"},
        ),
        VarSpec(
            key="UI_PORT",
            kind="int",
            group="Deploy",
            description="Port for the frontend dev server (Makefile).",
            default_by_profile={"dev": "3000", "prod": "3000"},
            min_value=1,
            max_value=65535,
        ),
        VarSpec(
            key="ISPEC_DB_PATH",
            kind="path",
            group="Database",
            description="Primary SQLite DB path or sqlite:/// URI.",
            default_by_profile={"dev": "iSPEC/data/ispec-import.db", "prod": "/var/lib/ispec/ispec.db"},
        ),
        VarSpec(
            key="ISPEC_SCHEDULE_DB_PATH",
            kind="path",
            group="Database",
            description="Schedule SQLite DB path or sqlite:/// URI (defaults next to ISPEC_DB_PATH).",
            default_by_profile={"dev": "iSPEC/data/ispec-schedule.db", "prod": "/var/lib/ispec/ispec-schedule.db"},
        ),
        VarSpec(
            key="ISPEC_OMICS_DB_PATH",
            kind="path",
            group="Database",
            description="Omics SQLite DB path or sqlite:/// URI (defaults next to ISPEC_DB_PATH).",
            default_by_profile={"dev": "iSPEC/data/ispec-omics.db", "prod": "/var/lib/ispec/ispec-omics.db"},
        ),
        VarSpec(
            key="ISPEC_ASSISTANT_DB_PATH",
            kind="path",
            group="Database",
            description="Support assistant SQLite DB path or sqlite:/// URI (defaults next to ISPEC_DB_PATH).",
            default_by_profile={"dev": "iSPEC/data/ispec-assistant.db", "prod": "/var/lib/ispec/ispec-assistant.db"},
        ),
        VarSpec(
            key="ISPEC_AGENT_DB_PATH",
            kind="path",
            group="Database",
            description="Agent telemetry SQLite DB path or sqlite:/// URI (defaults next to ISPEC_DB_PATH).",
            default_by_profile={"dev": "iSPEC/data/ispec-agent.db", "prod": "/var/lib/ispec/ispec-agent.db"},
        ),
        VarSpec(
            key="ISPEC_API_KEY",
            kind="string",
            group="Security",
            description="Shared API key required for non-local binds and recommended for all deployments.",
            required_in=frozenset({"prod"}),
            recommended_in=frozenset({"dev"}),
            secret=True,
            generator=GeneratorSpec(kind="token_urlsafe", nbytes=32),
            min_length=20,
            forbid_values=("change-me",),
        ),
        VarSpec(
            key="ISPEC_API_RESOURCES",
            kind="string",
            group="Security",
            description="Comma-separated list of API resources to expose, or 'all'/'*'.",
            default_by_profile={"dev": "all", "prod": "all"},
        ),
        VarSpec(
            key="ISPEC_REQUIRE_LOGIN",
            kind="bool",
            group="Auth",
            description="Require cookie login for protected CRUD routes.",
            required_in=frozenset({"prod"}),
            default_by_profile={"dev": "1", "prod": "1"},
        ),
        VarSpec(
            key="ISPEC_PASSWORD_PEPPER",
            kind="string",
            group="Auth",
            description="Server-side pepper for PBKDF2 password hashing (must stay stable once users exist).",
            required_in=frozenset({"prod"}),
            recommended_in=frozenset({"dev"}),
            secret=True,
            generator=GeneratorSpec(kind="token_urlsafe", nbytes=32),
            min_length=20,
            forbid_values=("dev-change-me", "change-me", "admin"),
        ),
        VarSpec(
            key="ISPEC_PASSWORD_ITERATIONS",
            kind="int",
            group="Auth",
            description="PBKDF2 iterations (>= 50k; defaults to 250k).",
            default_by_profile={"dev": "250000", "prod": "250000"},
            min_value=50_000,
        ),
        VarSpec(
            key="ISPEC_DEV_DEFAULT_ADMIN",
            kind="bool",
            group="Auth",
            description="DEV ONLY: auto-create admin user on startup when DB has no users.",
            default_by_profile={"dev": "1", "prod": "0"},
        ),
        VarSpec(
            key="ISPEC_DEV_ADMIN_USERNAME",
            kind="string",
            group="Auth",
            description="DEV ONLY: default admin username (when ISPEC_DEV_DEFAULT_ADMIN=1).",
            default_by_profile={"dev": "admin", "prod": ""},
        ),
        VarSpec(
            key="ISPEC_DEV_ADMIN_PASSWORD",
            kind="string",
            group="Auth",
            description="DEV ONLY: default admin password (when ISPEC_DEV_DEFAULT_ADMIN=1).",
            default_by_profile={"dev": "admin", "prod": ""},
            secret=True,
        ),
        VarSpec(
            key="ISPEC_ASSISTANT_PROVIDER",
            kind="string",
            group="Assistant",
            description="Provider for /api/support/chat: stub | ollama | vllm",
            default_by_profile={"dev": "vllm", "prod": "vllm"},
            choices=("stub", "ollama", "vllm"),
        ),
        VarSpec(
            key="ISPEC_ASSISTANT_TOOL_PROTOCOL",
            kind="string",
            group="Assistant",
            description="Tool calling protocol: line (TOOL_CALL) | openai (structured tool_calls).",
            default_by_profile={"dev": "openai", "prod": "openai"},
            choices=("line", "openai"),
        ),
        VarSpec(
            key="ISPEC_VLLM_URL",
            kind="url",
            group="Assistant",
            description="Base URL for vLLM OpenAI-compatible server.",
            required_if=RequiredIf(key="ISPEC_ASSISTANT_PROVIDER", equals="vllm"),
            default_by_profile={"dev": "http://127.0.0.1:8000", "prod": "http://127.0.0.1:8000"},
        ),
        VarSpec(
            key="ISPEC_VLLM_MODEL",
            kind="string",
            group="Assistant",
            description="vLLM model name; if omitted, iSPEC queries /v1/models.",
            required_if=None,
            default_by_profile={"dev": "", "prod": ""},
        ),
        VarSpec(
            key="ISPEC_VLLM_API_KEY",
            kind="string",
            group="Assistant",
            description="Optional API key for vLLM server (sent as Authorization: Bearer ...).",
            secret=True,
        ),
        VarSpec(
            key="ISPEC_OLLAMA_URL",
            kind="url",
            group="Assistant",
            description="Base URL for Ollama server.",
            required_if=RequiredIf(key="ISPEC_ASSISTANT_PROVIDER", equals="ollama"),
            default_by_profile={"dev": "http://127.0.0.1:11434", "prod": "http://127.0.0.1:11434"},
        ),
        VarSpec(
            key="ISPEC_OLLAMA_MODEL",
            kind="string",
            group="Assistant",
            description="Ollama model name.",
            required_if=RequiredIf(key="ISPEC_ASSISTANT_PROVIDER", equals="ollama"),
            default_by_profile={"dev": "llama3.2:2b", "prod": "llama3.2:2b"},
        ),
    )
