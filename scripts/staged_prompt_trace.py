"""Read saved wire evidence once; expose named messages and template reconstructions."""

from dataclasses import dataclass
from functools import lru_cache, partial
import hashlib
import json
from pathlib import Path

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class FunctionCall(EvidenceRecord):
    name: str
    arguments: str | dict


class ToolCall(EvidenceRecord):
    id: str = ""
    type: str = "function"
    function: FunctionCall

    def template_value(self) -> dict:
        arguments = self.function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object for template reconstruction")
        return {"id": self.id, "type": self.type,
                "function": {"name": self.function.name, "arguments": arguments}}


class WireMessage(EvidenceRecord):
    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning: str | None = None

    def template_value(self) -> dict:
        value = self.model_dump(exclude_unset=True)
        value["content"] = self.content or ""
        if self.tool_calls:
            value["tool_calls"] = [call.template_value() for call in self.tool_calls]
        else:
            value.pop("tool_calls", None)
        return value


class ToolDefinitionFunction(EvidenceRecord):
    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)


class ToolDefinition(EvidenceRecord):
    type: str = "function"
    function: ToolDefinitionFunction


class WireRequest(EvidenceRecord):
    model: str = "unknown"
    messages: tuple[WireMessage, ...] = ()
    tools: tuple[ToolDefinition, ...] | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template: str | None = None
    chat_template_kwargs: dict = Field(default_factory=dict)

    @property
    def latest_user_text(self) -> str | None:
        return next((message.content for message in reversed(self.messages) if message.role == "user"), None)


class WireChoice(EvidenceRecord):
    message: WireMessage
    finish_reason: str | None = None


class WireResponse(EvidenceRecord):
    choices: tuple[WireChoice, ...] = ()


class InferenceRecord(EvidenceRecord):
    path: str
    request: WireRequest
    response: WireResponse | None = None
    error: str | dict | None = None
    elapsed_ms: float | None = None


class WireCapture(EvidenceRecord):
    calls: tuple[InferenceRecord, ...]

    @classmethod
    def read(cls, *, path: Path) -> "WireCapture":
        return cls.model_validate_json(path.read_text())


class TemplateSource(EvidenceRecord):
    tokenizer_config: Path
    tokenizer_sha256: str
    chat_template: Path | None = None
    template_sha256: str
    provenance: str


@dataclass(frozen=True, kw_only=True)
class TemplateProfile:
    template: str
    bos_token: str
    eos_token: str
    source: str
    sha256: str
    provenance: str


@dataclass(frozen=True, kw_only=True)
class RenderedPrompt:
    text: str | None
    explanation: str


def load_profiles(*, path: Path | None) -> dict[str, TemplateProfile]:
    if path is None:
        return {}
    profiles = {}
    for model, value in json.loads(path.read_text()).items():
        spec = TemplateSource.model_validate(value)
        config_bytes = spec.tokenizer_config.read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != spec.tokenizer_sha256:
            raise ValueError(f"Tokenizer fingerprint mismatch: {model}")
        config = json.loads(config_bytes)
        template = spec.chat_template.read_text() if spec.chat_template else config["chat_template"]
        digest = hashlib.sha256(template.encode()).hexdigest()
        if digest != spec.template_sha256:
            raise ValueError(f"Template fingerprint mismatch: {model}")
        profiles[model] = TemplateProfile(
            template=template, bos_token=config.get("bos_token", ""), eos_token=config.get("eos_token", ""),
            source=str(spec.chat_template or spec.tokenizer_config), sha256=digest, provenance=spec.provenance,
        )
    return profiles


@lru_cache(maxsize=8)
def compiled_template(template: str):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    # Match the Transformers tojson filter rather than Jinja's HTML-escaping filter.
    env.filters["tojson"] = partial(json.dumps, ensure_ascii=False)
    return env.from_string(template)


def render_prompt(*, request: WireRequest, profile: TemplateProfile | None) -> RenderedPrompt:
    if profile is None:
        return RenderedPrompt(text=None, explanation="No verified template mapping supplied for this model. API messages remain available below.")
    if request.continue_final_message or request.chat_template or request.chat_template_kwargs:
        return RenderedPrompt(text=None, explanation="Request-specific template overrides are not reconstructed by this viewer; inspect the preserved API request.")
    try:
        text = compiled_template(profile.template).render(
            messages=[message.template_value() for message in request.messages],
            tools=[tool.model_dump(exclude_unset=True) for tool in request.tools] if request.tools is not None else None,
            bos_token=profile.bos_token, eos_token=profile.eos_token,
            add_generation_prompt=request.add_generation_prompt,
        )
    except (TemplateError, ValueError, TypeError) as error:
        return RenderedPrompt(text=None, explanation=f"Template reconstruction unavailable: {type(error).__name__}: {error}")
    return RenderedPrompt(text=text, explanation=(
        "Reconstructed from the saved API request and the identified template. "
        "Assistant tool arguments are decoded from JSON for template input; absent text becomes empty text. "
        "This is not a captured token stream or a new model run."
    ))
