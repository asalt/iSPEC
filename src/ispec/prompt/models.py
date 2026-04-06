from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptSource:
    family: str
    source_path: str
    title: str | None
    notes: str | None
    body: str
    body_sha256: str


@dataclass(frozen=True)
class PromptBindingMeta:
    family: str
    module: str
    qualname: str
    binding_kind: str = "wrapper"
    source_file: str | None = None
    source_line: int | None = None

    @property
    def binding_ref(self) -> str:
        module = (self.module or "").strip()
        qualname = (self.qualname or "").strip()
        if module and qualname:
            return f"{module}:{qualname}"
        return module or qualname


@dataclass(frozen=True)
class PromptVersionInfo:
    version_id: int | None = None
    version_num: int | None = None


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    source: PromptSource
    binding: PromptBindingMeta | None = None
    version: PromptVersionInfo = PromptVersionInfo()

    def observability_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt_family": self.source.family,
            "prompt_sha256": self.source.body_sha256,
            "prompt_source_path": self.source.source_path,
        }
        if self.binding is not None:
            payload["prompt_binding"] = self.binding.binding_ref
        if self.version.version_num is not None:
            payload["prompt_version_num"] = int(self.version.version_num)
        if self.version.version_id is not None:
            payload["prompt_version_id"] = int(self.version.version_id)
        return payload
