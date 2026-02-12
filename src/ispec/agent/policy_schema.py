from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

JSON = dict[str, Any]


def canon_json(x: Any) -> str:
    """Stable JSON for hashing + diffs. Keep this consistent forever."""

    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_hash(x: Any, *, n: int = 16) -> str:
    """Short stable hash of canonical JSON."""

    return sha256_hex(canon_json(x))[:n]


def _source_for_callable(obj: Any) -> str | None:
    try:
        source = inspect.getsource(obj)
    except Exception:
        return None
    source = textwrap.dedent(source).strip()
    return source or None


def callable_id_from_callable(
    obj: Any,
    *,
    mode: Literal["source", "code", "fallback"] = "source",
) -> "CallableId":
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or str(obj)
    module = getattr(obj, "__module__", None) or ""

    if mode == "source":
        source = _source_for_callable(obj)
        if source is not None:
            return CallableId(
                mode="source",
                qualname=str(qualname),
                digest=sha256_hex(source),
                extras={"module": str(module)},
            )
        mode = "code"

    if mode == "code":
        code = getattr(obj, "__code__", None)
        if code is not None:
            try:
                sanitized = code.replace(co_filename="")
            except Exception:
                sanitized = code
            payload = marshal.dumps(sanitized)
            return CallableId(
                mode="code",
                qualname=str(qualname),
                digest=hashlib.sha256(payload).hexdigest(),
                extras={
                    "module": str(module),
                    "firstlineno": int(getattr(code, "co_firstlineno", 0) or 0),
                },
            )
        mode = "fallback"

    file_hint = ""
    try:
        file_hint = str(inspect.getfile(obj))
    except Exception:
        file_hint = ""

    file_stat: dict[str, Any] = {}
    if file_hint:
        try:
            path = Path(file_hint)
            stat = path.stat()
            file_stat = {"path": path.name, "mtime": int(stat.st_mtime), "size": int(stat.st_size)}
        except Exception:
            file_stat = {"path": Path(file_hint).name}

    digest_input = canon_json({"module": module, "qualname": qualname, "file": file_stat})
    return CallableId(
        mode="fallback",
        qualname=str(qualname),
        digest=sha256_hex(digest_input),
        extras={"module": str(module), "file": file_stat},
    )


@dataclass(frozen=True)
class CallableId:
    """
    Identity for "what code ran".

    - mode='code': derived from __code__ (version-sensitive but local-repro friendly)
    - mode='source': derived from inspect.getsource when available
    - mode='fallback': module:qualname + file mtime, etc.
    """

    mode: Literal["code", "source", "fallback"]
    qualname: str
    digest: str
    extras: JSON = field(default_factory=dict)

    def id(self) -> str:
        return stable_hash(
            {
                "mode": self.mode,
                "qualname": self.qualname,
                "digest": self.digest,
                "extras": self.extras,
            }
        )


@dataclass(frozen=True)
class ParamSet:
    """Identity for "with what parameters". Keep params JSON-serializable."""

    name: str
    params: Mapping[str, Any]

    def param_hash(self) -> str:
        return stable_hash({"params": dict(self.params)})

    def id(self) -> str:
        return stable_hash({"name": self.name, "params": dict(self.params)})


@dataclass(frozen=True)
class PolicyRef:
    """Serializable pointer to a callable + semantics."""

    kind: str
    callable_id: CallableId
    schema_version: str = "1"

    def id(self) -> str:
        return stable_hash(
            {
                "kind": self.kind,
                "callable_id": self.callable_id.id(),
                "schema_version": self.schema_version,
            }
        )


@dataclass(frozen=True)
class PolicySpec:
    """Fully-serializable policy spec: what it is + how it's configured."""

    ref: PolicyRef
    param_set: ParamSet
    tags: Mapping[str, Any] = field(default_factory=dict)

    def policy_id(self) -> str:
        return stable_hash(
            {
                "ref": {
                    "kind": self.ref.kind,
                    "callable_id": self.ref.callable_id.id(),
                    "schema_version": self.ref.schema_version,
                },
                "param_set_id": self.param_set.id(),
                "tags": dict(self.tags),
            }
        )


@dataclass(frozen=True)
class ComposeSpec:
    """Composition is structural: output = modifiers(...(base(...)))."""

    base: PolicySpec
    modifiers: tuple[PolicySpec, ...] = ()

    def policy_id(self) -> str:
        return stable_hash(
            {"base": self.base.policy_id(), "modifiers": [m.policy_id() for m in self.modifiers]}
        )


@dataclass
class PolicyState:
    """Minimal state to support stepwise iteration and resume/replay."""

    step: int = 0
    data: JSON = field(default_factory=dict)

    def state_hash(self) -> str:
        return stable_hash({"step": self.step, "data": self.data})


@dataclass(frozen=True)
class InputSignature:
    """Identifies inputs/args/context. Keep small; reference large blobs by hash/pointer."""

    inputs: Mapping[str, Any] = field(default_factory=dict)
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)

    def input_hash(self) -> str:
        return stable_hash(
            {
                "inputs": dict(self.inputs),
                "args": list(self.args),
                "kwargs": dict(self.kwargs),
                "context": dict(self.context),
            }
        )


@dataclass(frozen=True)
class CacheKey:
    """Caching key for a policy step decision."""

    policy_id: str
    input_hash: str
    state_hash: str
    step: int
    namespace: str = "policy_step"

    def key(self) -> str:
        return stable_hash(
            {
                "namespace": self.namespace,
                "policy_id": self.policy_id,
                "input_hash": self.input_hash,
                "state_hash": self.state_hash,
                "step": self.step,
            }
        )


@dataclass(frozen=True)
class DecisionOutput:
    """What happened. Keep results small; reference large artifacts by hash/pointer."""

    result: Mapping[str, Any]
    terms: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def output_hash(self) -> str:
        return stable_hash(
            {
                "result": dict(self.result),
                "terms": dict(self.terms),
                "metrics": dict(self.metrics),
            }
        )


@dataclass(frozen=True)
class DecisionEvent:
    """Record for replay/audit: what decision was made and with what state/inputs."""

    run_id: str
    step_id: str
    ts_unix: float

    compose_id: str
    policy_id: str

    cache_key: str
    cache_hit: bool

    policy_state_before: PolicyState
    policy_state_after: PolicyState

    input_sig: InputSignature
    output: DecisionOutput

    tags: Mapping[str, Any] = field(default_factory=dict)

    def event_id(self) -> str:
        return stable_hash(
            {
                "run_id": self.run_id,
                "step_id": self.step_id,
                "ts_unix": self.ts_unix,
                "cache_key": self.cache_key,
                "compose_id": self.compose_id,
            }
        )


@dataclass(frozen=True)
class PolicyRegistryRecord:
    """Discovery/governance record of a policy existing in the system."""

    ref: PolicyRef
    description: str = ""
    owner: str = ""
    created_ts_unix: float | None = None


@dataclass(frozen=True)
class ParamRegistryRecord:
    """Discovery record of a parameter set existing in the system."""

    param_set: ParamSet
    description: str = ""
    created_ts_unix: float | None = None

