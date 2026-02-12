from __future__ import annotations

import ast
import math
from functools import lru_cache
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from ispec.agent.policy_schema import stable_hash

from .jit import jit_if_available

ExprTier = Literal["basic", "extended"]


def _min2(a: float, b: float) -> float:
    return float(a) if a < b else float(b)


def _max2(a: float, b: float) -> float:
    return float(a) if a > b else float(b)


def clamp_current(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def ifelse_current(cond: Any, a: float, b: float) -> float:
    return float(a) if bool(cond) else float(b)


def piecewise_current(*args: Any) -> Any:
    """Return the first value whose condition is truthy, else the default.

    Arguments are interpreted as (cond1, val1, cond2, val2, ..., default).
    """

    if len(args) < 3:
        raise ValueError("piecewise requires at least (cond, value, default)")
    if len(args) % 2 == 0:
        raise ValueError("piecewise requires an odd number of args: (cond, value)* + default")

    for idx in range(0, len(args) - 1, 2):
        if bool(args[idx]):
            return args[idx + 1]
    return args[-1]


_SAFE_CONSTANTS: dict[str, float] = {
    "pi": float(math.pi),
    "e": float(math.e),
    "ln2": float(math.log(2.0)),
}


def _safe_functions_for_tier(tier: ExprTier) -> dict[str, Any]:
    funcs: dict[str, Any] = {
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "abs": abs,
    }
    if tier == "extended":
        funcs.update(
            {
                "min": _min2,
                "max": _max2,
                "clamp": clamp_current,
                "ifelse": ifelse_current,
                "piecewise": piecewise_current,
            }
        )
    return funcs


_BANNED_NAME_PREFIXES = ("__", "_")
_BANNED_NAMES = {"None", "True", "False"}


class ExprKernelSpec(BaseModel):
    expr: str = Field(min_length=1, max_length=20_000)
    args: tuple[str, ...] = Field(default_factory=tuple)
    tier: ExprTier = "basic"

    @model_validator(mode="after")
    def _validate_spec(self) -> "ExprKernelSpec":
        expr = (self.expr or "").strip()
        if not expr:
            raise ValueError("expr must be non-empty")

        if not self.args:
            raise ValueError("args must be non-empty")

        seen: set[str] = set()
        for name in self.args:
            raw = str(name).strip()
            if not raw:
                raise ValueError("arg names must be non-empty")
            if raw in _BANNED_NAMES:
                raise ValueError(f"arg name is reserved: {raw}")
            if raw.startswith(_BANNED_NAME_PREFIXES):
                raise ValueError(f"arg name cannot start with '_' or '__': {raw}")
            if not raw.isidentifier():
                raise ValueError(f"arg name is not a valid identifier: {raw}")
            if raw in seen:
                raise ValueError(f"duplicate arg name: {raw}")
            seen.add(raw)

        _validate_expr(expr, args=set(self.args), tier=self.tier)
        return self

    def spec_id(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


def _validate_expr(expr: str, *, args: set[str], tier: ExprTier) -> None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc}") from exc

    allowed_funcs = set(_safe_functions_for_tier(tier).keys())
    allowed_names = set(args) | set(_SAFE_CONSTANTS.keys()) | allowed_funcs

    allow_comparisons = tier == "extended"
    allow_bool = tier == "extended"

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.Mod,
                ast.UAdd,
                ast.USub,
                ast.And,
                ast.Or,
                ast.Not,
                ast.Lt,
                ast.LtE,
                ast.Gt,
                ast.GtE,
                ast.Eq,
                ast.NotEq,
            ),
        ):
            continue
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.Load):
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                continue
            raise ValueError(f"unsupported constant type: {type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                continue
            raise ValueError(f"unknown name: {node.id}")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, (ast.UAdd, ast.USub)):
                continue
            if allow_bool and isinstance(node.op, ast.Not):
                continue
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                continue
            raise ValueError(f"unsupported binary operator: {type(node.op).__name__}")
        if isinstance(node, ast.BoolOp):
            if not allow_bool:
                raise ValueError("boolean operators are not allowed in basic tier")
            if isinstance(node.op, (ast.And, ast.Or)):
                continue
            raise ValueError(f"unsupported boolean operator: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            if not allow_comparisons:
                raise ValueError("comparisons are not allowed in basic tier")
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("chained comparisons are not supported")
            op = node.ops[0]
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                continue
            raise ValueError(f"unsupported comparison operator: {type(op).__name__}")
        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            func = node.func
            if not isinstance(func, ast.Name):
                raise ValueError("only direct function calls are allowed (no attributes)")
            if func.id not in allowed_funcs:
                raise ValueError(f"call to unsupported function: {func.id}")
            continue

        # Anything else is disallowed: Attribute/Subscript/Lambda/Comprehensions, etc.
        raise ValueError(f"unsupported expression node: {type(node).__name__}")


@lru_cache(maxsize=None)
def _compile_expr_kernel_cached(
    spec_json: str,
    *,
    backend: str,
    jit_cache: bool,
    jit_fastmath: bool,
) -> Callable[..., Any]:
    spec = ExprKernelSpec.model_validate_json(spec_json)
    fn_name = f"expr_kernel_{spec.spec_id()}"
    args = ", ".join(spec.args)
    source = f"def {fn_name}({args}):\n    return {spec.expr}\n"
    code = compile(source, f"<ispec.expr:{fn_name}>", "exec")

    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_globals.update(_SAFE_CONSTANTS)
    safe_globals.update(_safe_functions_for_tier(spec.tier))

    namespace: dict[str, Any] = {}
    exec(code, safe_globals, namespace)
    fn = namespace[fn_name]

    if backend == "numba":
        return jit_if_available(fn, cache=jit_cache, fastmath=jit_fastmath)
    if backend == "python":
        return fn
    if backend == "auto":
        return jit_if_available(fn, cache=jit_cache, fastmath=jit_fastmath)
    raise ValueError(f"unknown backend: {backend}")


def compile_expr_kernel_current(
    spec: ExprKernelSpec,
    *,
    backend: Literal["auto", "python", "numba"] = "auto",
    jit_cache: bool = False,
    jit_fastmath: bool = False,
) -> Callable[..., Any]:
    """Compile a validated expression into a callable.

    Security model:
    - expression is parsed and validated via a strict AST whitelist
    - compiled function executes with ``__builtins__ = {}`` and only a small
      safe registry of constants/functions injected
    """

    spec_json = spec.model_dump_json()
    return _compile_expr_kernel_cached(
        spec_json,
        backend=str(backend),
        jit_cache=bool(jit_cache),
        jit_fastmath=bool(jit_fastmath),
    )
