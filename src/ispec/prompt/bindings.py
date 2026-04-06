from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import PromptBindingMeta


def prompt_binding(family: str, *, kind: str = "wrapper") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    cleaned_family = str(family or "").strip()
    cleaned_kind = str(kind or "").strip() or "wrapper"

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, "__prompt_family__", cleaned_family)
        setattr(fn, "__prompt_binding_kind__", cleaned_kind)
        return fn

    return decorator


def prompt_family_for(binding: Callable[..., Any]) -> str:
    family = str(getattr(binding, "__prompt_family__", "") or "").strip()
    if not family:
        raise KeyError(f"Callable {binding!r} is not bound to a prompt family.")
    return family


def binding_meta_for_callable(binding: Callable[..., Any]) -> PromptBindingMeta:
    family = prompt_family_for(binding)
    module = str(getattr(binding, "__module__", "") or "").strip()
    qualname = str(getattr(binding, "__qualname__", getattr(binding, "__name__", "")) or "").strip()
    binding_kind = str(getattr(binding, "__prompt_binding_kind__", "") or "wrapper").strip() or "wrapper"
    source_file = inspect.getsourcefile(binding)
    source_line: int | None = None
    try:
        _lines, source_line = inspect.getsourcelines(binding)
    except Exception:
        source_line = None
    return PromptBindingMeta(
        family=family,
        module=module,
        qualname=qualname,
        binding_kind=binding_kind,
        source_file=source_file,
        source_line=source_line,
    )


def _decorator_family(decorator: ast.AST) -> tuple[str, str] | None:
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    func_name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
    if func_name != "prompt_binding":
        return None
    if not decorator.args:
        return None
    first_arg = decorator.args[0]
    if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
        return None
    family = first_arg.value.strip()
    if not family:
        return None
    kind = "wrapper"
    for keyword in decorator.keywords:
        if keyword.arg == "kind" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            kind = keyword.value.value.strip() or "wrapper"
    return family, kind


def discover_prompt_bindings_ast(*, source_root: str | Path) -> list[PromptBindingMeta]:
    root = Path(source_root).expanduser().resolve()
    bindings: list[PromptBindingMeta] = []
    for path in sorted(root.rglob("*.py")):
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(raw, filename=str(path))
        except SyntaxError:
            continue
        rel = path.relative_to(root)
        module = ".".join(rel.with_suffix("").parts)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                match = _decorator_family(decorator)
                if match is None:
                    continue
                family, kind = match
                bindings.append(
                    PromptBindingMeta(
                        family=family,
                        module=module,
                        qualname=node.name,
                        binding_kind=kind,
                        source_file=str(path),
                        source_line=int(getattr(node, "lineno", 0) or 0) or None,
                    )
                )
                break
    return bindings
