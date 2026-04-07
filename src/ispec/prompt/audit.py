from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


_KEYWORDS = (
    "You are ",
    "Response format:",
    "Output only",
    "Return only",
    "Tool use",
    "Tool calling protocol:",
    "Do not call tools.",
)


@dataclass(frozen=True)
class InlinePromptFinding:
    source_file: str
    line: int
    char_count: int
    newline_count: int
    preview: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "line": self.line,
            "char_count": self.char_count,
            "newline_count": self.newline_count,
            "preview": self.preview,
        }


def _literal_string_value(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("{expr}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string_value(node.left)
        right = _literal_string_value(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def _is_docstring_node(node: ast.AST, *, parent_map: dict[ast.AST, ast.AST]) -> bool:
    parent = parent_map.get(node)
    if not isinstance(parent, ast.Expr):
        return False
    owner = parent_map.get(parent)
    if not isinstance(owner, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    body = getattr(owner, "body", None)
    return bool(body) and body[0] is parent


def _is_nested_string_expr(node: ast.AST, *, parent_map: dict[ast.AST, ast.AST]) -> bool:
    parent = parent_map.get(node)
    return isinstance(parent, (ast.JoinedStr, ast.BinOp))


def audit_inline_prompt_literals(
    *,
    source_root: str | Path,
    min_chars: int = 160,
    min_newlines: int = 4,
) -> list[InlinePromptFinding]:
    root = Path(source_root).expanduser().resolve()
    findings: list[InlinePromptFinding] = []
    for path in sorted(root.rglob("*.py")):
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(raw, filename=str(path))
        except SyntaxError:
            continue
        parent_map: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
                continue
            if _is_nested_string_expr(node, parent_map=parent_map):
                continue
            if _is_docstring_node(node, parent_map=parent_map):
                continue
            text = _literal_string_value(node)
            if not text:
                continue
            stripped = text.strip()
            newline_count = stripped.count("\n")
            if len(stripped) < int(min_chars) or newline_count < int(min_newlines):
                continue
            if not any(keyword in stripped for keyword in _KEYWORDS):
                continue
            preview = " ".join(part.strip() for part in stripped.splitlines()[:3] if part.strip())[:200]
            findings.append(
                InlinePromptFinding(
                    source_file=str(path),
                    line=int(getattr(node, "lineno", 0) or 0),
                    char_count=len(stripped),
                    newline_count=newline_count,
                    preview=preview,
                )
            )
    findings.sort(key=lambda item: (item.source_file, item.line))
    return findings
