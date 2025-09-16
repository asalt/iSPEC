#!/usr/bin/env python3
"""Utility to refresh dynamic sections in README.md."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"

TOC_START = "<!-- TOC_START -->"
TOC_END = "<!-- TOC_END -->"
TREE_START = "<!-- PROJECT_TREE_START -->"
TREE_END = "<!-- PROJECT_TREE_END -->"

COMMENT_COLUMN = 38


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the table of contents and project tree in README.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return a non-zero exit code if README.md needs to be regenerated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original = README_PATH.read_text(encoding="utf-8")
    updated = update_readme(original)

    if args.check:
        if updated != original:
            print("README.md is out of date. Run docs/update_readme.py to refresh.")
            return 1
        return 0

    if updated != original:
        README_PATH.write_text(updated, encoding="utf-8")
        print("README.md updated")
    else:
        print("README.md already up to date")

    return 0


def update_readme(text: str) -> str:
    text = update_section(text, TOC_START, TOC_END, build_toc(text))
    text = update_section(text, TREE_START, TREE_END, render_project_tree())
    return text


def update_section(text: str, start: str, end: str, replacement: str) -> str:
    start_idx = text.find(start)
    if start_idx == -1:
        raise ValueError(f"Could not find start marker {start!r} in README.md")

    end_idx = text.find(end, start_idx)
    if end_idx == -1:
        raise ValueError(f"Could not find end marker {end!r} in README.md")

    before = text[: start_idx + len(start)]
    after = text[end_idx:]
    content = replacement.strip("\n")

    if not before.endswith("\n"):
        before += "\n"
    if not after.startswith("\n"):
        after = "\n" + after

    updated = before + content + after
    if not updated.endswith("\n"):
        updated += "\n"
    return updated


def build_toc(text: str) -> str:
    headings: List[tuple[int, str, str]] = []
    inside_code_block = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            inside_code_block = not inside_code_block
            continue
        if inside_code_block or not stripped.startswith("#"):
            continue

        match = re.match(r"^(#{2,6})\s+(.*)$", raw_line)
        if not match:
            continue

        level = len(match.group(1))
        title = match.group(2).strip()

        if title.lower() == "table of contents":
            continue

        headings.append((level, title, slugify(title)))

    toc_lines = []
    for level, title, anchor in headings:
        indent = "  " * (level - 2)
        toc_lines.append(f"{indent}- [{title}](#{anchor})")

    return "\n".join(toc_lines)


def slugify(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"[`~!@#$%^&*()=+[{]}\\|;:'\",<.>/?]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug


@dataclass
class TreeNode:
    label: str
    comment: str | None = None
    children: List["TreeNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.children and not self.label.endswith("/"):
            self.label = f"{self.label}/"


def render_project_tree() -> str:
    root = build_tree()
    lines = [root.label]
    for index, child in enumerate(root.children):
        lines.extend(render_child(child, prefix="", is_last=index == len(root.children) - 1))
    joined = "\n".join(lines)
    return f"```text\n{joined}\n```"


def build_tree() -> TreeNode:
    docs = dir_node("docs", "Generated and hand-written documentation assets")
    sql = dir_node("sql", "SQL initialization scripts")
    src = dir_node(
        "src/ispec",
        "Package source code",
        children=[
            dir_node("src/ispec/api", "FastAPI app, routers, and schema builders", label="api"),
            dir_node("src/ispec/cli", "argparse-powered CLI entry points", label="cli"),
            dir_node(
                "src/ispec/db",
                "Database models, CRUD helpers, and session tooling",
                label="db",
            ),
            dir_node("src/ispec/io", "File import/export utilities", label="io"),
            dir_node("src/ispec/logging", "Logging configuration helpers", label="logging"),
        ],
    )
    tests = dir_node("tests", "Unit and integration tests")
    pyproject = file_node("pyproject.toml", "Project metadata and dependency declarations")
    readme = file_node("README.md", "This guide")

    return TreeNode(
        "iSPEC",
        children=[docs, sql, src, tests, pyproject, readme],
    )


def dir_node(rel_path: str, comment: str | None = None, label: str | None = None, children: Iterable[TreeNode] | None = None) -> TreeNode:
    path = ROOT / rel_path
    if not path.is_dir():
        raise FileNotFoundError(f"Expected directory at {rel_path}")
    node_label = label if label is not None else Path(rel_path).as_posix()
    if not node_label.endswith("/"):
        node_label = f"{node_label}/"
    return TreeNode(node_label, comment=comment, children=list(children or []))


def file_node(rel_path: str, comment: str | None = None, label: str | None = None) -> TreeNode:
    path = ROOT / rel_path
    if not path.is_file():
        raise FileNotFoundError(f"Expected file at {rel_path}")
    node_label = label if label is not None else Path(rel_path).name
    return TreeNode(node_label, comment=comment)


def render_child(node: TreeNode, *, prefix: str, is_last: bool) -> List[str]:
    connector = "└── " if is_last else "├── "
    line = f"{prefix}{connector}{node.label}"
    if node.comment:
        padding = max(1, COMMENT_COLUMN - len(line))
        line = f"{line}{' ' * padding}# {node.comment}"

    lines = [line]
    if node.children:
        child_prefix = prefix + ("    " if is_last else "│   ")
        for index, child in enumerate(node.children):
            lines.extend(
                render_child(child, prefix=child_prefix, is_last=index == len(node.children) - 1)
            )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
