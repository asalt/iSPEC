from __future__ import annotations

from textwrap import dedent

from ispec.prompt.bindings import binding_meta_for_callable, discover_prompt_bindings_ast, prompt_binding, prompt_family_for


@prompt_binding("assistant.binding.example")
def _bound_example() -> str:
    return "unused"


def test_prompt_binding_attaches_runtime_metadata():
    assert prompt_family_for(_bound_example) == "assistant.binding.example"

    meta = binding_meta_for_callable(_bound_example)
    assert meta.family == "assistant.binding.example"
    assert meta.binding_kind == "wrapper"
    assert meta.qualname.endswith("_bound_example")
    assert meta.binding_ref.endswith(":_bound_example")


def test_discover_prompt_bindings_ast_finds_explicit_decorators(tmp_path):
    source_root = tmp_path / "src"
    pkg_dir = source_root / "pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "module.py").write_text(
        dedent(
            """
            from ispec.prompt import prompt_binding

            @prompt_binding("assistant.alpha")
            def alpha():
                return "alpha"

            @prompt_binding("assistant.beta", kind="helper")
            def beta():
                return "beta"

            family = "assistant.dynamic"

            @prompt_binding(family)
            def dynamic():
                return "dynamic"
            """
        ),
        encoding="utf-8",
    )

    bindings = discover_prompt_bindings_ast(source_root=source_root)
    pairs = sorted((item.family, item.module, item.qualname, item.binding_kind) for item in bindings)

    assert pairs == [
        ("assistant.alpha", "pkg.module", "alpha", "wrapper"),
        ("assistant.beta", "pkg.module", "beta", "helper"),
    ]
