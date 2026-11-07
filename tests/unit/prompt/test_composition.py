import json

import pytest

from ispec.prompt import loader
from ispec.prompt.composition import (
    MAX_COMPONENTS,
    clean_prompt_composition,
    prompt_composition,
)
from ispec.prompt import load_bound_prompt, prompt_binding
from ispec.prompt.sync import sync_prompts


@pytest.fixture
def render(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "resolve_prompt_root", lambda: tmp_path)
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "not-created.db"))

    def render_source(family, body, **kwargs):
        path = tmp_path / (family + ".md")
        path.write_text(body)
        loader._PROMPT_SOURCE_CACHE.pop(str(path.resolve()), None)
        return loader.render_prompt(family, **kwargs)

    return render_source


def test_objects_render_exactly_like_text_and_keep_nested_provenance(render):
    base = render("base", "Hello $name.", values={"name": "private $value"})
    task = render("task", "$base\nDraft the note.", values={"base": base})
    parent = render(
        "parent", "$task\n$json", values={"task": task, "json": {"ready": True}}
    )
    flat = render(
        "parent", "$task\n$json", values={"task": task.text, "json": {"ready": True}}
    )
    assert (
        parent.text
        == flat.text
        == 'Hello private $value.\nDraft the note.\n{"ready": true}'
    )
    graph = parent.observability_fields()["prompt_composition"]
    assert [n["family"] for n in graph["nodes"]] == ["parent", "task", "base"]
    assert [n["parent"] for n in graph["nodes"]] == [None, 0, 1]
    assert graph["nodes"][2]["value_slots"] == ["name"]
    assert "private" not in json.dumps(graph)
    assert "rendered_sha256" not in parent.observability_fields()
    assert clean_prompt_composition(graph) == graph


def test_only_inserted_components_are_recorded_in_template_order(render):
    first = render("first", "First")
    second = render("second", "Second")
    parent = render(
        "parent", "$b $$a ${a} $a", values={"a": first, "b": second, "unused": first}
    )
    assert parent.text == "Second $a First First"
    assert [(p.slot, p.occurrence) for p in parent.components] == [
        ("b", 0),
        ("a", 0),
        ("a", 1),
    ]
    assert [p.prompt.source.family for p in parent.components] == [
        "second",
        "first",
        "first",
    ]


def test_variant_changes_do_not_create_new_template_identity(render):
    one = render(
        "ack", "Optional: $phrase", values={"phrase": "Gotcha."}, variant_id="gotcha"
    )
    two = render(
        "ack",
        "Optional: $phrase",
        values={"phrase": "Understood."},
        variant_id="understood",
    )
    assert one.source.body_sha256 == two.source.body_sha256
    assert one.text != two.text
    assert prompt_composition(one)["nodes"][0]["variant_id"] == "gotcha"
    assert prompt_composition(two)["nodes"][0]["variant_id"] == "understood"


def test_bound_components_keep_independent_indexed_versions(tmp_path, monkeypatch):
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    (prompt_root / "base.md").write_text("Base")
    (prompt_root / "answer.md").write_text("$base\nAnswer")
    monkeypatch.setattr(loader, "resolve_prompt_root", lambda: prompt_root)
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    sync_prompts(prompt_root=prompt_root, source_root=tmp_path / "no-source")
    (prompt_root / "base.md").write_text("Updated base")
    sync_prompts(prompt_root=prompt_root, source_root=tmp_path / "no-source")

    @prompt_binding("base")
    def base():
        return load_bound_prompt(base)

    @prompt_binding("answer")
    def answer():
        return load_bound_prompt(answer, values={"base": base()})

    graph = prompt_composition(answer())
    assert [node["version_num"] for node in graph["nodes"]] == [1, 2]
    assert graph["nodes"][0]["binding"].endswith(".answer")
    assert graph["nodes"][1]["binding"].endswith(".base")
    assert clean_prompt_composition(graph) == graph


def test_manifest_is_bounded_and_marks_missing_tail(render):
    prompt = render("leaf", "Leaf")
    for _ in range(MAX_COMPONENTS + 3):
        prompt = render("parent", "$child", values={"child": prompt})
    graph = prompt_composition(prompt)
    assert len(graph["nodes"]) == MAX_COMPONENTS
    assert graph["truncated"] is True
    assert clean_prompt_composition(graph) == graph


def test_log_boundary_drops_unknown_content_and_rejects_bad_edges(render):
    graph = prompt_composition(render("base", "Private body"))
    graph["messages"] = "private conversation"
    graph["nodes"][0]["text"] = "private prompt"
    cleaned = clean_prompt_composition(graph)
    assert "private" not in json.dumps(cleaned)
    graph["nodes"][0]["parent"] = 0
    assert clean_prompt_composition(graph) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "no"),
        ("variant_id", {}),
        ("version_num", True),
        ("binding", []),
        ("value_slots", [123]),
    ],
)
def test_bad_metadata_fails_closed_at_logging_boundary(render, field, value):
    graph = prompt_composition(render("base", "Text"))
    graph["nodes"][0][field] = value
    assert clean_prompt_composition(graph) is None


def test_boolean_is_not_a_schema_version(render):
    graph = prompt_composition(render("base", "Text"))
    graph["schema_version"] = True
    assert clean_prompt_composition(graph) is None


def test_missing_values_still_raise_and_json_values_remain_supported(render):
    with pytest.raises(KeyError):
        render("parent", "$missing", values={"unused": "value"})
    result = render("parent", "$x $y $z", values={"x": None, "y": 3, "z": ["value"]})
    assert result.text == ' 3 ["value"]'


def test_interactive_stages_keep_one_base_and_identical_text(monkeypatch, tmp_path):
    from ispec.assistant import service

    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    for name in (
        "ISPEC_ASSISTANT_SYSTEM_PROMPT",
        "ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH",
        "ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA",
        "ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    original = loader._render_value
    for build in (
        service._render_system_prompt_answer,
        service._render_system_prompt_review,
        service._render_system_prompt_review_decider,
        lambda: service._render_system_prompt_planner(
            tools_available=True, tool_names={"get_project"}
        ),
    ):
        result = build()
        assert [p.slot for p in result.components] == ["base_prompt"]
        assert result.components[0].prompt.source.family == "assistant.base.system"
        assert result.text.count(result.components[0].prompt.text) == 1
        with monkeypatch.context() as flat:
            flat.setattr(
                loader,
                "_render_value",
                lambda value: original(
                    value.text if isinstance(value, loader.RenderedPrompt) else value
                ),
            )
            assert build().text == result.text
    monkeypatch.setenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA", "Extra context")
    assert service._render_system_prompt_answer().components
    monkeypatch.setenv("ISPEC_ASSISTANT_SYSTEM_PROMPT", "Replacement prompt")
    replaced = service._render_system_prompt_answer()
    assert replaced.text == "Replacement prompt\n\nExtra context"
    assert not replaced.components
