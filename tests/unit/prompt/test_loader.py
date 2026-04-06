from __future__ import annotations

from ispec.prompt import load_bound_prompt, prompt_binding, prompt_observability_context
from ispec.prompt.loader import _PROMPT_SOURCE_CACHE
from ispec.prompt.sync import sync_prompts
import ispec.prompt.loader as prompt_loader


@prompt_binding("assistant.loader.example")
def _loader_example_prompt() -> str:
    return "unused"


def test_load_bound_prompt_enriches_version_and_observability_fields(tmp_path, monkeypatch):
    prompt_root = tmp_path / "prompts"
    source_root = tmp_path / "src"
    prompt_root.mkdir()
    source_root.mkdir()
    (prompt_root / "assistant.loader.example.md").write_text(
        "+++\n"
        'title = "Loader Example"\n'
        'notes = "For runtime lookup tests."\n'
        "+++\n"
        "Hello $name\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "prompts.db"
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(db_path))
    sync_prompts(prompt_root=prompt_root, source_root=source_root)

    _PROMPT_SOURCE_CACHE.clear()
    monkeypatch.setattr(prompt_loader, "resolve_prompt_root", lambda: prompt_root)

    prompt = load_bound_prompt(_loader_example_prompt, values={"name": "Alex"})

    assert prompt.text == "Hello Alex\n"
    assert prompt.source.family == "assistant.loader.example"
    assert prompt.version.version_num == 1
    context = prompt_observability_context(prompt, extra={"surface": "support_chat"})
    assert context["surface"] == "support_chat"
    assert context["prompt_family"] == "assistant.loader.example"
    assert context["prompt_version_num"] == 1
    assert context["prompt_binding"].endswith(":_loader_example_prompt")
