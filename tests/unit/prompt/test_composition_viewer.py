import importlib.util
import json
from pathlib import Path

import pytest

from ispec.prompt.composition import prompt_composition
from ispec.prompt.models import RenderedPrompt
from ispec.prompt.parser import parse_prompt_file


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "composition_viewer", ROOT / "scripts/view_prompt_composition.py"
)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def test_export_keeps_only_metadata_and_never_overwrites(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("Private prompt")
    graph = prompt_composition(
        RenderedPrompt(text="Private rendered body", source=parse_prompt_file(source))
    )
    log = tmp_path / "usage.jsonl"
    row = {
        "session_id": "</script><img src=x>",
        "prompt_composition": graph,
        "elapsed_ms": float("nan"),
        "messages": "Private message",
        "response": "Private reply",
    }
    log.write_text("invalid\n" + json.dumps(row) + "\n")
    output = tmp_path / "viewer.html"
    result = viewer.export_view(log, output)
    assert result["calls"] == 1 and result["skipped"] == 1
    html = output.read_text()
    assert "Private" not in html
    assert "<img src=x>" not in html
    assert "\\u003c/script>" in html
    assert "/* COMPOSITION_DATA */ {}" not in html
    assert "NaN" not in html
    with pytest.raises(FileExistsError):
        viewer.export_view(log, output)


def test_latest_call_limit_and_legacy_rows_are_explicit(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("Body")
    graph = prompt_composition(
        RenderedPrompt(text="Body", source=parse_prompt_file(source))
    )
    log = tmp_path / "usage.jsonl"
    log.write_text(
        "{}\n"
        + "\n".join(
            json.dumps({"message_id": n, "prompt_composition": graph}) for n in range(5)
        )
    )
    calls, skipped = viewer.read_calls(log, limit=2)
    assert [call["message_id"] for call in calls] == [3, 4]
    assert skipped == 1
    with pytest.raises(ValueError):
        viewer.read_calls(log, limit=501)
