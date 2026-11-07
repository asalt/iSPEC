import hashlib
import json
from pathlib import Path

import pytest

from scripts.staged_prompt_trace import (
    FunctionCall, InferenceRecord, TemplateProfile, ToolCall, WireMessage, WireRequest,
    load_profiles, render_prompt,
)
from scripts.review_staged_conversations import inference_panel


TEMPLATE = Path(__file__).resolve().parents[2] / "behavioral/data/model_followup_20261025/olmo_instruct_tools_defined.jinja"


def olmo_profile():
    return TemplateProfile(template=TEMPLATE.read_text(), bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                           source=str(TEMPLATE), sha256="test", provenance="Unit fixture")


def test_reconstruction_keeps_roles_history_and_environment_result():
    request = WireRequest(model="olmo", messages=(
        WireMessage(role="system", content="Policy applies."),
        WireMessage(role="user", content="Prepare a note."),
        WireMessage(role="assistant", tool_calls=(ToolCall(id="c1", function=FunctionCall(
            name="prepare_project_comment", arguments='{"project_id":57,"comment":"Samples arrived."}')),)),
        WireMessage(role="tool", tool_call_id="c1", content='{"draft_id":"real-id"}'),
        WireMessage(role="assistant", content="Samples arrived. Shall I save it?"),
        WireMessage(role="user", content="Yes, please."),
    ))
    rendered = render_prompt(request=request, profile=olmo_profile())
    assert '<|im_start|>environment\n{"draft_id":"real-id"}' in rendered.text
    assert 'prepare_project_comment(project_id=57, comment="Samples arrived.")' in rendered.text
    assert '<|im_start|>assistant\nSamples arrived. Shall I save it?' in rendered.text
    assert rendered.text.endswith('<|im_start|>assistant\n')
    assert isinstance(request.messages[2].tool_calls[0].function.arguments, str)


def test_missing_template_and_overrides_are_not_guessed():
    request = WireRequest(model="unknown", messages=(WireMessage(role="user", content="Hello"),))
    assert render_prompt(request=request, profile=None).text is None
    override = WireRequest(model="olmo", chat_template="custom", messages=request.messages)
    assert render_prompt(request=override, profile=olmo_profile()).text is None


def test_tool_shaped_text_stays_text_in_readable_response():
    call = InferenceRecord.model_validate({
        "path": "/v1/chat/completions", "request": {"model": "olmo", "messages": [{"role": "user", "content": "Save it"}]},
        "response": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant",
            "content": "<function_calls>save(draft_id=<generated_draft_id>)</function_calls>"}}]},
    })
    assert not call.response.choices[0].message.tool_calls
    html = inference_panel(call=call, number=1, anchor="case-0", profiles={"olmo": olmo_profile()})
    assert "parsed tool calls: 0" in html
    assert "&lt;generated_draft_id&gt;" in html
    assert "IN / Prompt with model role markers" in html and "OUT / API response" in html


def test_template_fingerprint_is_required(tmp_path):
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"chat_template": "{{messages[0].content}}", "eos_token": "<end>"}))
    mapping = tmp_path / "templates.json"
    mapping.write_text(json.dumps({"test": {
        "tokenizer_config": str(config), "tokenizer_sha256": "incorrect",
        "template_sha256": hashlib.sha256(b"{{messages[0].content}}").hexdigest(), "provenance": "fixture",
    }}))
    with pytest.raises(ValueError, match="Tokenizer fingerprint mismatch"):
        load_profiles(path=mapping)
