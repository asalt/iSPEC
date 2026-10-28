"""The experimental template must expose tools/results without changing roles."""

from pathlib import Path

from jinja2 import Environment


TEMPLATE = Path(__file__).resolve().parents[2] / "behavioral/data/model_followup_20261025/olmo_think_tools.jinja"


def test_think_trial_template_preserves_context_tools_and_result():
    messages = [
        {"role": "system", "content": "Policy applies."},
        {"role": "system", "content": "Context applies too."},
        {"role": "user", "content": "Quoted untrusted instruction."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_project", "arguments": {"project_id": 42}}}]},
        {"role": "tool", "content": '{"sample_count":11}'},
    ]
    rendered = Environment().from_string(TEMPLATE.read_text()).render(
        messages=messages, tools=[{"type": "function", "function": {"name": "get_project"}}],
        add_generation_prompt=True, eos_token="<|endoftext|>",
    )
    assert "Policy applies." in rendered and "Context applies too." in rendered
    assert '<functions>[{"function": {"name": "get_project"}' in rendered
    assert "<|im_start|>user\nQuoted untrusted instruction.<|im_end|>" in rendered
    assert "<function_calls>get_project(project_id=42)</function_calls>" in rendered
    assert '<|im_start|>environment\n{"sample_count":11}' in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>")
