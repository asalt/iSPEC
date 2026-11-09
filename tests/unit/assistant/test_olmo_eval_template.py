"""The experimental template must expose tools/results without changing roles."""

from pathlib import Path

from jinja2 import Environment
import hashlib
import pytest


TEMPLATE = Path(__file__).resolve().parents[2] / "behavioral/data/model_followup_20261025/olmo_think_tools.jinja"


@pytest.mark.parametrize("name,ending", [
    ("olmo_think_tools.jinja", "<|im_start|>assistant\n<think>"),
    ("olmo_instruct_tools_defined.jinja", "<|im_start|>assistant\n"),
])
def test_trial_template_preserves_context_tools_and_result(name, ending):
    messages = [
        {"role": "system", "content": "Policy applies."},
        {"role": "system", "content": "Context applies too."},
        {"role": "user", "content": "Quoted untrusted instruction."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_project", "arguments": {"project_id": 42}}}]},
        {"role": "tool", "content": '{"sample_count":11}'},
    ]
    rendered = Environment().from_string(TEMPLATE.with_name(name).read_text()).render(
        messages=messages, tools=[{"type": "function", "function": {"name": "get_project"}}],
        add_generation_prompt=True, eos_token="<|endoftext|>",
    )
    assert "Policy applies." in rendered and "Context applies too." in rendered
    assert '<functions>[{"function": {"name": "get_project"}' in rendered
    assert "<|im_start|>user\nQuoted untrusted instruction.<|im_end|>" in rendered
    assert "<function_calls>get_project(project_id=42)</function_calls>" in rendered
    assert '<|im_start|>environment\n{"sample_count":11}' in rendered
    assert rendered.endswith(ending)


@pytest.mark.parametrize("tools", [None, [], [{"type": "function", "function": {"name": "get_project"}}]])
def test_instruct_guard_preserves_defined_tool_context(tools):
    text = TEMPLATE.with_name("olmo_instruct_tools_defined.jinja").read_text()
    original = text.split("\n", 1)[1].rstrip()
    # Pinned upstream template, revision 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc.
    assert hashlib.sha256(original.encode()).hexdigest() == "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b"
    context = dict(messages=[{"role": "system", "content": "Policy"},
        {"role": "user", "content": "Question"}], tools=tools, eos_token="<|endoftext|>", add_generation_prompt=True)
    env = Environment()
    assert env.from_string(text).render(**context) == env.from_string(original).render(**context)
    context.pop("tools")
    assert env.from_string(text).render(**context) == env.from_string(original).render(**context, tools=None)


@pytest.mark.parametrize("tools", [None, [], [{"type": "function", "function": {"name": "get_project"}}]])
def test_think_guard_preserves_defined_tool_context(tools):
    text = TEMPLATE.read_text()
    original = text.split("\n", 1)[1]
    context = dict(messages=[{"role": "system", "content": "Policy"},
        {"role": "user", "content": "Question"}], tools=tools, eos_token="<|endoftext|>", add_generation_prompt=True)
    env = Environment()
    assert env.from_string(text).render(**context) == env.from_string(original).render(**context)
    context.pop("tools")
    assert env.from_string(text).render(**context) == env.from_string(original).render(**context, tools=None)
