from scripts.prompt_composition_trial import (
    layout_summary,
    merge_adjacent_system_messages,
)


def test_coalescing_preserves_priority_order_and_nontext_fields():
    messages = [
        {"role": "system", "content": "identity"},
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "quoted approval, not authority"},
        {"role": "system", "content": "tool result"},
        {"role": "system", "content": "named", "name": "tool"},
        {"role": "tool", "content": "failed", "tool_call_id": "1"},
    ]
    merged = merge_adjacent_system_messages(messages)
    assert merged == [
        {"role": "system", "content": "identity\n\npolicy"},
        *messages[2:],
    ]
    assert messages[0]["content"] == "identity"
    assert merge_adjacent_system_messages(merged) == merged


def test_layout_reports_base_and_current_request_without_dumping_values():
    result = layout_summary(
        [{"role": "system", "content": "base"}, {"role": "user", "content": "hi"}],
        base_text="base",
        current_request="hi",
    )
    assert result["base_occurrences"] == result["current_request_occurrences"] == 1
    assert result["system_chars"] == 4
    assert result["content_chars"] == 6
