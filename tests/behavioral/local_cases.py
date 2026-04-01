from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOCAL_CASE_DIR = Path(__file__).resolve().parent / "local"
_SUPPORTED_EXPECT_KEYS = {
    "assistant_message_count",
    "final_assistant_contains",
    "final_assistant_not_contains",
    "response_contract_mode",
    "response_contract_selected_contract",
    "response_contract_shadow_candidate_contains",
    "reply_interpretation_runtime_kind",
    "reply_interpretation_runtime_action",
    "tool_call_names_include",
}


def local_behavioral_case_dir(root: str | Path | None = None) -> Path:
    if root is None:
        return LOCAL_CASE_DIR
    return Path(root).expanduser().resolve()


def discover_local_behavioral_case_paths(root: str | Path | None = None) -> list[Path]:
    case_dir = local_behavioral_case_dir(root)
    if not case_dir.exists():
        return []
    return sorted(path for path in case_dir.glob('*.json') if path.is_file())


def _as_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    raise ValueError('expected a string or list of strings')


def _validate_messages(messages: Any, *, path: Path) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f'{path}: messages must be a non-empty list')
    normalized: list[dict[str, Any]] = []
    last_id: int | None = None
    for index, item in enumerate(messages, start=1):
        if not isinstance(item, dict):
            raise ValueError(f'{path}: message #{index} must be an object')
        role = str(item.get('role') or '').strip()
        content = item.get('content')
        msg_id = item.get('id')
        if role not in {'user', 'assistant', 'system'}:
            raise ValueError(f'{path}: message #{index} has invalid role {role!r}')
        if not isinstance(content, str):
            raise ValueError(f'{path}: message #{index} content must be a string')
        if not isinstance(msg_id, int) or msg_id <= 0:
            raise ValueError(f'{path}: message #{index} id must be a positive integer')
        if last_id is not None and msg_id < last_id:
            raise ValueError(f'{path}: message ids must be non-decreasing')
        last_id = msg_id
        normalized_item = {
            'id': msg_id,
            'role': role,
            'content': content,
        }
        created_at = item.get('created_at')
        if created_at is not None and not isinstance(created_at, str):
            raise ValueError(f'{path}: message #{index} created_at must be a string when present')
        if isinstance(created_at, str):
            normalized_item['created_at'] = created_at
        assistant_meta = item.get('assistant_meta')
        if assistant_meta is not None:
            if role != 'assistant' or not isinstance(assistant_meta, dict):
                raise ValueError(f'{path}: message #{index} assistant_meta must be an object on assistant messages only')
            normalized_item['assistant_meta'] = assistant_meta
        normalized.append(normalized_item)
    return normalized


def load_local_behavioral_case(path: str | Path) -> dict[str, Any]:
    case_path = Path(path).expanduser().resolve()
    data = json.loads(case_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'{case_path}: case must be a JSON object')
    label = str(data.get('label') or '').strip()
    if not label:
        raise ValueError(f'{case_path}: label is required')
    tags = _as_string_list(data.get('tags'))
    messages = _validate_messages(data.get('messages'), path=case_path)
    case: dict[str, Any] = {
        'label': label,
        'tags': tags,
        'messages': messages,
    }
    source = data.get('source')
    if source is not None:
        if not isinstance(source, dict):
            raise ValueError(f'{case_path}: source must be an object when present')
        case['source'] = source
    notes = data.get('notes')
    if notes is not None:
        if not isinstance(notes, str):
            raise ValueError(f'{case_path}: notes must be a string when present')
        case['notes'] = notes
    expect = data.get('expect')
    if expect is not None:
        if not isinstance(expect, dict):
            raise ValueError(f'{case_path}: expect must be an object when present')
        unknown = sorted(set(expect) - _SUPPORTED_EXPECT_KEYS)
        if unknown:
            raise ValueError(f'{case_path}: unsupported expect keys: {", ".join(unknown)}')
        case['expect'] = expect
    return case


def load_local_behavioral_cases(root: str | Path | None = None) -> list[dict[str, Any]]:
    return [load_local_behavioral_case(path) for path in discover_local_behavioral_case_paths(root)]


def assistant_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in case.get('messages', []) if isinstance(item, dict) and item.get('role') == 'assistant']


def final_assistant_message(case: dict[str, Any]) -> dict[str, Any] | None:
    messages = assistant_messages(case)
    return messages[-1] if messages else None


def assert_behavioral_case_expectations(case: dict[str, Any]) -> None:
    expect = case.get('expect')
    if not isinstance(expect, dict) or not expect:
        return

    assistant_msgs = assistant_messages(case)
    final_assistant = final_assistant_message(case)
    final_text = str((final_assistant or {}).get('content') or '')
    final_meta = (final_assistant or {}).get('assistant_meta')
    final_meta = final_meta if isinstance(final_meta, dict) else {}
    response_contract = final_meta.get('response_contract')
    response_contract = response_contract if isinstance(response_contract, dict) else {}
    reply_interpretation = final_meta.get('reply_interpretation')
    reply_interpretation = reply_interpretation if isinstance(reply_interpretation, dict) else {}
    tool_calls = final_meta.get('tool_calls')
    tool_calls = tool_calls if isinstance(tool_calls, list) else []

    if 'assistant_message_count' in expect:
        assert len(assistant_msgs) == int(expect['assistant_message_count'])

    for needle in _as_string_list(expect.get('final_assistant_contains')):
        assert needle in final_text
    for needle in _as_string_list(expect.get('final_assistant_not_contains')):
        assert needle not in final_text

    if 'response_contract_mode' in expect:
        assert response_contract.get('configured_mode') == expect['response_contract_mode']
    if 'response_contract_selected_contract' in expect:
        assert response_contract.get('selected_contract') == expect['response_contract_selected_contract']
    for needle in _as_string_list(expect.get('response_contract_shadow_candidate_contains')):
        shadow_text = str(response_contract.get('shadow_candidate') or '')
        assert needle in shadow_text

    if 'reply_interpretation_runtime_kind' in expect:
        assert reply_interpretation.get('runtime_kind') == expect['reply_interpretation_runtime_kind']
    if 'reply_interpretation_runtime_action' in expect:
        assert reply_interpretation.get('runtime_action') == expect['reply_interpretation_runtime_action']

    if 'tool_call_names_include' in expect:
        names = {
            str(item.get('name') or '').strip()
            for item in tool_calls
            if isinstance(item, dict) and str(item.get('name') or '').strip()
        }
        for name in _as_string_list(expect.get('tool_call_names_include')):
            assert name in names
