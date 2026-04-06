from __future__ import annotations

import json
from pathlib import Path

import ispec.assistant.classifier_service as classifier_service
import ispec.assistant.service as service
from ispec.assistant.usage_logging import record_inference_usage_event


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_record_inference_usage_event_writes_jsonl(tmp_path, monkeypatch):
    log_dir = tmp_path / 'logs'
    monkeypatch.setenv('ISPEC_LOG_DIR', str(log_dir))
    monkeypatch.setenv('ISPEC_INFERENCE_USAGE_LOG_ENABLED', '1')

    record_inference_usage_event(
        provider='vllm',
        model='test-model',
        meta={'elapsed_ms': 12, 'usage': {'total_tokens': 5}},
        ok=True,
        observability_context={'surface': 'support_chat', 'session_id': 's1'},
    )

    files = list((log_dir / 'inference-usage').glob('usage-*.jsonl'))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding='utf-8').strip())
    assert payload['surface'] == 'support_chat'
    assert payload['usage']['total_tokens'] == 5
    assert 'content' not in payload


def test_generate_reply_records_usage_event(tmp_path, monkeypatch):
    log_dir = tmp_path / 'logs'
    monkeypatch.setenv('ISPEC_LOG_DIR', str(log_dir))
    monkeypatch.setenv('ISPEC_INFERENCE_USAGE_LOG_ENABLED', '1')
    monkeypatch.setenv('ISPEC_ASSISTANT_PROVIDER', 'vllm')
    monkeypatch.setenv('ISPEC_VLLM_URL', 'http://127.0.0.1:8000')
    monkeypatch.setenv('ISPEC_VLLM_MODEL', 'test-model')

    def fake_post(url, *, json, headers, timeout):
        return _DummyResponse({
            'choices': [{'message': {'role': 'assistant', 'content': 'Hi'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3},
        })

    monkeypatch.setattr(service.requests, 'post', fake_post)

    reply = service.generate_reply(
        message='Hello',
        history=None,
        context=None,
        observability_context={'surface': 'support_chat', 'session_id': 'abc'},
    )
    assert reply.ok is True

    files = list((log_dir / 'inference-usage').glob('usage-*.jsonl'))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding='utf-8').strip())
    assert payload['surface'] == 'support_chat'
    assert payload['session_id'] == 'abc'
    assert payload['provider'] == 'vllm'


def test_generate_classifier_reply_records_usage_event(tmp_path, monkeypatch):
    log_dir = tmp_path / 'logs'
    monkeypatch.setenv('ISPEC_LOG_DIR', str(log_dir))
    monkeypatch.setenv('ISPEC_INFERENCE_USAGE_LOG_ENABLED', '1')
    monkeypatch.setenv('ISPEC_ASSISTANT_CLASSIFIER_PROVIDER', 'vllm')
    monkeypatch.setenv('ISPEC_ASSISTANT_CLASSIFIER_VLLM_URL', 'http://127.0.0.1:8000')
    monkeypatch.setenv('ISPEC_ASSISTANT_CLASSIFIER_VLLM_MODEL', 'tiny-classifier')

    def fake_post(url, *, json, headers, timeout):
        assert json['structured_outputs'] == {'json': {'type': 'object'}}
        assert 'guided_json' not in json
        return _DummyResponse({
            'choices': [{'message': {'role': 'assistant', 'content': '{"kind": "approve"}'}}],
            'usage': {'total_tokens': 4},
        })

    monkeypatch.setattr(classifier_service.requests, 'post', fake_post)

    reply = classifier_service.generate_classifier_reply(
        base_generate_reply_fn=service.generate_reply,
        messages=[{'role': 'user', 'content': 'yes'}],
        vllm_extra_body={'structured_outputs': {'json': {'type': 'object'}}},
        observability_context={'surface': 'turn_decision', 'task': 'support_chat'},
    )
    assert reply.ok is True

    files = list((log_dir / 'inference-usage').glob('usage-*.jsonl'))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding='utf-8').strip())
    assert payload['surface'] == 'turn_decision'
    assert payload['task'] == 'support_chat'
    assert payload['provider'] == 'classifier_vllm'
