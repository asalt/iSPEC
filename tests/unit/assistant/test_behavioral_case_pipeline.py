import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[3] / 'scripts/prepare_behavioral_case.py'
spec = importlib.util.spec_from_file_location('prepare_behavioral_case', PATH)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


@pytest.fixture
def staged(tmp_path):
    source = tmp_path / 'snapshot.json'
    source.write_text(json.dumps({'source': {'session_id': 'private-identifier'}, 'messages': [
        {'id': 899, 'role': 'user', 'content': 'Dr Example: project 1499 received 12 IP samples. Do not save yet.'},
        {'id': 900, 'role': 'assistant', 'content': 'Saved it.', 'assistant_meta': {
            'tool_calls': [{'name': 'create_project_comment', 'ok': False, 'error': '/home/private/failure'}]}},
    ]}))
    directory = pipeline.stage(source, directory=tmp_path / 'stage', label='pending-receipt',
                               replacements={'Dr Example': 'Robin', '1499': '42'})
    candidate = directory / 'candidate.json'
    candidate.write_bytes((directory / 'packet.json').read_bytes())
    expect = tmp_path / 'expect.json'
    expect.write_text(json.dumps({'assistant_message_count': 1, 'tool_call_names_include': ['create_project_comment']}))
    return directory, candidate, expect


def test_stage_keeps_provenance_private_and_preserves_failure_evidence(staged):
    directory, candidate, _ = staged
    case, flags = pipeline.candidate_checks(directory, candidate)
    assert flags == []
    assert '12 IP' in case['messages'][0]['content']
    assert 'Do not save yet' in case['messages'][0]['content']
    assert case['messages'][1]['content'] == 'Saved it.'
    assert 'private-identifier' not in candidate.read_text()
    assert json.loads((directory / 'evidence.json').read_text())['2']['tool_calls'][0]['ok'] is False
    assert directory.stat().st_mode & 0o077 == 0
    for name in ('packet.json', 'source.json', 'private-provenance.json', 'evidence.json'):
        assert (directory / name).stat().st_mode & 0o077 == 0


def test_promotion_is_separate_from_review_and_has_no_raw_provenance(staged, tmp_path):
    directory, candidate, expect = staged
    output = tmp_path / 'fixture.json'
    with pytest.raises(FileNotFoundError):
        pipeline.promote(directory, candidate, output)
    with pytest.raises(ValueError, match='Both'):
        pipeline.review(directory, candidate, expect, privacy_reviewed=True, semantics_reviewed=False)
    pipeline.review(directory, candidate, expect, privacy_reviewed=True, semantics_reviewed=True)
    assert not output.exists()
    result = pipeline.promote(directory, candidate, output)
    assert result['git_operations'] == 0
    exported = json.loads(output.read_text())
    assert set(exported) == {'label', 'tags', 'messages', 'expect'}
    assert exported['messages'][1]['assistant_meta']['tool_calls'][0]['ok'] is False
    assert 'private' not in output.read_text()
    with pytest.raises(FileExistsError):
        pipeline.promote(directory, candidate, output)


@pytest.mark.parametrize('target', ['candidate.json', 'packet.json', 'evidence.json'])
def test_post_review_edits_require_new_review(staged, tmp_path, target):
    directory, candidate, expect = staged
    pipeline.review(directory, candidate, expect, privacy_reviewed=True, semantics_reviewed=True)
    path = directory / target
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        pipeline.promote(directory, candidate, tmp_path / 'fixture.json')


@pytest.mark.parametrize('text', ['user@example.org', '10.16.1.51', 'https://example.org/private',
                                  'U0123456789', '/home/private/research', 'Dr Example', 'xoxb-synthetic-token'])
def test_leak_checks_block_review_without_echoing_private_text(staged, text):
    directory, candidate, expect = staged
    data = json.loads(candidate.read_text())
    data['messages'][0]['content'] += ' ' + text
    candidate.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='leak checks') as error:
        pipeline.review(directory, candidate, expect, privacy_reviewed=True, semantics_reviewed=True)
    assert text not in str(error.value)


def test_model_cannot_relabel_or_drop_messages(staged):
    directory, candidate, _ = staged
    data = json.loads(candidate.read_text())
    data['messages'].pop()
    candidate.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='order, count or roles'):
        pipeline.candidate_checks(directory, candidate)


def test_codex_is_opt_in_bounded_and_cannot_promote(staged, monkeypatch):
    directory, candidate, _ = staged
    candidate.unlink()
    monkeypatch.setenv('SLACK_BOT_TOKEN', 'synthetic-never-pass')
    with pytest.raises(ValueError, match='allow-model-send'):
        pipeline.rewrite(directory, model='selected-model')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[command.index('--sandbox') + 1] == 'read-only'
        assert '--ignore-user-config' in command and '--ephemeral' in command
        assert 'features.shell_tool=false' in command
        assert 'SLACK_BOT_TOKEN' not in kwargs['env']
        assert 'private-identifier' not in kwargs['input']
        return SimpleNamespace(returncode=0, stdout=(directory / 'packet.json').read_text())
    result = pipeline.rewrite(directory, model='selected-model', allow_model_send=True, run=run)
    assert len(calls) == 1 and result['review_required']
    assert not (directory / 'review.json').exists()
    assert not (directory / 'promotion.json').exists()


def test_slack_export_handles_participants_consistently(tmp_path):
    source = tmp_path / 'slack.json'
    source.write_text(json.dumps({'messages': [{'user': 'U0123456789', 'text': 'Received 12 samples.'},
                                               {'user': 'U0123456789', 'text': 'Please keep this as a draft.'}]}))
    directory = pipeline.stage(source, directory=tmp_path / 'stage', label='slack-receipt')
    packet = json.loads((directory / 'packet.json').read_text())
    assert all(m['content'].startswith('Participant 1:') for m in packet['messages'])
    assert 'U0123456789' not in json.dumps(packet)


def test_cli_review_and_export_use_existing_fixture_loader(staged, tmp_path):
    directory, candidate, expect = staged
    output = tmp_path / 'export.json'
    for arguments in [
        ['review', str(directory), '--expectations', str(expect), '--privacy-reviewed', '--semantics-reviewed'],
        ['promote', str(directory), '--output', str(output)],
    ]:
        result = subprocess.run([sys.executable, str(PATH), *arguments], cwd=tmp_path,
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert 'Dr Example' not in result.stdout
    from tests.behavioral.local_cases import load_local_behavioral_case
    assert load_local_behavioral_case(output)['messages'][1]['assistant_meta']['tool_calls'][0]['ok'] is False


def test_credentials_block_the_model_call(staged):
    directory, candidate, _ = staged
    candidate.unlink()
    packet = json.loads((directory / 'packet.json').read_text())
    packet['messages'][0]['content'] += ' xoxb-synthetic-private-token'
    (directory / 'packet.json').write_text(json.dumps(packet))
    with pytest.raises(ValueError, match='credentials'):
        pipeline.rewrite(directory, model='selected-model', allow_model_send=True,
                         run=lambda *a, **k: pytest.fail('Credentials must not reach a provider'))
