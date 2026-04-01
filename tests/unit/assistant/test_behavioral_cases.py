from __future__ import annotations

import json
import runpy
from pathlib import Path

from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from tests.behavioral.local_cases import assert_behavioral_case_expectations, load_local_behavioral_case


SCRIPT_PATH = Path(__file__).resolve().parents[3] / 'scripts' / 'extract_behavioral_scenarios.py'
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT_PATH))
extract_behavioral_cases = SCRIPT_GLOBALS['extract_behavioral_cases']
write_behavioral_cases = SCRIPT_GLOBALS['write_behavioral_cases']


def test_load_local_behavioral_case_and_expectations(tmp_path):
    path = tmp_path / 'case.json'
    path.write_text(
        json.dumps(
            {
                'label': 'session_244',
                'tags': ['project-note'],
                'messages': [
                    {'id': 1, 'role': 'user', 'content': 'Please save it.'},
                    {
                        'id': 2,
                        'role': 'assistant',
                        'content': 'Saved the note. Comment ID is 7.',
                        'assistant_meta': {
                            'response_contract': {'configured_mode': 'shadow', 'selected_contract': 'direct'},
                            'reply_interpretation': {'runtime_kind': 'approve', 'runtime_action': 'approve_save'},
                            'tool_calls': [{'name': 'create_project_comment', 'ok': True, 'protocol': 'openai'}],
                        },
                    },
                ],
                'expect': {
                    'assistant_message_count': 1,
                    'final_assistant_contains': 'Saved the note.',
                    'response_contract_mode': 'shadow',
                    'reply_interpretation_runtime_kind': 'approve',
                    'tool_call_names_include': 'create_project_comment',
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    case = load_local_behavioral_case(path)
    assert case['label'] == 'session_244'
    assert_behavioral_case_expectations(case)


def test_extract_behavioral_cases_exports_assistant_meta_subset(tmp_path):
    assistant_db_path = tmp_path / 'assistant.db'
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id='session-export-1', user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add(SupportMessage(session_pk=session.id, role='user', content='thanks'))
        assistant_db.add(
            SupportMessage(
                session_pk=session.id,
                role='assistant',
                content='You are welcome.',
                meta_json=json.dumps(
                    {
                        'provider': 'vllm',
                        'model': 'local-model',
                        'response_contract': {
                            'configured_mode': 'shadow',
                            'selected_contract': 'direct',
                            'shadow_candidate': 'Thanks for the note.',
                        },
                        'reply_interpretation': {
                            'runtime_kind': 'approve',
                            'runtime_action': 'approve_save',
                            'applied': True,
                        },
                        'tool_calls': [
                            {'name': 'create_project_comment', 'ok': True, 'protocol': 'openai'},
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
        )
        assistant_db.commit()
        session_pk = int(session.id)

    cases = extract_behavioral_cases(
        assistant_db_path=assistant_db_path,
        session_pks=[session_pk],
        tags=['social-close'],
        notes='exported from test',
    )
    assert len(cases) == 1
    case = cases[0]
    assert case['label'] == f'session_{session_pk}'
    assert case['tags'] == ['social-close']
    assert case['notes'] == 'exported from test'
    assert len(case['messages']) == 2
    assistant_message = case['messages'][1]
    assert assistant_message['assistant_meta']['provider'] == 'vllm'
    assert assistant_message['assistant_meta']['response_contract']['configured_mode'] == 'shadow'
    assert assistant_message['assistant_meta']['reply_interpretation']['runtime_kind'] == 'approve'
    assert assistant_message['assistant_meta']['tool_calls'][0]['name'] == 'create_project_comment'

    written = write_behavioral_cases(cases, tmp_path / 'local')
    assert len(written) == 1
    loaded = load_local_behavioral_case(written[0])
    assert loaded['label'] == case['label']
