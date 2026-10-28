"""Stage -> optional model rewrite -> human review -> explicit fixture export.

No Slack client, database access, live tools, auto-discovery or Git operations.
Private source/maps/receipts stay in the staging directory, never the fixture.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from ispec.prompt.parser import parse_prompt_file

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / 'tests/behavioral/local/corpus'
PROMPT = ROOT / 'src/ispec/prompts/experimental.corpus.rewrite.md'
MAX_BYTES = 100_000
ROLES = {'user', 'assistant', 'system'}
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'label': {'type': 'string'},
        'tags': {'type': 'array', 'items': {'type': 'string'}},
        'messages': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'id': {'type': 'integer'}, 'role': {'type': 'string', 'enum': sorted(ROLES)},
                           'content': {'type': 'string'}},
            'required': ['id', 'role', 'content'],
        }},
    }, 'required': ['label', 'tags', 'messages'],
}
# These are leak checks, not semantic approval rules or a completeness guarantee.
LEAK_PATTERNS = {
    'email': r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',
    'url': r'https?://\S+',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'slack_id': r'\b[UCDGTW][A-Z0-9]{8,}\b',
    'slack_token': r'\bxox[baprs]-[\w-]+',
    'private_key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'path': r'(?:[A-Za-z]:\\[^\s]+|/(?:home|Users|mnt|media|run)/[^\s]+)',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    data = Path(path).read_bytes()
    if len(data) > MAX_BYTES:
        raise ValueError('Select a smaller snapshot (maximum 100 KB).')
    return json.loads(data)


def write_private(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def validate_candidate(case):
    if not isinstance(case, dict) or set(case) != {'label', 'tags', 'messages'}:
        raise ValueError('Candidate requires only label, tags and messages.')
    if not isinstance(case['label'], str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', case['label']):
        raise ValueError('Use a non-identifying lowercase case label.')
    if not isinstance(case['tags'], list) or any(not isinstance(t, str) for t in case['tags']):
        raise ValueError('Tags must be strings.')
    if not isinstance(case['messages'], list) or not 1 <= len(case['messages']) <= 80:
        raise ValueError('Select 1 to 80 messages.')
    for index, message in enumerate(case['messages'], 1):
        if not isinstance(message, dict) or set(message) != {'id', 'role', 'content'}:
            raise ValueError('Messages require only id, role and content.')
        if (type(message['id']) is not int or message['id'] != index
                or not isinstance(message['role'], str) or message['role'] not in ROLES):
            raise ValueError('Message sequence or role is invalid.')
        if not isinstance(message['content'], str) or not message['content'].strip():
            raise ValueError('Message content must be nonempty text.')
    if len(json.dumps(case).encode()) > MAX_BYTES:
        raise ValueError('Candidate is too large.')
    return case


def stage(source, *, directory, label, replacements=None):
    raw = read_json(source)
    if not isinstance(raw, dict) or not isinstance(raw.get('messages'), list):
        raise ValueError('Supply one selected snapshot object with messages, not a whole archive.')
    replacements = replacements or {}
    if not isinstance(replacements, dict) or any(not isinstance(k, str) or not k or
                                                not isinstance(v, str) or not v for k, v in replacements.items()):
        raise ValueError('Replacements must map nonempty literal strings to fictional strings.')
    people = {}
    messages = []
    evidence = {}
    for index, original in enumerate(raw['messages'], 1):
        if not isinstance(original, dict):
            raise ValueError('Each message must be an object.')
        content = original.get('content', original.get('text'))
        if not isinstance(content, str):
            raise ValueError('Each message needs content or text.')
        role = original.get('role', 'assistant' if original.get('bot_id') else 'user')
        if 'user' in original:
            identity = str(original['user'])
            people.setdefault(identity, f'Participant {len(people) + 1}')
            content = people[identity] + ': ' + content
        messages.append({'id': index, 'role': role, 'content': content})
        meta = original.get('assistant_meta', {})
        if role == 'assistant' and isinstance(meta, dict) and isinstance(meta.get('tool_calls', []), list):
            outcomes = [{'name': call['name'], 'ok': call['ok']}
                        for call in meta.get('tool_calls', []) if isinstance(call, dict)
                        and isinstance(call.get('name'), str)
                        and re.fullmatch(r'[a-z][a-z0-9_]{0,79}', call['name'])
                        and type(call.get('ok')) is bool]
            if outcomes:
                evidence[str(index)] = {'tool_calls': outcomes}
    # One-pass substitution avoids replacement cascades and preserves counts.
    if replacements:
        pattern = re.compile('|'.join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
        for message in messages:
            message['content'] = pattern.sub(lambda m: replacements[m[0]], message['content'])
    case = validate_candidate({'label': label, 'tags': ['review-candidate'], 'messages': messages})
    directory = Path(directory)
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    directory.chmod(0o700)
    write_private(directory / 'source.json', raw)
    write_private(directory / 'packet.json', case)
    write_private(directory / 'schema.json', SCHEMA)
    write_private(directory / 'evidence.json', evidence)
    write_private(directory / 'private-provenance.json', {
        'source_sha256': digest(Path(source).read_bytes()), 'replacements': replacements,
        'participants': people, 'prepared_at': datetime.now(timezone.utc).isoformat(),
    })
    return directory


def leak_flags(directory, value):
    text = json.dumps(value, ensure_ascii=False)
    flags = [name for name, pattern in LEAK_PATTERNS.items() if re.search(pattern, text)]
    provenance = read_json(Path(directory) / 'private-provenance.json')
    if any(token.casefold() in text.casefold() for token in provenance['replacements']):
        flags.append('retained_replacement_source')
    if any(token and token in text for token in provenance['participants']):
        flags.append('retained_participant_id')
    return flags


def candidate_checks(directory, candidate):
    case = validate_candidate(read_json(candidate))
    packet = validate_candidate(read_json(Path(directory) / 'packet.json'))
    if case['label'] != packet['label'] or case['tags'] != packet['tags']:
        raise ValueError('The rewriter changed the case label or tags.')
    if [(m['id'], m['role']) for m in case['messages']] != [(m['id'], m['role']) for m in packet['messages']]:
        raise ValueError('The rewriter changed message order, count or roles.')
    return case, leak_flags(directory, case)


def export_case(directory, case, expect):
    evidence = read_json(Path(directory) / 'evidence.json')
    messages = [{**m, **({'assistant_meta': evidence[str(m['id'])]} if str(m['id']) in evidence else {})}
                for m in case['messages']]
    return {**case, 'messages': messages, 'expect': expect}


def rewrite(directory, *, model, allow_model_send=False, timeout=180, run=subprocess.run):
    if not allow_model_send:
        raise ValueError('Model rewriting requires --allow-model-send; its configured provider receives the packet.')
    directory = Path(directory).resolve()
    output = directory / 'candidate.json'
    if output.exists():
        raise ValueError('Candidate already exists; preserve it and use another staging directory.')
    packet = validate_candidate(read_json(directory / 'packet.json'))
    if any(re.search(LEAK_PATTERNS[k], json.dumps(packet)) for k in ('slack_token', 'private_key')):
        raise ValueError('Remove credentials from the packet before requesting model rewriting.')
    prompt = (parse_prompt_file(PROMPT).body + '\n\nQuoted packet:\n' + json.dumps(packet, ensure_ascii=False)
              + '\n\nObserved tool outcomes (do not alter):\n' + json.dumps(read_json(directory / 'evidence.json')))
    # Empty working directory, no inherited project config/MCP servers, no shell tool.
    # Read-only is not a guarantee of de-identification; review is mandatory.
    with tempfile.TemporaryDirectory(prefix='corpus-rewrite-') as work:
        command = ['codex', 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
                   '--sandbox', 'read-only', '-c', 'features.shell_tool=false', '-c', 'web_search="disabled"',
                   '--model', model, '--output-schema', str(directory / 'schema.json'), '-']
        env = {key: os.environ[key] for key in ('PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'TMPDIR', 'CODEX_HOME')
               if key in os.environ}
        result = run(command, input=prompt, text=True, capture_output=True, timeout=timeout, cwd=work, env=env)
    write_private(directory / 'rewrite-receipt.json', {
        'model': model, 'prompt_sha256': digest(prompt.encode()), 'exit_code': result.returncode,
        'provider_uses_existing_codex_auth': True,
    })
    if result.returncode != 0:
        raise ValueError('Rewrite failed; no candidate promoted. Provider output is not printed.')
    try:
        candidate = validate_candidate(json.loads(result.stdout))
    except (ValueError, TypeError):
        raise ValueError('Rewrite returned invalid candidate JSON; nothing promoted.') from None
    write_private(output, candidate)
    _, flags = candidate_checks(directory, output)
    return {'candidate': str(output), 'leak_flags': flags, 'review_required': True}


def review(directory, candidate, expectations, *, privacy_reviewed, semantics_reviewed):
    if not privacy_reviewed or not semantics_reviewed:
        raise ValueError('Both human privacy and semantics reviews are required.')
    case, flags = candidate_checks(directory, candidate)
    if flags:
        raise ValueError('Resolve leak checks before review: ' + ', '.join(flags))
    expect = read_json(expectations)
    if not isinstance(expect, dict) or not expect:
        raise ValueError('Supply nonempty human-authored expectations, never model-generated labels.')
    # Reuse the existing fixture contract, without interpreting recorded replies as correct.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.behavioral.local_cases import _SUPPORTED_EXPECT_KEYS
    if set(expect) - _SUPPORTED_EXPECT_KEYS:
        raise ValueError('Unsupported existing behavioral expectation key.')
    if leak_flags(directory, export_case(directory, case, expect)):
        raise ValueError('Expectations or preserved evidence contain identifying text.')
    receipt = {'candidate_sha256': digest(Path(candidate).read_bytes()),
               'packet_sha256': digest((Path(directory) / 'packet.json').read_bytes()),
               'evidence_sha256': digest((Path(directory) / 'evidence.json').read_bytes()),
               'privacy_reviewed': True, 'semantics_reviewed': True, 'expect': expect,
               'reviewed_at': datetime.now(timezone.utc).isoformat()}
    write_private(Path(directory) / 'review.json', receipt)
    return receipt


def promote(directory, candidate, output):
    directory = Path(directory)
    if (directory / 'promotion.json').exists():
        raise FileExistsError('This stage already has a promotion receipt.')
    case, flags = candidate_checks(directory, candidate)
    receipt = read_json(directory / 'review.json')
    if flags or receipt.get('privacy_reviewed') is not True or receipt.get('semantics_reviewed') is not True:
        raise ValueError('Candidate has not passed review.')
    if (receipt['candidate_sha256'] != digest(Path(candidate).read_bytes())
            or receipt['packet_sha256'] != digest((directory / 'packet.json').read_bytes())
            or receipt['evidence_sha256'] != digest((directory / 'evidence.json').read_bytes())):
        raise ValueError('Content changed after review; a new review is required.')
    exported = export_case(directory, case, receipt['expect'])
    if leak_flags(directory, exported):
        raise ValueError('Export contains identifying text.')
    # Exclusively create the requested file. No git add/commit or overwrite.
    write_private(output, exported)
    write_private(directory / 'promotion.json', {'fixture_sha256': digest(Path(output).read_bytes()),
                                               'promoted_at': datetime.now(timezone.utc).isoformat()})
    return {'output': str(output), 'git_operations': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('stage')
    p.add_argument('source', type=Path)
    p.add_argument('--label', required=True)
    p.add_argument('--directory', type=Path)
    p.add_argument('--replacements', type=Path)
    for name in ('rewrite', 'check', 'review', 'promote'):
        p = sub.add_parser(name)
        p.add_argument('directory', type=Path)
        if name == 'rewrite':
            p.add_argument('--model', required=True)
            p.add_argument('--allow-model-send', action='store_true')
        else:
            p.add_argument('--candidate', type=Path)
        if name == 'review':
            p.add_argument('--expectations', type=Path, required=True)
            p.add_argument('--privacy-reviewed', action='store_true')
            p.add_argument('--semantics-reviewed', action='store_true')
        if name == 'promote':
            p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'stage':
        directory = args.directory or PRIVATE_ROOT / args.label
        result = {'directory': str(stage(args.source, directory=directory, label=args.label,
                                        replacements=read_json(args.replacements) if args.replacements else None)),
                  'review_required': True}
    elif args.action == 'rewrite':
        result = rewrite(args.directory, model=args.model, allow_model_send=args.allow_model_send)
    else:
        candidate = args.candidate or args.directory / 'candidate.json'
        if args.action == 'check':
            _, flags = candidate_checks(args.directory, candidate)
            result = {'leak_flags': flags, 'review_required': True}
        elif args.action == 'review':
            review(args.directory, candidate, args.expectations, privacy_reviewed=args.privacy_reviewed,
                   semantics_reviewed=args.semantics_reviewed)
            result = {'review_recorded': True, 'promoted': False}
        else:
            result = promote(args.directory, candidate, args.output)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        # Provider output, raw JSON parse context and paths can be private.
        raise SystemExit(f'Corpus step failed ({type(error).__name__}); no automatic promotion.') from None
