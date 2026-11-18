# OLMo Template Packaging

November 12, 2026. Local packaging verification only; no model run or deployment.

## Shared Resource

`src/ispec/assistant/templates/chat/olmo3.j2` is the native OLMo Instruct
checkpoint template plus this existing startup compatibility line:

```jinja2
{%- set tools = tools | default(none) -%}
```

Its bytes match the reviewed historical fixture, including role formatting and
tool-result `environment` tags. SHA256:
`69c9c09e966a5e36f61ddfff4c655e10138178ad291ac830f4008925e6ac55f4`.
The shared base prompt, native-tool preamble and workflow wording are unchanged.
There are no vLLM package edits or model-cache modifications.

`chat_templates.py` resolves an absolute path and rejects a hash mismatch.
The wrapper's Instruct profile selects it with `VLLM_CHAT_TEMPLATE_NAME=olmo3`;
`make dev-vllm` passes `--chat-template` explicitly. Other profiles clear the
selection. The isolated gate resolves the template from its frozen working-tree
snapshot and records the served path, hash and exact launch command.

Installed-package lookup, without launching a model:

```sh
python -m ispec.assistant.chat_templates olmo3
```

A future reviewed production launcher can pass that returned path to vLLM.
Current production service definitions are unchanged. Alex reports that amms06's
driver was updated and is awaiting reboot; no remote runtime verification was
performed during this packaging pass.

## Offline Verification

The backend wheel now includes its subpackages (including namespace packages),
Markdown prompts and Jinja templates. Build from clean release source, not this
dirty working tree with its abandoned untracked experiments:

```sh
python -m pip wheel --no-deps --no-build-isolation --no-index . --wheel-dir /tmp/ispec-wheel
```

This uses already installed build dependencies. It does not install the wheel.
The checked artifact was built from HEAD plus only the reviewed packaging edits.
An isolated extraction verified all 265 Python/prompt/template files against that
source, then imported the packaged loader and loaded all 65 prompt files. The
five abandoned Jeff experiment files were excluded. Evidence is local:
`/tmp/ispec-prompt-package.sWcmaU/verification.json` and its adjacent verifier and
wheel; these temporary artifacts are not a production release.

All 70 focused backend tests passed from the clean packaging snapshot; all 127
wrapper tests passed. Backend checks cover omitted `tools`, canonical bytes, tamper rejection,
launcher command generation, system-message merging, native result roles and
rendered controller prompts with the real tokenizer. The wrapper suite executes
a fake vLLM to check actual shell expansion and refusal before launch on a bad
hash. The historical fixture remains available for historical replay.

## Separate Routing Failure

Saved case N07 in `olmo-integration-repair-v2` exhausted the router's 200 output
tokens with incomplete JSON. No valid route or hinted groups were available, so
`support.chat` retained the full authorized catalog: 51 tools. The next rendered
request used 7,236 input tokens, over the 6,000 input cap; reserving 1,024 output
tokens would also exceed the 8,192 context window by 68. The adapter refused
generation, leaving the requested revision undone.

This is a conversation-breaking reliability defect, not a reason to enlarge the
budget or relax parsing/permissions. It occurred in one of nine selected cases;
that small targeted batch does not establish a general failure rate. Packaging
did not repair it. The subsequent [November 13 repair](router-repair-20261113.md)
adds a bounded router-error fallback and verifies both previously truncated
router calls. Long-history and complete revised-conversation behavior remain
separate checks.
