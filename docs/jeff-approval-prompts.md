# Jeff approval prompts

These are isolated experiments, not the live approval path. Each request combines
three editable Markdown files: a question, context template and choice definitions.
The existing prompt loader handles the frontmatter; only the body reaches Jeff.

## Questions

All paths below are relative to this document. The longer original remains a
control, not a new instruction added to the short questions.

| CLI name | Question file | Question |
| --- | --- | --- |
| `affirmation` | [affirmation](../src/ispec/prompts/experimental.project_comment.jeff_approval.affirmation.md) | From this conversation, is the user's last reply an affirmation to save the project comment? |
| `authorize` | [authorize](../src/ispec/prompts/experimental.project_comment.jeff_approval.authorize.md) | Does the user's last reply authorize saving the project comment? |
| `approval` | [giving approval](../src/ispec/prompts/experimental.project_comment.jeff_approval.approval.md) | Is the user's last reply giving approval to save the project comment? |
| `save_now` | [save now](../src/ispec/prompts/experimental.project_comment.jeff_approval.save_now.md) | Does the user want this draft saved now? |
| `baseline` | [original control](../src/ispec/prompts/experimental.project_comment.jeff_approval.md) | Original longer question with explicit context guidance. |

Shared [choice descriptions](../src/ispec/prompts/experimental.project_comment.jeff_approval.choices.md)
keep `yes`, `no`, `unsure` in that order. The earlier layout diagnostic's
[descriptive choices](../src/ispec/prompts/experimental.project_comment.jeff_approval.choices_descriptive.md)
are retained only for reproducing that test, not varied in the short-question sweep.
The fixed experimental policy uses probability >=0.75 and margin >=0.20.
Changing wording does not change permission, pending-state or result validation.
The [binary choice file](../src/ispec/prompts/experimental.project_comment.jeff_approval.choices_binary.md)
keeps the exact same yes/no descriptions and omits unsure. `--choices binary`
selects it for a single variant; this is measured inference, not renormalization.

## Contexts

Only the initial note request, assistant draft/save question and latest reply are
included. There is no older history and no repetitive length padding in this sweep.

| CLI name | File | Jeff v1.3 behavior |
| --- | --- | --- |
| `transcript` | [labeled text](../src/ispec/prompts/experimental.project_comment.jeff_approval.context_transcript.md) | State text, then question, then options. |
| `messages` | [message array](../src/ispec/prompts/experimental.project_comment.jeff_approval.context_array.md) | Array serialized to JSON text, then question, then options. |
| `live_last` | [latest reply last](../src/ispec/prompts/experimental.project_comment.jeff_approval.context_live_last.md) | Question, initial request/draft, options, then latest reply alone. |
| `reply` | [latest reply only](../src/ispec/prompts/experimental.project_comment.jeff_approval.context_reply.md) | Only the final user reply as plain text, with no preceding ask. Diagnostic control, not a complete approval context. |
| `object` | [old nested object](../src/ispec/prompts/experimental.project_comment.jeff_approval.context_messages.md) | Historical control: question, project ID, options, then the whole message array. |

`role` keys inside a message array are data, not native conversation roles. The
installed checkpoint's `live-last` setting handles object fields in insertion
order. Keep `latest_user_reply` last in that template. These are not all equivalent
serialization choices: object layouts also change where the content is placed.
JSON context fields are escaped before template substitution; text is not truncated.

## Preview and compare

From the backend checkout, preview without an endpoint or model:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.eval_jeff_approval \
  --preview example_confirm --question authorize --context transcript
```

Replace the names to inspect another combination. `--preview short_retract` shows
one of the fresh cases. Do not edit cases after inspecting results and call them
held out again; add a new frozen set instead.

With an explicitly started loopback Jeff server:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.eval_jeff_approval \
  --prompt-sweep --url http://127.0.0.1:18765 \
  --output-dir tests/behavioral/local/NEW-short-prompts
```

This makes 288 calls: 4 questions x 4 context layouts x 18 cases. Six cases are
seen diagnostics and twelve are fresh for this sweep, reported separately. The
runner rotates combination order across cases and saves raw requests/responses,
token counts, latency, prompt files, source snapshots and hashes. It never starts
a model, executes a tool or writes a project comment. Output directories must be
new. An exit status of 1 records errors, false approvals or missed approvals.

Without `--prompt-sweep`, the original 35-case corpus remains the default, including
its synthetic length controls. `--question` and `--context` choose a single
combination for that corpus. `--layout-probe` reproduces the earlier 2x2 diagnostic.

The newer `--choice-sweep` compares 2 questions (`authorize`, `approval`),
2 contexts (`live_last`, `reply`), and 2 choice sets on 36 cases: 18 seen and
18 fresh variations. Use repeated `--case ID` to limit a run. Endpoint or contract
failure stops the sweep with partial evidence, rather than counting unavailable
requests as model answers.

For automatic port selection and owned-server cleanup, use
[`run_jeff_trial`](jeff-approval-replay.md#thresholds-and-realistic-ambiguity).
The [checkpoint guide](jeff-approval-replay.md) also documents before/after
conversation exports, follow-up review branches, and offline threshold comparisons.

See [evaluation results](jeff-approval-evaluation-20261104.md) for measured behavior,
limitations and the no-promotion decision.
The [November 5 comparison](jeff-choice-evaluation-20261105.md) adds binary/ternary,
question wording, threshold and ambiguity results plus runtime/replay evidence.
