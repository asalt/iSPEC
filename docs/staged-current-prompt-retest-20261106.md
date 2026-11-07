# Current-prompt baseline, November 6

This is a fresh run of the current checkout, not a restatement of historical
failures. It used the restored default prompts, retained context-refresh fix,
and tool-result compatibility messages. No prompt, parser, threshold, permission
gate, model weights or production settings were changed during this run.

## Results

| Conversation | Tulu | OLMo Instruct |
| --- | --- | --- |
| explicit_save | Pass | Pass |
| natural_approval | Fail | Fail |
| revision_then_approval | Fail | Fail |
| wording_only | Pass | Pass |
| cancel | Pass | Pass |
| social_close | Pass | Pass |
| mixed_thread | Fail: saved unrelated chatter | Fail: no preparation/save |
| held_approval | Fail | Fail |
| held_denial | Pass | Pass |
| held_revision_defer | Pass | Pass |
| held_mixed_thread | Fail | Fail |

Each model passed 6/11 mechanical tests. Tulu completed two of six intended
fixture saves, but one included WiFi/lunch chatter and failed the grounding
assertion. OLMo completed one of six. Both completed the explicit-save case
with the intended text. In particular, Tulu's historical explicit-save failure
does not describe this run.

The identical fail lists conceal different behavior. For natural approval,
Tulu omitted the note text from its initial presentation, whereas OLMo
registered and displayed it correctly but never called save after approval.
For fresh approval wording, Tulu superseded a draft then used the old ID;
OLMo delayed preparation until the approval turn and attempted a same-turn save.
Neither of these cases reached the approval classifier.

Both models displayed revisions without registering them. The deferred-revision
tests still pass their no-comment-write assertions despite stale stored drafts.
OLMo's denial case also falsely claimed cancellation while retaining a pending
draft. These are separate behavioral findings, not extra pytest failures.

No non-save conversation inserted a comment in this run. Wording-only agreement
did not trigger a save call or approval classification. That is a current
end-to-end result, not a direct retest of the earlier classifier false approval.
Response guards intervened in four Tulu turns and three OLMo turns; intervention
alone is not a semantic verdict about either model or the guard.

## Evidence and Reproduction

Current-only viewer:
`tests/behavioral/local/staged-20261106/current-prompts-retest/postmortems.html`.
Each inference includes readable role-tagged input, API messages, returned
content and parsed tool calls. Review annotations live alongside the report.
Historical reports remain separate and unchanged.

22 conversations, 48 user turns, 133 inference requests (70 Tulu, 63 OLMo).
All inference HTTP responses were 200. Wall time including both cold starts was
581 seconds. Maximum observed prompt/completion tokens: Tulu 2369/195,
OLMo 5888/78. These are repeated regression cases, not new held-out evidence.

Local Quadro RTX 6000; cached FP16 Tulu 8B and OLMo 3 7B Instruct, vLLM 0.32.0,
8192 context, 512 maximum output, temperature zero, three tool calls per turn.
Each model was loaded once. Tests used actual `support.chat` and fixture tools,
isolated SQLite databases, no inherited application env files and loopback-only
HTTP guards. The launcher saved source snapshots and hashes in `receipt.json`;
`source_changed` is empty. Both owned runtimes exited, test port 18030 closed,
and the final host GPU check showed no compute processes. The pre-existing
Ollama listener was left alone. No Slack or legacy writes occurred.

The bounded local launcher accepts an optional pytest selection:

```sh
../.venv/bin/python tests/behavioral/local/staged-20261106/run.py \
  tests/behavioral/local/staged-20261106/next-distinct-trial \
  'natural_approval or revision_then_approval or wording_only'
```

Use a new output directory each time. This launcher loads both models; for
multiple prompt arms on one model, reuse one owned runtime and the existing
`trial_environment` helper instead of repeatedly loading weights.

## Next Focused Trials

Do not keep rerunning the whole suite while tuning. Select one failure and a
relevant negative control; change one prompt component, preserving all gates.
Start with natural approval (presentation versus follow-through), then revision
registration, then mixed-thread note grounding. Keep explicit_save as a known
successful control and wording_only as a non-save control. Recheck the full
suite once a candidate is ready, including semantic review of apparent passes.

These results support targeted debugging, not production promotion or a general
claim that either model is superior.
