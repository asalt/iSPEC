# OLMo and quantized Tulu follow-up: October 23, 2026

## Scope

Alex approved downloading both comparison options. We installed OLMo-3-7B
Instruct Q4_K_M and FP16, plus Tulu-3-8B Q4_K_M on amms06. The downloads total
about 24.0 GB; the root filesystem has about 35.8 GiB free afterward. They are
cached for future manual tests, not selected as the production bot model.

All new inference ran sequentially through Ollama 0.40.8 on amms06's 24 GiB
Quadro RTX 6000, using a temporary loopback SSH forward. No real tools, Slack
sends, project writes, production configuration changes, or service restarts
were performed. Each benchmark model was unloaded afterward; `/api/ps` was
empty at cleanup, GPU use returned to 474 MiB, and the temporary tunnel was
closed. No Think checkpoint was tested in this slice.

This extends the [earlier comparison](model-comparison-2026-10-23.md), whose
Tulu FP16 results used Transformers, not vLLM. Those earlier results remain
unchanged. Do not attribute every cross-runtime difference to quantization.

## Installed builds and memory

| Model tag | Installed bytes | Resident VRAM at 8K context |
| --- | ---: | ---: |
| `olmo-3:7b-instruct-q4_K_M` | 4,472,011,552 | 6.04 GiB |
| `olmo-3:7b-instruct-fp16` | 14,601,052,221 | 14.62 GiB |
| `tulu3:8b-q4_K_M` | 4,920,788,987 | 5.54 GiB |

Memory is an Ollama `size_vram` snapshot, not peak use or whole-card usage.
The OLMo packages share the same chat-template hash and published Instruct
variant. This is our closest controlled precision comparison, although the
packages do not establish an exact shared Hugging Face source commit.

Manifest digests:

- OLMo Q4: `ea72df8c85d75c6783ab0d00278803c53a634fc9d0fe38d3fd4ab6f35adbdbb2`
- OLMo FP16: `10bc19cdab78f7fc34d3637dba6a6e261e868ebb157c624d90995cb4293d27c5`
- Tulu Q4: `7119be59a531ffd16fe44335639911542c6136a2544f7f80e5fd687f41c8ee97`

Tulu Q4_K_M is **not MXFP4**. The published
[Tulu tags](https://ollama.com/library/tulu3/tags) provide Q4_K_M, Q8_0, and FP16
builds, not an MXFP4 build. MXFP4 would be a separate conversion/runtime
experiment; this trial does not answer its quality question. The selected
[OLMo tags](https://ollama.com/library/olmo-3/tags) explicitly identify Instruct;
the default 7B/Think tag is a different checkpoint, not a reasoning toggle here.

## Frozen prompt-only comparison

We reused the exact 35 requests and separate expectation manifest from
`tests/behavioral/local/model-comparison-20261023-compact-all/`. Input hashes
were checked during scoring. These use temperature 0, context 8192, and budgets
of 160 approval, 512 note, and 256 grounding tokens. The JSON schema is visible
in the prompt; native constrained decoding is not used in this comparison.
Reasoning is unset for these non-thinking models.

| Model | Approval exact labels / 20 | False approvals at 0.75 | Missed approvals | Note checks / 10 | Grounding / 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLMo Instruct Q4 | 12 | 0 | 1 | 7 | 2 |
| OLMo Instruct FP16 | 12 | 0 | 1 | 6 | 2 |
| Tulu Q4 | 16 | 2 | 0 | 7 | 1 |

All 35 outputs from each model were schema-valid. Each separate no-pending
control skipped inference. Exact labels distinguish several no-write outcomes;
a safe label disagreement is not the same failure as an unauthorized approval.

| Model | Approval median / p95 | Note median / p95 | Grounding median / p95 |
| --- | --- | --- | --- |
| OLMo Q4 | 1.02 / 1.17 s | 0.78 / 1.70 s | 0.63 / 0.77 s |
| OLMo FP16 | 1.47 / 1.70 s | 1.76 / 2.35 s | 1.11 / 1.29 s |
| Tulu Q4 | 0.67 / 0.82 s | 0.68 / 0.80 s | 0.60 / 0.62 s |

The first OLMo requests included loading: Q4 took 4.32 s (3.33 s loading),
FP16 7.16 s (5.75 s loading). FP16's cold answer exceeds the current six-second
approval deadline. Warm inference latency is not full conversation latency.

Notable outputs from manual review:

- Both OLMo precisions withheld approval for the explicit commit request with
  an added aside about testing approval. Their other label disagreements stayed
  on the no-write side. FP16 did not improve this prompt-only behavior.
- Tulu Q4 approved a social closing at confidence 1.0 and tentative wording at
  0.9. Its larger exact-label score does not make it the safer authorizer.
- Both OLMo versions proposed a note for unrelated server-IP chatter and missed
  the already-reported profiling/BioID information in a contextual thread.
- Tulu Q4 followed an instruction embedded in quoted thread content, adding
  invented completed-analysis and saved-successfully claims. Both OLMo versions
  preserved the real sample receipt in that particular injection case.
- The note fragment checks are imperfect: an imperative upload task is a valid
  pending note but lacks the expected marker words. The FP16 phrase "No raw
  files have been uploaded" also trips a naive forbidden substring. Raw scores
  are preserved, with these caveats, rather than retuned after seeing outputs.
- The narrow new-comment `write_succeeded` boolean was correct in 5/5 OLMo
  cases but only 2/5 Tulu Q4 cases. Actual executor outcomes must remain the
  truth source; the broader state/attention classifications are less reliable.

## Production-shaped classifier and native tools

These are separate from the prompt-only matrix. The existing classifier replay
uses the actual classifier prompt, a native JSON-schema request, a 160-token
budget, a six-second deadline, and policy at the current 0.75 approval threshold.
Tickets exist only in memory; no tool executor or project database is invoked.

| Model | Replay tests / 21 | Valid classifier outputs / 20 | False tickets | Missed approvals | Warm median |
| --- | ---: | ---: | ---: | ---: | --- |
| OLMo Q4 | 17 passed, 4 failed | 17 | 0 | 4 | 1.11 s |
| OLMo FP16 | 20 passed, 1 failed | 20 | 0 | 1 | 1.64 s |
| Tulu Q4 | 20 passed, 1 failed | 20 | 1 | 0 | 0.70 s |

The 21 tests include the no-pending control. They assert valid output and the
save/no-save policy result, not agreement on every no-write label. Exact labels
on the 20 inferred cases were 8, 11, and 14 respectively.

OLMo Q4 returned `confidence: 99` for three approvals; the validator rejected
them instead of guessing that 99 meant 0.99. FP16 used valid confidence values
in those cases. Both still missed the approval-with-testing-aside case.
Tulu Q4 would have issued a ticket for a question about how the confirmation
protocol works, at confidence 0.85. This is a promotion-blocking false approval,
even though the total pass count equals OLMo FP16's. Nothing was actually saved.

All three new models passed exact-OK and bounded-denial smoke tests. All three
failed the one native lookup/result round-trip test, for different reasons:

- OLMo Q4 and FP16 accept native tools, but returned prose requesting a project
  ID even though project 42 was supplied. Raw API replay reproduced this; there
  was no wire-level tool call for the adapter to parse. This could involve the
  model/template/prompt combination; it is not proof OLMo can never call tools.
- The published Tulu Q4 package rejects tools with HTTP 400 and "does not support
  tools." Its metadata advertises completion only. This is a packaging/runtime
  boundary, not evidence that every Tulu deployment lacks tool capabilities.

The earlier GPT-OSS build passed that same native round trip and scored 8/10
compact note checks with zero false approvals in its prompt-only trial. Its
production-shaped classifier still had 4/21 failures (missing confidence or
exhausted output budget), all closed. It remains promising, not fully validated.

## Evidence, verification, and next step

New raw results, full model metadata/templates, request hashes, scores, pytest
logs/XML, and raw tool probes are under the ignored directory
`tests/behavioral/local/model-runtime-20261023/`. Prepared inputs and older results
remain in their original directories. The new tests exercised 105 prompt-only
inferences, 60 production-shaped classifier inferences plus three skipped-gate
controls, and nine native smoke tests; raw diagnostic replays are additional.

The final deterministic focused suite passed 132 tests in 5.77 seconds, with two
existing FastAPI lifespan warnings; its log is `deterministic-tests.log` in the
new results directory. It covered the evaluator, native runtime protocols,
proposal evaluation, approval policy, usage logging, Slack thread context, and
prompt modules. The native smoke runs had pytest `record_property`/JUnit
compatibility warnings. No full repository test run or production activation
was performed.

The only evaluator change was making the reasoning setting explicit rather
than always defaulting to `low`; its mocked transport test covers both omitted
and explicit settings. Reproduction uses `scripts/eval_model_matrix.py` and the
two existing opt-in inference test modules documented in the behavioral README.

Keep the deployed configuration unchanged. For the next comparison, prioritize
held-out natural approvals and source-grounded note drafts, then diagnose the
native tool/template boundary and classifier schema contract before promotion.
These are small single-pass diagnostics, not calibrated confidence estimates or
a general model ranking. The narrated two-voice replay video remains a TODO in
the [benchmark plan](local-model-benchmark-plan.md), with no video generated yet.
