# Local model comparison: context repair first

Planning snapshot: October 19, 2026. No model installation, download or switch
is part of this change.

## Baseline and serving preflight

Keep Tulu 3 8B with the repaired Slack thread context/router prompt as the
baseline. Compare the same scenarios against Qwen3.5-4B with thinking disabled,
then enabled. Do not combine a model swap with a new context or policy change.

Observed host: Quadro RTX 6000, 24 GiB; approximately 23 GiB allocated while
Tulu runs. Allocation includes serving caches and does not establish model
weight size. Do not try two GPU servers concurrently with the present setup.
The serving environment has vLLM 0.13.0, Transformers 4.57.3 and Torch 2.9.0;
its registries lack Qwen3.5 support. The current Make launcher also explicitly
uses `llama3_json`, which is not the Qwen tool parser.

Use a separate, pinned environment after approval, without upgrading the
working environment. Verify GPU/kernel support on this older Turing GPU before
claiming it can serve the model; parameter count alone is insufficient. Test
a supported dtype and a bounded context (initially 8K, not the advertised
maximum). Record engine, model revision, dtype, context and sampling settings.

The [official Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B) supports
thinking on/off with `chat_template_kwargs.enable_thinking` and documents the
`qwen3` reasoning parser plus `qwen3_coder` tool parser. Consult the current
[vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
when selecting the isolated runtime. Keep reasoning separate from final text
and validate tool parsing before judging conversational behavior. A tiny
classifier output budget can exhaust itself on thinking before producing JSON;
start bounded router/classifier calls with thinking off.

## Small, reviewable scenario set

Begin with 15-25 curated cases, not a large synthetic benchmark project. Reuse
`tests/behavioral/data/slack_reported_thread.json` and the existing sandbox
datastore. Keep raw/private conversation extracts under the ignored
`tests/behavioral/local/`; check in only reviewed synthetic/anonymized fixtures.

- Thread-dependent project note: identify 1131 and both prof/BioID reporting.
- Simple project lookup and an explicit assistant-history request.
- Pending note approval, natural confirmation, denial, revision and deferral.
- An old quoted approval followed by a current social reply: no new write.
- Task wording must not become completed work; preserve corrective-note intent.
- Missing or conflicting project identity: ask rather than choose arbitrarily.
- Tmux discovery/read summary versus explicit relay with optional Enter.
- Failed or absent tool outcome: never claim a successful save or send.
- Long/truncated thread context, repeated delivery and missing-context cases.

Use temporary databases and stub every write/send/tmux action. Start with the
same production-shaped prompts and tool schemas, not isolated chat questions.
The existing `test_slack_thread_router_vllm_live.py` provides an opt-in,
inference-only router check. The existing support benchmark runner provides
timing/report conventions, but must not be aimed at live mutation scenarios;
first adapt/reuse its measurements with the sandbox boundaries above.

## Measurements and decision

Record scenario expectation, selected tools/arguments, project/target identity,
authorization boundaries, truthful outcome reporting and clarification quality.
Also record total turn time, time to first token when available, model/tool call
counts, input/output/reasoning tokens when reported, and truncation/timeouts.
Distinguish parser/runtime failures from semantic mistakes. Cap and record the
reasoning/output budget; compare latency as well as quality, not token caps alone.

Run each case 2-3 times where practical, retain case-level results and manually
review differences. A small set is a regression screen, not a reliable population
accuracy estimate. Unauthorized writes/sends or invented success block promotion.
For basic tasks aim below a minute, flag over 90 seconds, and investigate any
multi-minute result. Keep the current model until the candidate demonstrates
useful gains without regressions on these boundaries. A larger holdout set can
follow from actual use; no need for another orchestration layer.

## Follow-up: replay video (deferred)

Alex suggested a small narrated review video of selected benchmark cases. Keep
this deferred until the comparison and prompts stabilize; no video generation
is part of the current model tests.

- Render simple text slides (Beamer or a similarly small text renderer), not a
  large animation project.
- Use distinct visual labels and two voices: one for the prompt/user context and
  one for recorded model output. `spd-say` can preview the narration; use a
  file-producing TTS path when building the synchronized video soundtrack.
- Show model/runtime/precision, relevant input, actual output, timing, and the
  expected-versus-observed outcome. Label excerpts and truncation explicitly.
- Use only recorded text. Show reasoning only if the local model returned it and
  it was retained; do not invent an explanation of the model's internal process.
- Start with a few synthetic cases: social reply mistaken for approval, a
  task-like note incorrectly made complete, and a quoted instruction attack.
  Keep private Slack content out of a shareable video unless separately reviewed.
- Retain captions and a case-ID/prompt-hash link to the underlying replay so the
  video stays a review aid, not a substitute for test evidence.
- Include approvals and denials side by side across models, with the actual
  policy outcome and whether a write occurred. Render from saved behavioral
  receipts rather than calling models again to create the video.
- Keep timestamped runtime/device snapshots in local logs, including total,
  used, and free GPU memory. Label inventory-only snapshots separately from
  combinations with passing test receipts; no blind updates or machine images.
