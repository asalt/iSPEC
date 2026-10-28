# Private snapshots to reviewed fixtures

Nothing is automatically published, added to Git, or committed. A model rewrite
is a private **candidate**, not certified anonymization. Synthetic reconstruction
is preferable when distinctive scientific details would still identify a source.

## Stages

1. Select a small snapshot explicitly: the existing assistant-session extractor,
   a Slack history/thread JSON export, or a hand-selected `{ "messages": [...] }`
   object. No bulk archive discovery. Roles/content and Slack user/text formats
   are accepted. Add any separate latest request as a message before staging.
2. Stage under ignored `tests/behavioral/local/corpus/`. Raw input, identity maps
   and receipts are private (directory 0700, files 0600). Optional literal
   replacements provide consistent fictional names/IDs before model access.
3. Rewrite manually or opt into a bounded Codex worker with an explicitly chosen
   model. Only the selected packet and minimal tool outcomes enter its prompt.
   Existing Codex authentication is used without printing credentials. This sends
   the packet to the configured model provider; do not opt in for material that
   must remain on-device. The worker uses read-only mode, no shell tool or web
   search, no project config, and an ephemeral session. It is not a privacy proof.
4. Review source versus candidate for identifying details **and** preserved
   meaning. Preserve ambiguity, negation, quoted instructions, sample counts,
   scientific distinctions, failures and false success claims. Tool names/success
   booleans are preserved separately, never rewritten by the model. Author expected
   behavior independently; the rewriter does not supply the answer key.
5. Record review, then explicitly export a fixture. Hashes invalidate review after
   edits. The export contains no original IDs, paths, identity map or raw-source
   hashes. Leak checks are deliberately incomplete; human review remains necessary.

```bash
python scripts/prepare_behavioral_case.py stage SELECTED.json --label pending-receipt \
  --replacements PRIVATE-REPLACEMENTS.json
python scripts/prepare_behavioral_case.py rewrite tests/behavioral/local/corpus/pending-receipt \
  --model YOUR_APPROVED_CODEX_MODEL --allow-model-send
python scripts/prepare_behavioral_case.py check tests/behavioral/local/corpus/pending-receipt
python scripts/prepare_behavioral_case.py review tests/behavioral/local/corpus/pending-receipt \
  --expectations EXPECTATIONS.json --privacy-reviewed --semantics-reviewed
python scripts/prepare_behavioral_case.py promote tests/behavioral/local/corpus/pending-receipt \
  --output tests/behavioral/local/reviewed-receipt.json
```

Replacements are a private JSON object, e.g. `{"Original Name": "Robin", "1499": "42"}`.
Keep the case label non-identifying. The candidate schema permits only label,
tags and ordered message IDs/roles/content. Known emails, URLs, IPs, paths, Slack
identifiers/tokens and retained replacement strings block review/export. Unknown
names, rare combinations and paraphrased identifiers can still escape detection.
Credentials are never acceptable model inputs; obvious tokens/private keys block
the rewriting call. Provider failure details are not echoed to the terminal.

Expectations use the existing `tests/behavioral/local_cases.py` contract, for
example `{"assistant_message_count": 1}`. A known-failure case can intentionally
fail its desired-behavior assertions. The recorded reply is not the answer key.
Retained tool outcomes describe original evidence, not a newly executed test.

Keep exported files local until separately reviewed for Git inclusion. Existing
`.gitignore` allowlisting protects the fixture directory too; explicit publication
is still required. Before a video, review the exact exported text again. Caption
voices as Prompt, Model, or Reviewer; do not present rewritten dialogue as a literal
recording. Comparisons must retain prompt/model/version and measured latency.

The Codex wrapper follows the documented
[non-interactive structured-output workflow](https://learn.chatgpt.com/docs/non-interactive-mode).
No new model, credentials, Slack send, project write or production setting is
installed or activated by this pipeline.
