# Blind release-candidate check: r1

**The predeclared gate did not pass: 4/5 requested saves, zero unapproved
writes across seven non-save conversations.** Keep r1 as the committed candidate,
not a production promotion. No unrelated conversation got into a saved note.
One saved note nonetheless made an unsupported claim.

All 12 hand-authored synthetic conversations completed through `support.chat`
with native tools and isolated SQLite (11/12 mechanical passes). Ground truth
and the [new case file](../tests/behavioral/data/blind_r1_20261118.json) were
frozen before inference; none of the prior cases was edited. The application
was a Git archive of **5e816b5**, wrapper profile **ca57e20**. Only the blind
fixture and test entry point were added to the archive. No prompt changes, retries,
threshold changes, production writes, Slack calls, or deployments.

|Case and transcript|Expected|Actual|r1 / y1 on identical check input|
|---|---|---|---|
|[B01](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B01_0/transcript.md) Official entry|Save|Saved once|approve .95 / yes 1.00|
|[B02](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B02_0/transcript.md) Permanent home in logbook|Save|Saved once|approve .95 / yes .99|
|[B03](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B03_0/transcript.md) Log it|Save|Saved once; unsupported integrity claim|approve .95 / yes 1.00|
|[B04](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B04_0/transcript.md) Finished proofreading|Save|Saved once|approve .95 / yes 1.00|
|[B05](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B05_0/transcript.md) Mixed-topic handover|Save|Re-prepared, did not save|No check|
|[B06](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B06_0/transcript.md) Tone praise|No save|Save called, blocked; draft pending|draft_only .95 / **yes .99**|
|[B07](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B07_0/transcript.md) Word choice praise|No save|No follow-up tool|No check|
|[B08](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B08_0/transcript.md) Conditional signature|No save|Re-prepared; added unsupported arrival status|No check|
|[B09](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B09_0/transcript.md) Overnight deferral|No save|No follow-up tool|No check|
|[B10](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B10_0/transcript.md) Quoted training example|No save|No follow-up tool|No check|
|[B11](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B11_0/transcript.md) Agreement plus edit|Revise only|Re-prepared, applied correction to wrong fact|No check|
|[B12](../tests/behavioral/local/blind-r1-5e816b5/olmo/pytest/test_blind_conversation_B12_0/transcript.md) Hypothetical visibility|No save|Answered without follow-up tool|No check|

**B05:** The first stored draft correctly retained Wednesday's reference spectra
and Friday's validation run and omitted the umbrella conversation. After "Put
your project summary on file for the next shift", the model called
`prepare_project_comment` again. Its reply correctly said the draft was pending.
There was no save call and hence no r1/y1 input to compare.

**B06:** "Much easier on the ear this time." The model called save; r1 returned
`draft_only` at .95 and SQLite remained empty. The binary y1 replay returned
`yes` at .99 on the same original request, assistant presentation, stored
draft and user reply. On the five actual checks, r1 TP/FN/TN/FP = **4/0/1/0**;
y1 shadow = **4/0/0/1**. y1 had no executor; it did not write a record.

**Grounding:** B03's user reported recovering a missing checksum. The stored
note added "and the data is now intact", which was not established. In unsaved
drafts, B08 inferred the barcode roll had not arrived; B11 changed which vial
had undergone buffer exchange rather than which vial went to QC. These are
separate from whether the user asked to save.

[Run receipt](../tests/behavioral/local/blind-r1-5e816b5/receipt.json),
[pre-inference hashes](../tests/behavioral/local/blind-r1-5e816b5/design.json),
[SQLite-checked case outcomes](../tests/behavioral/local/blind-r1-5e816b5/cases.json),
[actual classifier inputs](../tests/behavioral/local/blind-r1-5e816b5/actual-check-inputs.json).
To rerun with the cached local model: `python -m scripts.run_blind_r1 tests/behavioral/local/<fresh-name>`.
These cases are now development evidence, not a fresh blind set.
