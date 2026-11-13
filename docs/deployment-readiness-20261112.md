# Commit And Deployment Readiness

November 12, 2026. Source checkpoint only; nothing pushed or deployed.
The current behavioral decision remains **do not promote OLMo yet**.

## Committed Capabilities

| Repository | Commit | Scope |
| --- | --- | --- |
| Backend | `9fb7c1f` | Repair omitted-provider planner resolution, with regression tests |
| Backend | `026a188` | Exact confirmation cards, shared atomic text/button saves, scoped local Undo, web/Slack adapters |
| Backend | `9071b99` | Shared application prompt with native schemas, OLMo system merge and native results, serving-tokenizer budget guard |
| Backend | `29af63a` | Working-tree snapshots, isolated application gate, startup checks and failure watcher |
| Wrapper | `9013687` | OLMo Instruct profile selects its native-tool preamble |
| Frontend | `4d11116` | Separate cards with Save, Cancel, Undo and mocked browser coverage |

Documentation preserves the baseline, failed full run and repaired focused run
separately. Private traces, databases and generated reports remain ignored.
The dropped Jeff tool-use experiment remains uncommitted, outside this series.

## Verification

- Clean backend source archive at `29af63a`: 257 focused tests passed, with no
  skips; includes real HTTP adapters using synthetic databases and offline prompt
  rendering with the real OLMo tokenizer. Model HTTP replies are stubbed.
- Another 16 template, runtime-snapshot and prompt-observability tests passed.
- The independent staged card commit passed 165 checks before committing.
  These are a subset, not extra coverage to add to the final total.
- Wrapper: all 126 tests passed, including release and deployment test doubles.
- Frontend: production build, six mocked Cypress checks and two Node test files
  passed. Saved cards were inspected at desktop and mobile sizes. No browser
  requests reached production; the temporary preview was stopped.
- Existing FastAPI lifespan deprecations, Vite CommonJS and bundle-size warnings
  remain. No new real-model test was run for commit cleanup.

## Before Deployment

1. **Make the template deployable.** Move the exact upstream-plus-`tools`-default
   template into packaged application resources and resolve it from the launchers
   with explicit `--chat-template`. Tests and services must use the same bytes/hash.
   Currently the successful runner uses a repaired local checkpoint template;
   the matching committed copy lives under behavioral fixtures. Merely choosing
   the model profile does not ship that local cache correction.
2. **Finish integration coverage.** Consolidate the test renderer, replace the
   fixed tokenizer stub in rendered-controller tests, and test long history and
   router failure against both the 6,000 input budget and context/output reserve.
   Keep separate tests for correctly refused oversized requests. A blocked
   conversation is not a success. Compare/choose card delivery also needs coverage
   before enabling that path.
3. **Resolve behavior, then rerun selected cases.** P01/R01 re-prepare instead of
   saving, R01 includes printer chatter, N07 expands a failed router to 51 tools,
   and N09 describes a real cancellation inaccurately. Keep approval thresholds
   and permission checks unchanged. The repaired run's zero unapproved writes
   included only one true-approval classifier call; it is not negative-safety proof.
4. **Prepare an explicit runtime release.** The tracked production service is
   still Tulu/checkpoint/parser-specific. Prepare a reviewed OLMo configuration
   and retain a tested Tulu rollback; do not blindly apply these shared prompt/
   workflow changes to the current production Tulu and assume equivalence. Recheck
   amms06 occupancy and the previously reported driver/reboot issue before startup.
5. **Stage from clean exact commits.** The existing `deploy/mariana/stage.py`
   requires clean wrapper/backend/frontend sources and a matching frontend build
   receipt. Use clean release checkouts, leaving abandoned experiments untouched
   here. Record all three commits and runtime/template identities. Verify a fresh
   backup and schema initialization on a disposable restored database, including
   `project_comment_card`, before any production migration.
6. **Authorize activation separately.** Verify Slack Interactivity, receiver
   uniqueness, card delivery/actions, project ACLs and backup/rollback procedures.
   Keep legacy writeback off for local Undo; FileMaker reversal is not implemented.
   No live service status, credentials, production database or Slack app settings
   were inspected or changed during this commit pass.

Code rollback alone does not undo database changes. The comment-card feature and
OLMo model promotion are separate deployment decisions.
