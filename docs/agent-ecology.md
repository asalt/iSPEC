# Agent Ecology Notes

This is architectural context, not active runtime design. The current sentinel
should stay intentionally small: observe, diff, classify, report/log, and
optionally notify.

The longer-term idea is a broader ecology of scoped peer agents with role
specialization, explicit authority boundaries, append-only evidence, and
reputation derived from observed outcomes. Possible roles include sentinel or
orchestrator, UI repair, backend maintenance, SQL/data work, reviewer,
governance/spec writer, and simulation strategist.

## Constitutional Principles

- Attention can adapt; authority should remain explicit.
- Agents may write role, spec, or policy proposals, but activation should happen
  through a governed process.
- Reputation should be role-specific, not one global "good agent" score.
- The ledger stores evidence; reputation is a derived interpretation over that
  evidence.
- Simulation evidence may inform behavioral hypotheses, but should not directly
  grant real-world authority.
- Peer deliberation should not imply authority escalation.
- Authority escalation should be scoped, signed, logged, revocable, and
  time-limited.
- Historical events should be amended by adding records, not quietly rewriting
  old ones.
- Strong branches or lineages may emerge, but old prestige should remain
  contestable by newer evidence.

## Salience And Authority Boundary

Readable weights such as `calm`, `concerned`, `risk_sensitive`, and
`exploratory` are attention or salience state. They may eventually help shape
summaries, reporting priority, memory retrieval, or calibration loops.

They must not grant permissions. They must not decide whether tmux keys can be
sent, whether commands may execute, whether Slack/network access is enabled,
whether another peer receives authority, or whether audit logging can be
disabled.

For the current single-agent sentinel, salience should remain read-only context
in tests and review packets. It can be logged beside a report, but it should not
change deterministic sentinel scoring until a separate design explicitly
introduces and tests that behavior.

## Maturity Ladder

- M0: Current read-only sentinel with deterministic observation, diff,
  classification, report/log, and simulated notification candidates.
- M1: Richer behavioral scenarios and review packets for inspecting sentinel
  decisions.
- M2: Logged read-only salience or mood vectors using the existing agent-state
  store.
- M3: Append-only evidence ledger for observations, decisions, and outcomes.
- M4: Multiple scoped peer agents with explicit capability boundaries.
- M5: Role-specific reputation, replay packets, lineage, and proposal review.
- M6: Consensus or blockchain-style governance for ratified policy and
  authority changes.

M6 is several large steps beyond the current implementation. For now, the useful
preparation is boring and testable: clean event records, actor/source
identifiers, explicit capability boundaries, append-only-ish state where
practical, and separation between observation, interpretation, notification, and
action.
