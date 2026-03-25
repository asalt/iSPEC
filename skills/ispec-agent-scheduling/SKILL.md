---
name: ispec-agent-scheduling
description: Inspect, create, update, and delete scheduled assistant jobs in this repo using the hidden internal schedule-management tools. Use when Codex needs to manage recurring assistant tasks such as weekly project updates, Slack handoff jobs, or other supervisor-seeded assistant prompts backed by `ISPEC_ASSISTANT_SCHEDULE_PATH`.
---

# iSPEC Agent Scheduling

Use this skill for the internal scheduled-assistant job file, not the separate Slack text schedule.

## Preconditions

- The hidden tools are gated behind `ISPEC_ASSISTANT_SCHEDULE_TOOLS_ENABLED=1`.
- Editing requires `ISPEC_ASSISTANT_SCHEDULE_PATH` to point at a JSON file.

## Tools

- `assistant_list_scheduled_jobs`
- `assistant_upsert_scheduled_job`
- `assistant_delete_scheduled_job`

## Workflow

1. Start with `assistant_list_scheduled_jobs` before changing anything.
2. Use `assistant_upsert_scheduled_job` for both create and edit operations.
3. Keep `timezone` explicit. Prefer `America/Chicago` for local staff-facing schedules unless the task says otherwise.
4. Keep names stable and descriptive because the job `name` is the update key.
5. When a job must publish or hand off something, set `required_tool` so the supervisor can enforce that final step.
6. Only call `assistant_delete_scheduled_job` when deletion is explicitly intended.

## Notes

- `allowed_tools` should stay narrow.
- `required_tool` is automatically added to `allowed_tools` if needed.
- The file stores weekly recurring jobs using `weekday` plus local `time`.
