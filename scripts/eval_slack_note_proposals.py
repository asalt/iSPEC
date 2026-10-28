"""Offline selected-thread replay. No Slack client, tool executor, or core DB access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ispec.assistant.service import generate_reply
from ispec.assistant.runtime_settings import max_output_tokens
from ispec.cli.env import load_env_files
from ispec.prompt import load_bound_prompt, prompt_binding, prompt_observability_context


SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "decision": {"type": "string", "enum": ["propose", "already_recorded", "no_update"]},
    "note": {"type": "string"}, "reason": {"type": "string"},
}, "required": ["decision", "note", "reason"]}


@prompt_binding("experimental.slack.note_proposal")
def proposal_prompt():
    return load_bound_prompt(proposal_prompt)


def evaluate_case(case, *, generate_reply_fn=generate_reply, constrain_json=True, output_tokens=None):
    prompt = proposal_prompt()
    # Expectations belong in the report, not in the model's input.
    inputs = {key: case[key] for key in ("date", "project_id", "thread", "existing_comments")}
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
    ]
    output_tokens = output_tokens if output_tokens is not None else (max_output_tokens() or 1024)
    if output_tokens <= 0:
        raise ValueError("Output token budget must be positive")
    options = {"max_tokens": output_tokens, "temperature": 0}
    if constrain_json:
        options["structured_outputs"] = {"json": SCHEMA}
    reply = generate_reply_fn(messages=messages, tools=None, tool_choice="none", timeout_seconds=120,
        vllm_extra_body=options,
        observability_context=prompt_observability_context(prompt, extra={"surface": "offline_eval", "stage": "slack_note_proposal"}))
    result = {"id": case["id"], "project_id": case["project_id"], "ok": False,
              "provider": reply.provider, "model": reply.model, "meta": reply.meta,
              "expected_decision": case.get("expected_decision"), "sent": False, "saved": False,
              "constrained_json": constrain_json,
              "max_output_tokens": output_tokens,
              "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}
    try:
        if not reply.ok or reply.tool_calls:
            raise ValueError(reply.error or "Unexpected tool call")
        decision = json.loads(reply.content)
        if not isinstance(decision, dict) or set(decision) != set(SCHEMA["required"]):
            raise ValueError("Invalid decision shape")
        if decision["decision"] not in SCHEMA["properties"]["decision"]["enum"]:
            raise ValueError("Invalid decision label")
        if not all(isinstance(decision[key], str) for key in ("note", "reason")):
            raise ValueError("Invalid decision text")
        if bool(decision["note"].strip()) != (decision["decision"] == "propose"):
            raise ValueError("Note must be nonempty only for a proposal")
        result.update(ok=True, result=decision,
                      matches_expected=decision["decision"] == case.get("expected_decision"))
    except (ValueError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON list of selected, bounded thread cases")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-json-constraint", action="store_true", help="Prompt-only JSON for cross-runtime comparison")
    parser.add_argument("--env-file", action="append", default=[], help="Runtime preset/overrides; later files win")
    parser.add_argument("--max-output-tokens", type=int, help="Override preset generation budget, including reasoning")
    args = parser.parse_args(argv)
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    load_env_files(args.env_file)
    cases = json.loads(args.cases.read_text())
    report = [evaluate_case(case, constrain_json=not args.no_json_constraint,
                            output_tokens=args.max_output_tokens) for case in cases]
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(not all(row["ok"] for row in report))


if __name__ == "__main__":
    raise SystemExit(main())
