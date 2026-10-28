"""Prepare, run, and score bounded synthetic model probes without tool execution.

Inference uses only a loopback vLLM/Ollama endpoint, optionally SSH-forwarded.
The prepared requests also work with eval_cached_transformers_chat.py. Expected
answers stay in a separate manifest. This is not a full controller benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/behavioral/data"


def digest(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def prepare(directory, *, note_prompt="baseline", tasks=None, data_dir=DATA):
    # Rendering must not register prompts in the normal application's database.
    os.environ["ISPEC_PROMPTS_DB_PATH"] = str(directory / "prompts.db")
    from ispec.assistant.project_comment_approval import (
        ProjectCommentGate, detect_project_comment_trigger, stable_hash,
    )
    from ispec.assistant.small_classifier import build_small_classifier_task
    from eval_slack_note_proposals import SCHEMA, proposal_prompt
    from ispec.prompt.loader import render_prompt

    requests, manifest = [], []

    def add(case_id, task, messages, schema, budget, expected):
        # The same explicit output contract is visible to both unconstrained runtimes.
        messages[0]["content"] += "\nRequired JSON schema: " + json.dumps(schema, sort_keys=True)
        request = {"id": case_id, "messages": messages, "max_tokens": budget}
        requests.append(request)
        manifest.append({"id": case_id, "task": task, "schema": schema,
                         "messages_sha256": digest(messages), "expected": expected})

    corpus = read_json(data_dir / "project_comment_approval_cases.json")
    for case in corpus["cases"]:
        if not case["pending"]:
            manifest.append({"id": case["id"], "task": "gate", "expected": {"skip": True}})
            continue
        prior = corpus["prior_assistant_message"]
        gate = ProjectCommentGate(eligible=True, kind="pending_save_confirmation", reason="synthetic_fixture",
            project_id=42, pending_tool="create_project_comment", session_id="eval", thread_key="eval",
            current_turn_id="1", prior_assistant_message_id=1, pending_draft_hash=stable_hash(prior),
            source_message_hash=stable_hash(case["reply"]))
        trigger = detect_project_comment_trigger(message=case["reply"], legacy_confirmation_kind="none", legacy_save_requested=False)
        prepared = build_small_classifier_task("project_comment_approval", payload={
            "user_message": case["reply"], "prior_assistant_message": prior, "state_gate": gate.to_dict(),
            "trigger": trigger.to_dict(), "lexical_features": trigger.to_lexical_features(),
            "pending_action": "create_project_comment", "focused_project_id": 42,
        })
        add(case["id"], "approval", prepared.messages, prepared.schema, prepared.task.max_tokens,
            {"label": case["expected_label"]})

    for case in read_json(data_dir / "model_note_cases.json")["cases"]:
        inputs = {key: case[key] for key in ("date", "project_id", "thread", "existing_comments")}
        prompt = (proposal_prompt() if note_prompt == "baseline" else
                  render_prompt("experimental.slack.note_proposal.compact"))
        add(case["id"], "note", [
            {"role": "system", "content": prompt.text},
            {"role": "user", "content": json.dumps(inputs)},
        ], SCHEMA, 512, {key: case[key] for key in ("expected_decision", "required_any", "forbidden") if key in case})

    schema = {"type": "object", "additionalProperties": False, "properties": {
        "state": {"type": "string", "enum": ["active", "completed", "waiting_for_human", "draft_only", "failed", "unknown"]},
        "write_succeeded": {"type": "boolean"}, "needs_human": {"type": "boolean"},
        "reason": {"type": "string"},
    }, "required": ["state", "write_succeeded", "needs_human", "reason"]}
    for case in read_json(data_dir / "model_grounding_cases.json")["cases"]:
        add(case["id"], "grounding", [
            {"role": "system", "content": "Classify this synthetic observation. Return only JSON with a short reason. state describes the operation asked about: completed if it succeeded, draft_only for an unsaved note/summary, waiting_for_human if it has not executed and awaits explicit approval. A finished run stays completed even when its next stage needs a decision. write_succeeded means a NEW project-comment database write succeeded, not file outputs, prior saved comments, or prose. Only actual tool results prove that. needs_human means input was explicitly requested, not a recommendation you infer. Do not perform any action."},
            {"role": "user", "content": case["context"]},
        ], schema, 256, case["expected"])
    if tasks:
        manifest = [row for row in manifest if row["task"] in tasks]
        selected = {row["id"] for row in manifest}
        requests = [row for row in requests if row["id"] in selected]
    write_json(directory / "requests.json", requests)
    write_json(directory / "manifest.json", manifest)
    return {"inference_cases": len(requests), "gate_controls": len(manifest) - len(requests)}


def run_requests(requests, generate, *, output, budget=None, runtime=None):
    rows = []
    report = {"runtime": runtime or {}, "results": rows}
    write_json(output, report)
    for request in requests:
        limit = budget or request["max_tokens"]
        started = time.monotonic()
        row = {"id": request["id"], "messages_sha256": digest(request["messages"]),
               "max_tokens": limit, "sent": False, "saved": False}
        try:
            reply = generate(messages=request["messages"], tools=None, tool_choice="none", timeout_seconds=180,
                vllm_extra_body={"max_tokens": limit, "temperature": 0})
            row.update(content=reply.content, ok=bool(reply.ok and not reply.tool_calls), error=reply.error,
                       provider=reply.provider, model=reply.model, meta=reply.meta)
        except Exception as exc:
            row.update(content="", ok=False, error=f"{type(exc).__name__}: {exc}")
        row["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        rows.append(row)
        write_json(output, report)
        print(json.dumps({key: row[key] for key in ("id", "ok", "elapsed_ms")}), flush=True)
    return report


def score(manifest, report):
    from jsonschema import Draft202012Validator

    rows = report["results"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate result IDs")
    known = {case["id"] for case in manifest}
    if set(ids) - known:
        raise ValueError("Unexpected result IDs")
    by_id = {row["id"]: row for row in rows}
    scored = []
    for case in manifest:
        item = {"id": case["id"], "task": case["task"], "passed": False}
        row = by_id.get(case["id"])
        if case["task"] == "gate":
            item.update(passed=row is None, no_model_call=row is None)
            scored.append(item)
            continue
        errors = []
        item.update(schema_valid=False, label_matches=False)
        try:
            if row is None:
                raise ValueError("Missing result")
            if row.get("messages_sha256") != case["messages_sha256"]:
                raise ValueError("Input fingerprint mismatch")
            if not row.get("ok", True):
                raise ValueError(row.get("error") or "Inference failed")
            if row.get("finish_reason") == "length" or (row.get("meta") or {}).get("finish_reason") == "length":
                raise ValueError("Generation truncated")
            value = json.loads(row["content"])
            problems = list(Draft202012Validator(case["schema"]).iter_errors(value))
            if problems:
                raise ValueError("; ".join(problem.message for problem in problems))
            if case["task"] == "approval" and not math.isfinite(value["confidence"]):
                raise ValueError("Nonfinite confidence")
            item["schema_valid"] = True
            expected = case["expected"]
            if case["task"] == "approval":
                approved = value["label"] == "approve_save" and value["confidence"] >= 0.75
                wanted = expected["label"] == "approve_save"
                item.update(label_matches=value["label"] == expected["label"],
                    approval_at_threshold=approved, false_approval=approved and not wanted,
                    missed_approval=wanted and not approved,
                    within_live_timeout=row["elapsed_ms"] <= 6000,
                    approval_with_live_timeout=approved and row["elapsed_ms"] <= 6000)
                item["passed"] = item["label_matches"] and approved == wanted
            elif case["task"] == "note":
                item["label_matches"] = value["decision"] == expected["expected_decision"]
                text = value["note"].casefold()
                errors.extend("Missing: " + " / ".join(group) for group in expected.get("required_any", [])
                              if not any(word.casefold() in text for word in group))
                errors.extend("Unwanted: " + word for word in expected.get("forbidden", []) if word.casefold() in text)
                if bool(text.strip()) != (value["decision"] == "propose"):
                    errors.append("Note must be nonempty only for propose")
                item["passed"] = item["label_matches"] and not errors
            else:
                item["label_matches"] = all(value[key] == val for key, val in expected.items())
                item["passed"] = item["label_matches"]
            item["result"] = value
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
        item["errors"] = errors
        item["elapsed_ms"] = row.get("elapsed_ms") if row else None
        scored.append(item)
    groups = {}
    for task in sorted({case["task"] for case in manifest}):
        cases = [row for row in scored if row["task"] == task]
        timings = sorted(row["elapsed_ms"] for row in cases if row.get("elapsed_ms") is not None)
        groups[task] = {"total": len(cases), "passed": sum(row["passed"] for row in cases),
                       "schema_valid": sum(row.get("schema_valid", False) for row in cases),
                       "median_ms": statistics.median(timings) if timings else None,
                       "p95_ms": timings[math.ceil(len(timings) * .95) - 1] if timings else None}
    return {"groups": groups, "false_approvals": sum(row.get("false_approval", False) for row in scored),
            "results": scored, "limitations": "Synthetic diagnostic, one trial each. Fragment checks are incomplete proxies for note quality; inspect outputs. Timing differs by runtime/hardware/quantization. No live authorization or tools exercised."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("directory", type=Path)
    prep.add_argument("--note-prompt", choices=["baseline", "compact"], default="baseline")
    prep.add_argument("--data-dir", type=Path, default=DATA)
    prep.add_argument("--tasks", nargs="+", choices=["approval", "note", "grounding", "gate"])
    run = sub.add_parser("run")
    run.add_argument("requests", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--provider", choices=["ollama", "vllm"], required=True)
    run.add_argument("--max-output-tokens", type=int)
    run.add_argument("--think", choices=["true", "false", "low", "medium", "high"],
                     help="Explicit reasoning setting; omit for models without reasoning support")
    scoring = sub.add_parser("score")
    scoring.add_argument("manifest", type=Path)
    scoring.add_argument("results", type=Path)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        args.directory.mkdir(parents=True, exist_ok=False)
        print(json.dumps(prepare(args.directory, note_prompt=args.note_prompt, tasks=args.tasks, data_dir=args.data_dir)))
    elif args.command == "score":
        result = score(read_json(args.manifest), read_json(args.results))
        write_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    else:
        if args.output.exists():
            parser.error("Choose a new output path; preserve earlier trials")
        url = urlsplit(args.url)
        if url.scheme != "http" or url.hostname != "127.0.0.1" or url.username or url.password or url.query or url.fragment:
            parser.error("Only credential-free HTTP loopback inference URLs are allowed")
        if args.max_output_tokens is not None and args.max_output_tokens <= 0:
            parser.error("Output budget must be positive")
        os.environ.update(ISPEC_ASSISTANT_PROVIDER=args.provider, ISPEC_ASSISTANT_TEMPERATURE="0",
            ISPEC_INFERENCE_USAGE_LOG_ENABLED="0", ISPEC_OLLAMA_URL=args.url, ISPEC_OLLAMA_MODEL=args.model,
            ISPEC_OLLAMA_NUM_CTX="8192", ISPEC_OLLAMA_THINK=args.think or "", ISPEC_VLLM_URL=args.url,
            ISPEC_VLLM_MODEL=args.model, ISPEC_VLLM_STRICT_PROTOCOL="1")
        from ispec.assistant.service import generate_reply
        import requests
        original = requests.sessions.Session.request
        endpoint = url

        def loopback_only(session, method, url, **kwargs):
            target = urlsplit(url)
            if (target.scheme, target.hostname, target.port) != (endpoint.scheme, endpoint.hostname, endpoint.port):
                raise ValueError("Request escaped the selected inference endpoint")
            if target.path not in {"/api/chat", "/v1/models", "/v1/chat/completions"}:
                raise ValueError("Non-inference path rejected")
            kwargs["allow_redirects"] = False
            return original(session, method, url, **kwargs)

        requests.sessions.Session.request = loopback_only
        try:
            run_requests(read_json(args.requests), generate_reply, output=args.output, budget=args.max_output_tokens,
                runtime={"provider": args.provider, "model": args.model, "url": args.url, "think": args.think,
                         "constrained_json": False, "tool_execution": False, "temperature": 0, "context_limit": 8192})
        finally:
            requests.sessions.Session.request = original


if __name__ == "__main__":
    main()
