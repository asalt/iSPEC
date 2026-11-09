"""Replay synthetic router inputs only; no planner, tool execution or live DBs."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch
from urllib.parse import urlsplit

import jsonschema
import requests

from ispec.assistant.service import generate_reply
from ispec.assistant.tool_routing import (
    route_tool_groups_vllm,
    tool_groups_for_available_tools,
)
from scripts.run_instruct_refinement import require_ready_model, trial_environment
from scripts.summarize_prompt_composition_trials import read_records


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "mixed_then_corrected": "projects",
    "old_approval_is_context": "projects",
    "failed_lookup_continuity": "projects",
    "greeting_no_action": "assistant",
}


def prepare_cases(source):
    """Reuse exact first-turn router messages, plus two frozen category controls."""
    records = {
        row["case"]: row
        for row in read_records(source)
        if row["turn"] == 0 and row["case"] in EXPECTED
    }
    if set(records) != set(EXPECTED):
        raise ValueError("Choose the completed synthetic composition baseline log")
    cases = []
    for name, expected in EXPECTED.items():
        row = records[name]
        request = row["wire"][0]["request"]
        if request.get("max_tokens") != 200 or request.get("tools"):
            raise ValueError("First saved call is not the expected tool-free router")
        available = sorted(
            {
                tool
                for group in row["metadata"]["tool_router"]["groups"]
                for tool in group["tools"]
            }
        )
        cases.append(
            {
                "id": name,
                "expected_primary": expected,
                "user_message": row["request"],
                "available_tools": available,
                "context_messages": request["messages"][1:-1],
                "expected_messages": request["messages"],
                "expected_schema": request["structured_outputs"]["json"],
                "source": "saved_synthetic_router_request",
            }
        )
    for name, message, expected in (
        ("fresh_people", "Find the person record for Mira Example.", "people"),
        (
            "fresh_runs",
            "List the experiment runs belonging to project 2471.",
            "experiments",
        ),
    ):
        cases.append(
            {
                "id": name,
                "expected_primary": expected,
                "user_message": message,
                "available_tools": cases[0]["available_tools"],
                "context_messages": [],
                "source": "fresh_synthetic_category_control",
            }
        )
    return cases


def evaluate_case(case, budget, generate_fn):
    if budget not in (512, 1024):
        raise ValueError("This pilot compares only 512 and 1024 output tokens")
    request = {}

    def budgeted(**kwargs):
        assert kwargs["tools"] is None
        assert kwargs["vllm_extra_body"]["max_tokens"] == 200
        if "expected_messages" in case:
            assert (
                kwargs["messages"] == case["expected_messages"]
            ), "Router prompt drift"
            assert (
                kwargs["vllm_extra_body"]["structured_outputs"]["json"]
                == case["expected_schema"]
            )
        kwargs = deepcopy(kwargs)
        kwargs["vllm_extra_body"]["max_tokens"] = budget
        request.update(deepcopy(kwargs))
        return generate_fn(**kwargs)

    start = time.monotonic()
    decision, reply = route_tool_groups_vllm(
        user_message=case["user_message"],
        context_messages=tuple(deepcopy(case["context_messages"])),
        groups=tool_groups_for_available_tools(set(case["available_tools"])),
        generate_reply_fn=budgeted,
    )
    elapsed_ms = round((time.monotonic() - start) * 1000)
    schema = request["vllm_extra_body"]["structured_outputs"]["json"]
    raw_json_valid = False
    try:

        def reject_constant(value):
            raise ValueError("Non-finite JSON number")

        parsed = json.loads(reply.content, parse_constant=reject_constant)
        jsonschema.validate(parsed, schema)
        raw_json_valid = True
    except (ValueError, jsonschema.ValidationError):
        pass
    meta = reply.meta or {}
    valid = bool(
        reply.ok and raw_json_valid and decision is not None and not reply.tool_calls
    )
    matched = bool(valid and decision["primary"] == case["expected_primary"])
    return {
        "case": case["id"],
        "budget": budget,
        "expected_primary": case["expected_primary"],
        "decision": decision,
        "provider_ok": reply.ok,
        "raw_json_valid": raw_json_valid,
        "usable": valid,
        "primary_correct": matched,
        "truncated": meta.get("finish_reason") == "length",
        "elapsed_ms": elapsed_ms,
        "content": reply.content,
        "error": reply.error,
        "provider_meta": meta,
        "request": request,
    }


@contextmanager
def inference_only(url, wire):
    target = urlsplit(url)
    original = requests.Session.request

    def guarded(session, method, url, **kwargs):
        actual = urlsplit(url)
        assert (actual.scheme, actual.hostname, actual.port) == (
            target.scheme,
            target.hostname,
            target.port,
        )
        assert (method.upper(), actual.path) in {
            ("GET", "/health"),
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
        }
        assert not kwargs.get("auth")
        assert not any(
            key.lower() == "authorization" for key in kwargs.get("headers", {})
        )
        session.trust_env = False
        kwargs["allow_redirects"] = False
        if method.upper() != "POST":
            return original(session, method, url, **kwargs)
        body = deepcopy(kwargs["json"])
        assert not body.get("tools")
        item = {"request": body}
        wire.append(item)
        response = original(session, method, url, **kwargs)
        item.update(status=response.status_code, response=response.json())
        return response

    with patch.object(requests.Session, "request", guarded):
        yield


def worker(cases, output, url, model):
    wire, records = [], []
    with inference_only(url, wire):
        require_ready_model(url, model)
        for index, case in enumerate(cases):
            # Alternate order so one budget does not always receive the warm prefix.
            for budget in (512, 1024) if index % 2 == 0 else (1024, 512):
                first_wire = len(wire)
                result = evaluate_case(
                    case,
                    budget,
                    lambda **kwargs: generate_reply(**kwargs, timeout_seconds=90),
                )
                result["wire"] = wire[first_wire:]
                records.append(result)
                with (output / "results.jsonl").open("a") as handle:
                    handle.write(json.dumps(result) + "\n")
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "case",
                                "budget",
                                "usable",
                                "primary_correct",
                                "truncated",
                                "elapsed_ms",
                            )
                        }
                    ),
                    flush=True,
                )
    summary = []
    for budget in (512, 1024):
        rows = [row for row in records if row["budget"] == budget]
        summary.append(
            {
                "budget": budget,
                "cases": len(rows),
                "usable": sum(row["usable"] for row in rows),
                "primary_correct": sum(row["primary_correct"] for row in rows),
                "truncated": sum(row["truncated"] for row in rows),
                "median_ms": statistics.median(row["elapsed_ms"] for row in rows),
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0  # A completed evaluation is not a claim that the model passed.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18030")
    parser.add_argument("--model", default="olmo-think-router-eval")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    env = trial_environment(
        output,
        url=args.url,
        model=args.model,
        trial="baseline",
        split="dev",
        reasoning_parser="olmo3",
        max_output_tokens=1024,
    )
    if args.worker:
        return worker(
            json.loads((output / "cases.json").read_text()),
            output,
            args.url,
            args.model,
        )
    if args.source is None:
        parser.error("--source must name the saved synthetic baseline log")
    cases = prepare_cases(args.source)
    output.mkdir(parents=True, exist_ok=False)
    (output / "home").mkdir()
    case_text = json.dumps(cases, indent=2) + "\n"
    (output / "cases.json").write_text(case_text)
    sources = [
        Path(__file__),
        ROOT / "src/ispec/assistant/tool_routing.py",
        ROOT / "src/ispec/assistant/service.py",
        ROOT / "src/ispec/prompts/assistant.tool_router.classifier.md",
    ]
    receipt = {
        "production": False,
        "started_at": time.time(),
        "source_log": str(args.source.resolve()),
        "source_log_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(case_text.encode()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "model": args.model,
        "url": args.url,
        "budgets": [512, 1024],
    }
    (output / "run.json").write_text(json.dumps(receipt, indent=2))
    command = [
        sys.executable,
        str(Path(__file__)),
        "--worker",
        "--output-dir",
        str(output),
        "--url",
        args.url,
        "--model",
        args.model,
    ]
    with (output / "worker.log").open("x") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1500,
        )
    receipt.update(ended_at=time.time(), exit_code=result.returncode)
    (output / "run.json").write_text(json.dumps(receipt, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
