#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ispec.assistant.support_benchmark import (
    api_key,
    apply_loaded_env,
    assert_benchmark_baseline,
    assert_case_expectations,
    assistant_session_rows,
    benchmark_output_path,
    collect_turn_metrics,
    default_scenario_file,
    format_benchmark_summary,
    load_local_benchmark_scenarios,
    load_synthetic_benchmark_scenarios,
    load_workspace_env,
    summarize_benchmark_results,
    support_chat_url,
    wait_for_latest_assistant_message,
    workspace_root,
    _json_post,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run support-chat vLLM benchmarks against the local API.")
    parser.add_argument("--scenario-file", default=str(default_scenario_file()))
    parser.add_argument("--scenario-label", action="append", default=[], help="Optional synthetic scenario label filter.")
    parser.add_argument("--family", action="append", default=[], help="Optional scenario family filter.")
    parser.add_argument("--local-case", action="append", default=[], help="Optional local behavioral case selector or path.")
    parser.add_argument("--lookup-project-id", type=int, default=1596)
    parser.add_argument("--write-project-id", type=int, default=1531)
    parser.add_argument("--tmux-alias", default="ispec")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--run-label", default="baseline")
    parser.add_argument("--api-url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--assistant-wait-seconds", type=float, default=15.0)
    parser.add_argument("--allow-supervisor-running", action="store_true")
    parser.add_argument("--allow-non-vllm-provider", action="store_true")
    parser.add_argument("--skip-vllm-health-check", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    parser.add_argument("--output-file", help="Optional explicit JSON output path.")
    return parser


def _payload(session_id: str, message: str) -> dict[str, Any]:
    return {
        "sessionId": session_id,
        "message": message,
        "history": [],
        "ui": None,
        "meta": {
            "source": "benchmark_runner",
            "_queue_force_inline": True,
        },
    }


def _scenario_case(messages: list[Any]) -> dict[str, Any]:
    case_messages: list[dict[str, Any]] = []
    for row in messages:
        item = {
            "id": int(row.id),
            "role": str(row.role or ""),
            "content": str(row.content or ""),
        }
        if str(row.role or "") == "assistant":
            try:
                meta = json.loads(getattr(row, "meta_json", None) or "{}")
            except Exception:
                meta = {}
            if isinstance(meta, dict) and meta:
                item["assistant_meta"] = meta
        case_messages.append(item)
    return {"messages": case_messages}


def _run_turn(*, session_id: str, message: str, expect: dict[str, Any] | None, url: str, api_key_value: str | None, timeout_seconds: float, assistant_wait_seconds: float) -> dict[str, Any]:
    messages_before, _ = assistant_session_rows(session_id)
    last_assistant_id = max([int(row.id) for row in messages_before if str(row.role or "") == "assistant"], default=0)
    started = time.monotonic()
    response = _json_post(url=url, payload=_payload(session_id, message), api_key_value=api_key_value, timeout_seconds=timeout_seconds)
    wall_clock_ms = int((time.monotonic() - started) * 1000)
    assistant_row = wait_for_latest_assistant_message(session_id=session_id, min_message_id=last_assistant_id, timeout_seconds=assistant_wait_seconds)
    all_messages, _ = assistant_session_rows(session_id)
    case = _scenario_case(all_messages)
    ok = True
    expectation_error: str | None = None
    try:
        assert_case_expectations(case, expect)
    except AssertionError as exc:
        ok = False
        expectation_error = str(exc) or exc.__class__.__name__
    metrics = collect_turn_metrics(session_id=session_id, assistant_row=assistant_row, wall_clock_ms=wall_clock_ms)
    return {
        "message": message,
        "response_message": str(response.get("message") or ""),
        "http_response": response,
        "ok": ok,
        "expectation_error": expectation_error,
        **metrics,
    }


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    env = load_workspace_env(workspace_root())
    apply_loaded_env(env, root=workspace_root())
    baseline = assert_benchmark_baseline(
        env=env,
        allow_supervisor_running=bool(args.allow_supervisor_running),
        allow_non_vllm_provider=bool(args.allow_non_vllm_provider),
        check_vllm_health=not bool(args.skip_vllm_health_check),
    )
    render_vars = {
        "lookup_project_id": int(args.lookup_project_id),
        "write_project_id": int(args.write_project_id),
        "tmux_alias": str(args.tmux_alias),
    }
    labels = {str(item).strip() for item in args.scenario_label if str(item).strip()}
    families = {str(item).strip() for item in args.family if str(item).strip()}
    scenarios = load_synthetic_benchmark_scenarios(path=args.scenario_file, variables=render_vars, labels=labels or None, families=families or None)
    if args.local_case:
        scenarios.extend(load_local_benchmark_scenarios(case_selectors=list(args.local_case)))
    if not scenarios:
        raise SystemExit("No benchmark scenarios selected.")

    url = support_chat_url(env, override=args.api_url)
    api_key_value = api_key(env, override=args.api_key)
    run_id = uuid.uuid4().hex
    results: list[dict[str, Any]] = []
    repetitions = max(1, int(args.repeat))
    for scenario in scenarios:
        for repetition in range(1, repetitions + 1):
            session_id = f"benchmark:{args.run_label}:{scenario['label']}:{repetition}:{uuid.uuid4().hex[:8]}"
            turn_results: list[dict[str, Any]] = []
            for turn in scenario["turns"]:
                turn_results.append(_run_turn(
                    session_id=session_id,
                    message=str(turn["message"]),
                    expect=turn.get("expect"),
                    url=url,
                    api_key_value=api_key_value,
                    timeout_seconds=float(args.timeout_seconds),
                    assistant_wait_seconds=float(args.assistant_wait_seconds),
                ))
            scenario_case_messages, _ = assistant_session_rows(session_id)
            scenario_case = _scenario_case(scenario_case_messages)
            final_expect_error: str | None = None
            scenario_ok = all(bool(item.get("ok")) for item in turn_results)
            try:
                assert_case_expectations(scenario_case, scenario.get("final_expect"))
            except AssertionError as exc:
                scenario_ok = False
                final_expect_error = str(exc) or exc.__class__.__name__
            results.append({
                "label": scenario["label"],
                "family": scenario["family"],
                "tags": list(scenario.get("tags") or []),
                "notes": scenario.get("notes"),
                "source_kind": scenario.get("source_kind"),
                "source_path": scenario.get("source_path"),
                "repetition": repetition,
                "session_id": session_id,
                "ok": scenario_ok,
                "final_expectation_error": final_expect_error,
                "turns": turn_results,
            })

    summary = summarize_benchmark_results(results)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "run_label": str(args.run_label),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline,
        "scenario_file": str(Path(args.scenario_file).expanduser().resolve()),
        "local_case_selectors": list(args.local_case or []),
        "render_vars": render_vars,
        "summary": summary,
        "results": results,
    }
    output_path = Path(args.output_file).expanduser().resolve() if args.output_file else benchmark_output_path(run_label=args.run_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report["output_file"] = str(output_path)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run(argv)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_benchmark_summary(report["summary"]))
        print("")
        print(f"Output: {report['output_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
