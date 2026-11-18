"""Offline CPU category scoring of saved synthetic router inputs; no tool execution."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import statistics
import time

from ispec.assistant.tool_routing import _GROUP_DEFS
from scripts.jeff_router_catalog import SelectionBudget, ToolCatalog, select_categories, selection_tokens


MISC_DESCRIPTION = (
    "Inspect API schemas and documentation; find endpoints, operations, accepted "
    "parameters and request/response fields for any record type; DB file stats and recent activity."
)


@dataclass(frozen=True, kw_only=True)
class DraftContext:
    draft_id: str
    text: str
    status: str
    project_id: int | None
    comment_id: int | None

    def render(self) -> str:
        identity = f"Draft {self.draft_id}; status={self.status}"
        if self.project_id is not None:
            identity += f"; project={self.project_id}"
        if self.comment_id is not None:
            identity += f"; comment={self.comment_id}"
        return f"{identity}\n{self.text}"


@dataclass(frozen=True, kw_only=True)
class RecordedTurn:
    user: str
    assistant: str | None
    drafts: tuple[DraftContext, ...]


def recorded_turns(*, conversation: dict) -> tuple[RecordedTurn, ...]:
    """Read delivered text and end-of-turn draft snapshots from saved evidence."""
    turns = []
    cards_by_turn = conversation.get("cards_by_turn", [])
    for index, turn in enumerate(conversation["turns"]):
        cards = cards_by_turn[index] if index < len(cards_by_turn) else []
        project_ids = {card["draft_id"]: card["project_id"] for card in cards}
        drafts = tuple(
            DraftContext(
                draft_id=draft["id"], text=draft["text"], status=draft["status"],
                project_id=project_ids.get(draft["id"]), comment_id=draft.get("comment_id"),
            )
            for draft in turn.get("drafts", [])
        )
        turns.append(RecordedTurn(user=turn["user"], assistant=turn["assistant"], drafts=drafts))
    requests = conversation.get("case_definition", {}).get("messages", [])
    if len(requests) > len(turns):
        # A failed inference can leave the current request in wire.json without a completed turn.
        turns.append(RecordedTurn(user=requests[len(turns)], assistant=None, drafts=()))
    return tuple(turns)


def conversation_context(
    *, messages: list[dict], turns: tuple[RecordedTurn, ...], turn_index: int,
) -> str:
    """Render the latest request with two preceding exchanges and prior draft state."""
    parts = []
    if messages[1:-1]:
        quoted = "\n\n".join(
            f"{message['role']}: {message['content']}" for message in messages[1:-1]
        )
        parts.append(f"Quoted source context:\n{quoted}")
    if turn_index > 2:
        parts.append(f"Original user request:\n{turns[0].user}")
    for turn in turns[max(0, turn_index - 2):turn_index]:
        if turn.assistant is None:
            raise ValueError("A preceding exchange is missing its recorded assistant response")
        parts.append(f"User:\n{turn.user}\n\nAssistant:\n{turn.assistant}")
    if turn_index > 0:
        drafts = turns[turn_index - 1].drafts
        draft_text = "\n\n".join(draft.render() for draft in drafts) or "No registered drafts."
        parts.append(f"Server draft state before latest reply:\n{draft_text}")
    parts.append(f"Latest user request:\n{turns[turn_index].user}")
    return "\n\n".join(parts)


@dataclass(frozen=True, kw_only=True)
class RouterInput:
    key: str
    case: str
    state: str
    expected: str | None
    messages: list[dict]
    sources: list[dict]


def expected_category(*, case: str, latest: str) -> str | None:
    """Hand labels from the saved synthetic case definitions, frozen before scoring."""
    controls = {
        "C01_project": "projects", "C02_person": "people",
        "C03_runs": "experiments", "C04_greeting": None, "C05_missing_id": "projects",
    }
    if case in controls:
        return controls[case]
    if case == "N17_unrelated_question" and "which day is the staff meeting" in latest:
        return "schedule"
    if case.startswith(("P", "N", "R")):
        return "projects"
    raise ValueError(f"No reviewed expected category for {case}")


def collect_inputs(*, roots: list[Path]) -> list[RouterInput]:
    """Pair router calls with prior conversation state, retaining source receipts."""
    found = {}
    for root in roots:
        for wire in sorted(root.glob("*/wire.json")):
            conversation = wire.with_name("conversation.json")
            if not conversation.is_file():
                continue
            evidence = json.loads(conversation.read_text())
            case = evidence["case"]
            turns = recorded_turns(conversation=evidence)
            next_turn = 0
            for index, call in enumerate(json.loads(wire.read_text())["calls"]):
                request = call["request"]
                schema = request.get("structured_outputs", {}).get("json", {})
                if "primary" not in schema.get("properties", {}):
                    continue
                messages = request["messages"]
                turn_index = next((
                    position for position in range(next_turn, len(turns))
                    if messages[-1]["content"] == f'User request: "{turns[position].user}"'
                ), None)
                if turn_index is None:
                    raise ValueError(f"Cannot align router call {index} with a recorded turn in {wire}")
                next_turn = turn_index + 1
                state = conversation_context(messages=messages, turns=turns, turn_index=turn_index)
                key = hashlib.sha256(state.encode()).hexdigest()
                expected = expected_category(case=case, latest=turns[turn_index].user)
                choice = call["response"]["choices"][0]
                try:
                    decision = json.loads(choice["message"]["content"])
                except (ValueError, TypeError):
                    decision = None
                source = {
                    "wire": str(wire), "call_index": index, "turn_index": turn_index,
                    "finish_reason": choice["finish_reason"], "decision": decision,
                    "latency_ms": call["elapsed_ms"],
                }
                if key in found:
                    if found[key].expected != expected:
                        raise ValueError(f"Conflicting expected labels for the same input in {wire}")
                    found[key].sources.append(source)
                    continue
                found[key] = RouterInput(
                    key=key, case=case, state=state, messages=messages, sources=[source],
                    expected=expected,
                )
    return list(found.values())


def routing_question(
    *, state: str, catalog: ToolCatalog,
    projects_description: str | None = None,
    none_description: str = "No tool category is needed to answer the latest request.",
) -> dict:
    """Rank category alternatives and none in one context-bearing choice question."""
    criteria = {
        group.name: group.description for group in _GROUP_DEFS
        if group.name in catalog.groups
    }
    if "misc" in criteria:
        criteria["misc"] = MISC_DESCRIPTION
    if projects_description is not None and "projects" in criteria:
        criteria["projects"] = projects_description
    criteria["none"] = none_description
    return {
        "state": state,
        "question": {
            "type": "choice",
            "criteria": criteria,
            "instructions": "Which tool category best fits the latest user request in this conversation, or none?",
        },
    }


def score_categories(*, model, state: str, catalog: ToolCatalog) -> dict[str, float]:
    """Read one distribution over category alternatives from one model input."""
    row = routing_question(state=state, catalog=catalog)
    probabilities, = model.predict(rows=[row], batch_size=1)
    return dict(zip(row["question"]["criteria"], probabilities, strict=True))


def threshold_curve(
    *, cases: list[RouterInput], results: dict, catalog: ToolCatalog, tokenizer,
) -> list[dict]:
    """Measure coverage and schema costs over cumulative mass, category and tool caps."""
    curve = []
    token_counts = {}
    for target, max_categories, max_tools in product((0.7, 0.8, 0.9, 0.95), (1, 3, 5), (12, 24)):
        budget = SelectionBudget(
            cumulative_probability=target, max_categories=max_categories, max_tools=max_tools,
        )
        covered = needed = recoverable = no_tool_correct = 0
        tools_per_case, tokens_per_case, selections = [], [], []
        for case in cases:
            selection = select_categories(scores=results[case.key]["scores"], catalog=catalog, budget=budget)
            expected_included = case.expected in selection.categories
            directory_needed = case.expected is not None and not expected_included
            covered += expected_included
            needed += directory_needed
            recoverable += directory_needed and case.expected in catalog.groups
            no_tool_correct += case.expected is None and not selection.categories
            if selection.tool_names not in token_counts:
                token_counts[selection.tool_names] = selection_tokens(
                    catalog=catalog, selection=selection, tokenizer=tokenizer,
                )
            tokens = token_counts[selection.tool_names]
            tools_per_case.append(selection.tool_count)
            tokens_per_case.append(tokens)
            selections.append({
                "key": case.key, "case": case.case, "expected": case.expected,
                "categories": selection.categories, "probability_mass": selection.probability_mass,
                "tool_count": selection.tool_count, "schema_json_tokens": tokens,
                "directory_needed": directory_needed, "stop_reason": selection.stop_reason,
            })
        curve.append({
            **asdict(budget), "cases": len(cases), "category_cases": sum(case.expected is not None for case in cases),
            "eligible_category_cases": sum(case.expected in catalog.groups for case in cases),
            "unavailable_category_cases": sum(case.expected is not None and case.expected not in catalog.groups for case in cases),
            "expected_included": covered, "directory_needed": needed, "directory_recoverable": recoverable,
            "no_tool_correct": no_tool_correct, "mean_tool_count": statistics.mean(tools_per_case),
            "max_tool_count": max(tools_per_case), "mean_schema_json_tokens": statistics.mean(tokens_per_case),
            "max_schema_json_tokens": max(tokens_per_case), "selections": selections,
            "top_category_over_tool_cap": sum(count > max_tools for count in tools_per_case),
        })
    return curve


def save_results(*, output: Path, results: dict) -> None:
    """Atomically replace the score checkpoint after a completed prediction."""
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(output / "results.json")


def record_prediction(*, model, case: RouterInput, threads: int, results: dict, output: Path, catalog: ToolCatalog) -> float:
    """Persist each completed prediction so interruption retains measured evidence."""
    started = time.monotonic()
    scores = score_categories(model=model, state=case.state, catalog=catalog)
    elapsed = time.monotonic() - started
    results[case.key] = {"scores": scores, "seconds": elapsed, "threads": threads}
    save_results(output=output, results=results)
    return elapsed


def reuse_staff_results(*, directory: Path | None, cases: list[RouterInput], checkpoint: Path, catalog: ToolCatalog) -> dict:
    """Reuse matching completed staff predictions from the interrupted single-choice run."""
    if directory is None:
        return {}
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["checkpoint"] != str(checkpoint.resolve()):
        raise ValueError("Reuse checkpoint differs")
    if manifest["question"]["question"] != routing_question(state="", catalog=catalog)["question"]:
        raise ValueError("Reuse question differs")
    saved_cases = {case["key"]: case["state"] for case in manifest["cases"]}
    saved_results = json.loads((directory / "results.json").read_text())
    reused = {}
    for case in cases:
        if saved_cases.get(case.key) == case.state and case.key in saved_results:
            reused[case.key] = {**saved_results[case.key], "reused_from": str(directory.resolve())}
    return reused


def resume_results(
    *, directory: Path | None, pass_name: str, staff: bool, cases: list[RouterInput],
    checkpoint: Path, catalog: ToolCatalog,
) -> dict:
    """Recover matching completed inputs from a stopped two-pass evaluation."""
    if directory is None or not (directory / pass_name / "results.json").is_file():
        return {}
    manifest = json.loads((directory / "manifest.json").read_text())
    prior_pass = manifest["passes"][pass_name]
    expected_groups = {name: sorted(tools) for name, tools in catalog.groups.items()}
    if manifest["checkpoint"] != str(checkpoint.resolve()):
        raise ValueError("Resume checkpoint differs")
    if prior_pass["question"]["question"] != routing_question(state="", catalog=catalog)["question"]:
        raise ValueError("Resume question differs")
    if prior_pass["directory_groups"] != expected_groups:
        raise ValueError("Resume catalog differs")
    saved_cases = {case["key"]: case["state"] for case in manifest["cases"]}
    saved = json.loads((directory / pass_name / "results.json").read_text())
    return {
        case.key: {**saved[case.key], "resumed_from": str(directory.resolve())}
        for case in cases if saved_cases.get(case.key) == case.state and case.key in saved
    }


def run(args) -> None:
    cases = collect_inputs(roots=[path.resolve() for path in args.inputs])
    if not cases:
        raise ValueError("No saved router calls found")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    captured = json.loads(args.catalog_wire.read_text())
    schemas = max((call["request"].get("tools") or [] for call in captured["calls"]), key=len)
    if not schemas:
        raise ValueError("The catalog wire has no captured tool schemas")
    # This replay uses the recorded synthetic staff catalog, not a live account.
    scenarios = {"staff": True, "general_user": False}
    catalogs = {
        name: ToolCatalog.from_authorized_schemas(schemas=schemas, staff=staff)
        for name, staff in scenarios.items()
    }
    manifest = {
        "source_kind": "synthetic_cases_with_recorded_model_outputs",
        "purpose": "offline_evaluation_not_training",
        "experiment": "contextual_single_choice_dual_catalog_v2",
        "expected_label_scope": "category for the latest request with preceding exchanges and prior draft state",
        "baseline_comparison": "Historical OLMo inputs lack the added conversation context; this is not a matched-input comparison.",
        "score_semantics": "one normalized distribution over alternative categories and none, not independent relevance",
        "checkpoint": str(args.checkpoint.resolve()),
        "cases": [asdict(case) for case in cases],
        "catalog_source": str(args.catalog_wire.resolve()),
        "catalog_source_sha256": hashlib.sha256(args.catalog_wire.read_bytes()).hexdigest(),
        "authorized_schemas": schemas,
        "passes": {
            name: {
                "question": routing_question(state="<preceding exchange, prior draft state and latest request>", catalog=catalogs[name]),
                "directory": catalogs[name].directory_schema(),
                "directory_groups": {group: sorted(tools) for group, tools in catalogs[name].groups.items()},
            }
            for name, staff in scenarios.items()
        },
        "selection_policy": "Filter before scoring. Accumulate all choices including none at full weight; retain the top real category. Max k and tool cap limit additions.",
        "general_user_policy": "Hide staff-only categories and tools using existing ToolScope public/user rules; execution still requires actual account and project ACL checks.",
        "token_measurement": "Compact schema JSON with the OLMo tokenizer, not a full rendered conversation",
        "tokenizer": str(args.tokenizer.resolve()),
        "threads": 1,
        "interop_threads": 1,
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "catalog_selector_sha256": hashlib.sha256(Path(__file__).with_name("jeff_router_catalog.py").read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(cases)} contextual inputs; one question with {len(_GROUP_DEFS)} categories + none", flush=True)
    if args.prepare_only:
        return

    import torch
    from jeff.models import load_decision_model
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    model = load_decision_model(checkpoint=args.checkpoint.resolve(), device="cpu", cpu_threads=1)
    assert model.device_name == "cpu"
    print(f"Loaded once on CPU in {time.monotonic() - started:.1f}s", flush=True)
    summaries = {}
    for name, staff in scenarios.items():
        pass_output = output / name
        pass_output.mkdir()
        catalog = catalogs[name]
        results = {}
        if staff:
            results = reuse_staff_results(directory=args.reuse_staff_results, cases=cases, checkpoint=args.checkpoint, catalog=catalog)
        results.update(resume_results(
            directory=args.resume_from, pass_name=name, staff=staff, cases=cases,
            checkpoint=args.checkpoint, catalog=catalog,
        ))
        reused_count = len(results)
        save_results(output=pass_output, results=results)
        print(f"{name}: {reused_count} matching predictions reused; one CPU thread, no timing probes", flush=True)
        for index, case in enumerate(cases):
            if case.key not in results:
                record_prediction(model=model, case=case, threads=1, results=results, output=pass_output, catalog=catalog)
            if index % 10 == 0:
                print(f"{name}: scored {index + 1}/{len(cases)}", flush=True)
        save_results(output=pass_output, results=results)
        top_correct = sum(
            max(results[case.key]["scores"], key=results[case.key]["scores"].get) == (case.expected or "none")
            for case in cases
        )
        summary = {
            "cases": len(cases), "top_correct": top_correct, "reused_count": reused_count,
            "median_seconds": statistics.median(row["seconds"] for row in results.values()),
            "total_prediction_seconds": sum(row["seconds"] for row in results.values()),
            "curve": threshold_curve(cases=cases, results=results, catalog=catalog, tokenizer=tokenizer),
        }
        (pass_output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries[name] = summary
        print(json.dumps({"pass": name, **{key: value for key, value in summary.items() if key != "curve"}}), flush=True)
    originals = [source for case in cases for source in case.sources]
    baseline_correct = sum(
        source["finish_reason"] == "stop" and isinstance(source["decision"], dict)
        and source["decision"].get("primary") == (case.expected or "none")
        for case in cases for source in case.sources
    )
    summary = {
        "cases": len(cases), "passes": summaries,
        "olmo_calls": len(originals), "olmo_valid_correct_primary": baseline_correct,
        "olmo_incomplete": sum(source["finish_reason"] != "stop" for source in originals),
        "baseline_comparison": manifest["baseline_comparison"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed both passes; summaries and curves in {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-wire", type=Path, required=True, help="Recorded synthetic staff request containing the available schemas")
    parser.add_argument("--tokenizer", type=Path, default=Path("/mnt/e/ispec-model-cache/olmo3-instruct-6e5971d9/tokenizer.json"))
    parser.add_argument("--reuse-staff-results", type=Path, help="Reuse matching predictions from the interrupted single-choice run")
    parser.add_argument("--resume-from", type=Path, help="Reuse matching completed staff/general-user inputs from a stopped two-pass run")
    parser.add_argument("--prepare-only", action="store_true", help="Write contextual inputs without loading a model")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
