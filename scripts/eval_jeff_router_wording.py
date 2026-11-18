"""Frozen offline wording sweep: one resident CPU model, no tool execution."""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time

from ispec.assistant.project_comment_workflow import COMMENT_WORKFLOW_TOOLS, comment_workflow_tools
from ispec.assistant.tools import _OPENAI_TOOL_SPECS
from scripts.eval_jeff_tool_routing import MISC_DESCRIPTION, DraftContext, routing_question, save_results
from scripts.jeff_router_catalog import SelectionBudget, ToolCatalog, select_categories, selection_tokens


PROJECTS = "Create/list/count/search projects; draft, revise, cancel, or save project notes/comments; project metadata."
NONE_VARIANTS = {
    "N0": "No tool category is needed to answer the latest request.",
    "N1": "No tool category is needed to help fulfill the latest request.",
    "N2": "The latest request can be answered from the conversation alone, without looking up or changing any data.",
    "N3": "No lookup, draft, save, or other action is needed; a direct reply is enough.",
    "N4": "Only greetings, thanks, or small talk.",
}
HISTORICAL_SAVE_KEY = "6c94fc0ff9749e43bc59b8ea1d6aa5f8a58d8ed95f9fc0d17b81161d313c0ef3"
BUDGET = SelectionBudget(cumulative_probability=0.9, max_categories=3, max_tools=24)


@dataclass(frozen=True, kw_only=True)
class EvaluationCase:
    id: str
    category: str
    state: str
    pending_draft: bool
    genuine_none: bool
    provenance: str


def digest(value) -> str:
    """Identify the complete model input, including choice order."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_cases(*, fixture: Path, historical: Path) -> list[EvaluationCase]:
    dataset = json.loads(fixture.read_text())
    cases = []
    for row in dataset["cases"]:
        parts = [f"User:\n{turn['user']}\n\nAssistant:\n{turn['assistant']}" for turn in row.get("history", [])]
        draft = DraftContext(**row["draft"]) if "draft" in row else None
        if draft:
            parts.append(f"Server draft state before latest reply:\n{draft.render()}")
        parts.append(f"Latest user request:\n{row['request']}")
        cases.append(EvaluationCase(
            id=row["id"], category=row["category"], state="\n\n".join(parts),
            pending_draft=bool(draft and draft.status == "pending"),
            genuine_none=row.get("genuine_none", False), provenance=dataset["provenance"],
        ))
    manifest = json.loads((historical / "manifest.json").read_text())
    saved = next(row for row in manifest["cases"] if row["key"] == HISTORICAL_SAVE_KEY)
    cases.append(EvaluationCase(
        id="HIST_SAVE57", category="projects", state=saved["state"], pending_draft=True,
        genuine_none=False, provenance="synthetic/derived with recorded model response; historical evaluation",
    ))
    return cases


def build_catalogs() -> dict[str, ToolCatalog]:
    """Build synthetic role fixtures from application schemas and existing tool scopes."""
    schemas = comment_workflow_tools(tools=list(_OPENAI_TOOL_SPECS.values()))
    return {
        role: ToolCatalog.from_authorized_schemas(schemas=schemas, staff=staff)
        for role, staff in (("staff", True), ("general_user", False))
    }


def variant_question(*, variant: str, state: str, catalog: ToolCatalog) -> dict:
    return routing_question(
        state=state, catalog=catalog,
        projects_description=None if variant == "baseline" else PROJECTS,
        none_description=NONE_VARIANTS["N0" if variant == "baseline" else variant],
    )


def pending_tools(*, case: EvaluationCase, catalog: ToolCatalog) -> frozenset[str]:
    return COMMENT_WORKFLOW_TOOLS & catalog.schemas.keys() if case.pending_draft else frozenset()


def historical_selection_audit(*, directory: Path) -> dict:
    """Compare the new limiter on old scores, retaining the old score population."""
    manifest = json.loads((directory / "manifest.json").read_text())
    report = {}
    for role, staff in (("staff", True), ("general_user", False)):
        catalog = ToolCatalog.from_authorized_schemas(schemas=manifest["authorized_schemas"], staff=staff)
        results = json.loads((directory / role / "results.json").read_text())
        included = available = top_real_correct = 0
        counts = []
        for case in manifest["cases"]:
            original = results[case["key"]]["scores"]
            top_real = max((name for name in original if name != "none"), key=original.get)
            top_real_correct += top_real == case["expected"]
            # Old inference offered unavailable categories. Preserve its former
            # eligible normalization only for this historical limiter comparison.
            eligible = {name: value for name, value in original.items() if name in catalog.groups or name == "none"}
            mass = sum(eligible.values())
            scores = {name: value / mass for name, value in eligible.items()}
            selected = select_categories(scores=scores, catalog=catalog, budget=BUDGET)
            available += case["expected"] in catalog.groups
            included += case["expected"] in selected.categories
            counts.append(len(selected.categories))
        report[role] = {
            "cases": len(counts), "top_non_none_matches_subject": top_real_correct,
            "available_category_cases": available, "included": included,
            "mean_categories": statistics.mean(counts),
        }
    report["limitation"] = "Saved scores normalized over the prior eligible catalog for limiter-only comparison. New inference filters before scoring. Historical subject labels are not action-need labels."
    return report


def prepare(*, args) -> None:
    """Freeze labels, every question, schema costs and source receipts before inference."""
    from tokenizers import Tokenizer

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = load_cases(fixture=args.fixture, historical=args.historical)
    catalogs = build_catalogs()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    manifest = {
        "purpose": "offline_evaluation_only_not_training",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_files": [
            {"name": str(path.relative_to(args.checkpoint)), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(args.checkpoint.rglob("*")) if path.is_file()
        ],
        "threads": 1, "budget": asdict(BUDGET), "cases": [asdict(case) for case in cases],
        "variants": ["baseline", *NONE_VARIANTS],
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "catalog_provenance": "Synthetic feature-complete staff catalog from application schemas; general user filtered by existing public/user ToolScope. No live account/config changes.",
        "catalogs": {role: {"schemas": catalog.schemas, "groups": {name: sorted(tools) for name, tools in catalog.groups.items()}} for role, catalog in catalogs.items()},
        "questions": {role: {variant: variant_question(variant=variant, state="", catalog=catalog)["question"] for variant in ["baseline", *NONE_VARIANTS]} for role, catalog in catalogs.items()},
        "tokenizer": str(args.tokenizer.resolve()),
        "selection_costs": {},
        "analysis_plan": "Compare balanced cases separately from historical save. Prefer tool-category coverage plus low none on actions and high none/no unnecessary categories on genuine no-tool cases. Report tradeoffs, unsupported requests separately, and fixed 0.9/k3/cap24 only. No tuning mid-run.",
    }
    for role, catalog in catalogs.items():
        manifest["selection_costs"][role] = {
            "directory_tokens": len(tokenizer.encode(json.dumps(catalog.directory_schema(), separators=(",", ":")), add_special_tokens=False).ids),
            "tool_counts": {name: len(tools) for name, tools in catalog.groups.items()},
        }
    for path in (Path(__file__), Path(__file__).with_name("jeff_router_catalog.py"), Path(__file__).with_name("eval_jeff_tool_routing.py"), args.fixture):
        (output / path.name).write_bytes(path.read_bytes())
    manifest["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "historical-selection.json").write_text(json.dumps(historical_selection_audit(directory=args.historical), indent=2) + "\n")
    print(f"Frozen {len(cases)} cases, six variants, two roles; {len(cases) * 12} combinations", flush=True)


def request_row(*, manifest: dict, role: str, variant: str, case: EvaluationCase) -> dict:
    return {"state": case.state, "question": manifest["questions"][role][variant]}


def prepare_schema_followup(*, source: Path, output: Path) -> None:
    """Freeze a staff-only single-description comparison with experiment controls."""
    prior = json.loads((source / "manifest.json").read_text())
    retained = {"M01", "M02", "M03", "E01", "E02", "E03", "P02", "Z03"}
    cases = [case for case in prior["cases"] if case["id"] in retained]
    for identifier, category, request in (
        ("API_FRESH", "misc", "Find the API endpoint and request parameters used to retrieve a person record."),
        ("RUN_FRESH", "experiments", "For experiment run 764, fetch the recorded label type and reference database."),
    ):
        cases.append(asdict(EvaluationCase(
            id=identifier, category=category, state=f"Latest user request:\n{request}",
            pending_draft=False, genuine_none=False, provenance="synthetic/hand-authored; fresh evaluation only",
        )))

    original = prior["questions"]["staff"]["N3"]
    explicit_schema = deepcopy(original)
    explicit_schema["criteria"]["misc"] = MISC_DESCRIPTION
    manifest = {
        **prior,
        "experiment": "N3 staff schema ownership, one description changed",
        "cases": cases,
        "variants": ["N3_original", "N3_schema"],
        "catalogs": {"staff": prior["catalogs"]["staff"]},
        "questions": {"staff": {"N3_original": original, "N3_schema": explicit_schema}},
        "selection_costs": {"staff": prior["selection_costs"]["staff"]},
        "source_evidence": str(source.resolve()),
        "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "analysis_plan": "M02 must include misc; schema variants should rank misc. Preserve experiments inclusion/top choice in E01-E03 and RUN_FRESH. Retain pending-save and wording-only controls. No mid-run changes, no general-user wording experiment.",
    }
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("eval_jeff_tool_routing.py"), Path(__file__).with_name("jeff_router_catalog.py")]
    for path in paths:
        (output / path.name).write_bytes(path.read_bytes())
    manifest["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    saved = json.loads((source / "results.json").read_text())
    matching = {}
    for row in cases:
        case = EvaluationCase(**row)
        request = request_row(manifest=manifest, role="staff", variant="N3_original", case=case)
        key = digest({"checkpoint": prior["checkpoint"], "row": request})
        if key in saved:
            matching[key] = {**saved[key], "reused_from": str(source.resolve())}
    save_results(output=output, results=matching)
    print(f"Frozen {len(cases)} staff cases, two arms; {len(matching)} exact scores reused, {2 * len(cases) - len(matching)} new predictions", flush=True)


def score_one(*, model, row: dict, checkpoint: str, results: dict, output: Path) -> bool:
    """Cache exact model inputs; return whether this call required new inference."""
    key = digest({"checkpoint": checkpoint, "row": row})
    if key in results:
        return False
    started = time.monotonic()
    probabilities, = model.predict(rows=[row], batch_size=1)
    scores = dict(zip(row["question"]["criteria"], map(float, probabilities), strict=True))
    results[key] = {"scores": scores, "seconds": time.monotonic() - started}
    save_results(output=output, results=results)
    return True


def summarize(*, output: Path) -> dict:
    """Report selection, none discrimination and costs without executing selected tools."""
    from tokenizers import Tokenizer

    manifest = json.loads((output / "manifest.json").read_text())
    results = json.loads((output / "results.json").read_text())
    tokenizer = Tokenizer.from_file(manifest["tokenizer"])
    cases = [EvaluationCase(**row) for row in manifest["cases"]]
    summary = {}
    for role, catalog_data in manifest["catalogs"].items():
        catalog = ToolCatalog(schemas=catalog_data["schemas"], groups={name: frozenset(tools) for name, tools in catalog_data["groups"].items()})
        token_cache = {}
        summary[role] = {}
        for variant in manifest["variants"]:
            rows = []
            for case in cases:
                row = request_row(manifest=manifest, role=role, variant=variant, case=case)
                result = results[digest({"checkpoint": manifest["checkpoint"], "row": row})]
                selection = select_categories(scores=result["scores"], catalog=catalog, budget=BUDGET, always_tool_names=pending_tools(case=case, catalog=catalog))
                if selection.tool_names not in token_cache:
                    token_cache[selection.tool_names] = selection_tokens(catalog=catalog, selection=selection, tokenizer=tokenizer)
                available = case.category in catalog.groups
                expected = case.category if available else "none"
                rows.append({
                    "case": case.id, "expected": expected, "subject": case.category,
                    "available": available, "genuine_none": case.genuine_none,
                    "none": result["scores"]["none"], "top": max(result["scores"], key=result["scores"].get),
                    "categories": selection.categories, "included": case.category in selection.categories,
                    "tools": selection.tool_count, "schema_tokens": token_cache[selection.tool_names],
                    "pending_tools": sorted(pending_tools(case=case, catalog=catalog)),
                    "seconds": result["seconds"],
                })
            balanced = [row for row in rows if row["case"] != "HIST_SAVE57"]
            tool_rows = [row for row in balanced if row["available"]]
            none_rows = [row for row in balanced if row["genuine_none"]]
            unsupported = [row for row in balanced if not row["available"] and not row["genuine_none"]]
            summary[role][variant] = {
                "cases": len(balanced), "top_correct": sum(row["top"] == row["expected"] for row in balanced),
                "tool_cases": len(tool_rows), "included": sum(row["included"] for row in tool_rows),
                "mean_none_on_tools": statistics.mean(row["none"] for row in tool_rows),
                "mean_none_on_genuine_none": statistics.mean(row["none"] for row in none_rows),
                "none_top_correct": sum(row["top"] == "none" for row in none_rows),
                "none_empty_selection": sum(not row["categories"] for row in none_rows),
                "unsupported_cases": len(unsupported), "unsupported_empty_selection": sum(not row["categories"] for row in unsupported),
                "mean_categories": statistics.mean(len(row["categories"]) for row in balanced),
                "mean_tools": statistics.mean(row["tools"] for row in balanced),
                "mean_schema_tokens": statistics.mean(row["schema_tokens"] for row in balanced),
                "median_seconds": statistics.median(row["seconds"] for row in rows), "rows": rows,
            }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(*, output: Path, historical: Path) -> None:
    import torch
    from jeff.models import load_decision_model

    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")
    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    # Old predictions are reusable only with exactly the same context/choices.
    old_manifest = json.loads((historical / "manifest.json").read_text())
    if old_manifest["checkpoint"] == manifest["checkpoint"]:
        for role in manifest["catalogs"]:
            old_results = json.loads((historical / role / "results.json").read_text())
            for case in old_manifest["cases"]:
                if case["key"] not in old_results:
                    continue
                row = {"state": case["state"], "question": old_manifest["passes"][role]["question"]["question"]}
                key = digest({"checkpoint": manifest["checkpoint"], "row": row})
                results.setdefault(key, {**old_results[case["key"]], "reused_from": str(historical / role)})
    save_results(output=output, results=results)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    model = load_decision_model(checkpoint=Path(manifest["checkpoint"]), device="cpu", cpu_threads=1)
    assert model.device_name == "cpu"
    print(f"Loaded once on CPU in {time.monotonic() - started:.1f}s", flush=True)
    cases = [EvaluationCase(**row) for row in manifest["cases"]]
    total = len(cases) * len(manifest["variants"]) * len(manifest["catalogs"])
    completed = inferred = 0
    for variant in manifest["variants"]:
        for role in manifest["catalogs"]:
            for case in cases:
                row = request_row(manifest=manifest, role=role, variant=variant, case=case)
                inferred += score_one(model=model, row=row, checkpoint=manifest["checkpoint"], results=results, output=output)
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total:
                    print(f"Completed {completed}/{total}; new inference={inferred}; {variant}/{role}/{case.id}", flush=True)
    summarize(output=output)
    print(f"Complete; {inferred} new predictions. All tools remained unexecuted.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "prepare-schema", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("tests/behavioral/local/router-repair-20261113/jeff-wording-sweep"))
    parser.add_argument("--fixture", type=Path, default=Path("tests/behavioral/data/jeff_router_wording_20261113.json"))
    parser.add_argument("--historical", type=Path, default=Path("tests/behavioral/local/router-repair-20261113/jeff-contextual-dual-catalog-resumed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/e/tools/jeff/jeff-base-v1.3"))
    parser.add_argument("--tokenizer", type=Path, default=Path("/mnt/e/ispec-model-cache/olmo3-instruct-6e5971d9/tokenizer.json"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args=args)
    if args.action == "prepare-schema":
        prepare_schema_followup(source=args.source.resolve(), output=args.output.resolve())
    if args.action == "run":
        run(output=args.output.resolve(), historical=args.historical.resolve())
    if args.action == "summarize":
        summarize(output=args.output.resolve())


if __name__ == "__main__":
    main()
