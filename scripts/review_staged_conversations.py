"""Render saved synthetic conversations for offline review. No inference or writes to iSPEC."""

import argparse
from collections import Counter
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from string import Template
from xml.etree import ElementTree

try:
    from scripts.staged_prompt_trace import InferenceRecord, TemplateProfile, WireCapture, WireMessage, load_profiles, render_prompt
except ModuleNotFoundError:  # Direct execution from scripts/ as well as python -m.
    from staged_prompt_trace import InferenceRecord, TemplateProfile, WireCapture, WireMessage, load_profiles, render_prompt


ASSETS = Path(__file__).with_name("staged_review_assets")
STATUSES = ("fail", "error", "pass", "skipped", "unknown")


@dataclass(frozen=True, kw_only=True)
class TestResult:
    status: str
    detail: str
    source: Path | None = None


@dataclass(frozen=True, kw_only=True)
class CaseReview:
    anchor: str
    batch: str
    path: Path
    payload: dict
    test: TestResult
    actual: tuple[int, ...]
    signals: tuple[str, ...]
    findings: tuple[str, ...]
    finding_source: str

    @property
    def needs_review(self) -> bool:
        return bool(self.signals or self.findings)


def block(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    return "<pre>" + escape(text) + "</pre>"


def details(*, title: str, body: str) -> str:
    return "<details><summary>" + escape(title) + "</summary>" + body + "</details>"


def badge(*, label: str, kind: str) -> str:
    return f'<span class="badge {escape(kind)}">{escape(label)}</span>'


def saved_cases(*, roots: list[Path]):
    seen = set()
    for root in roots:
        for candidate in sorted(root.rglob("conversation.json")):
            path = candidate.resolve()
            if path not in seen:
                seen.add(path)
                yield root.name, path, json.loads(path.read_text())


def recorded_result(*, path: Path, case_id: str, roots: list[Path]) -> TestResult:
    boundary = max((root.resolve() for root in roots if path.is_relative_to(root.resolve())),
                   key=lambda root: len(root.parts))
    for parent in path.parents:
        source = parent / "results.xml"
        if source.exists():
            try:
                names = {f"{function}[{case_id}]" for function in
                         ("test_staged_conversation", "test_promotion_conversation")}
                matches = [test for test in ElementTree.parse(source).iter("testcase")
                           if test.get("name") in names]
            except ElementTree.ParseError:
                return TestResult(status="unknown", detail="Unreadable JUnit XML.", source=source)
            if len(matches) != 1:
                return TestResult(status="unknown", detail="No unique matching JUnit test.", source=source)
            for tag, status in (("error", "error"), ("failure", "fail"), ("skipped", "skipped")):
                finding = matches[0].find(tag)
                if finding is not None:
                    return TestResult(status=status, detail=finding.text or finding.get("message", ""), source=source)
            return TestResult(status="pass", detail="Recorded pytest assertions passed.", source=source)
        if parent == boundary:
            break
    return TestResult(status="unknown", detail="No JUnit result found; counts alone do not establish a pass.")


def review_signals(*, case: dict) -> tuple[str, ...]:
    expected = case["expected_comment_counts"]
    actual = [len(turn["comments"]) for turn in case["turns"]]
    save_turns = {i for i, count in enumerate(expected) if count > (expected[i - 1] if i else 0)}
    saves = [call for call in case["tool_calls"] if call["name"] == "create_project_comment"]
    signals = []
    if any(count > (actual[i - 1] if i else 0) and i not in save_turns for i, count in enumerate(actual)):
        signals.append("Write on a non-save turn")
    if any(call["turn"] not in save_turns for call in saves):
        signals.append("Save attempted on a non-save turn")
    if save_turns - {call["turn"] for call in saves}:
        signals.append("Requested save not called")
    for call in saves:
        result = call.get("result") or {}
        if result.get("ok") is False:
            signals.append("Save rejected: " + str(result.get("error_code") or result.get("error") or "unspecified"))
    if any((turn["metadata"].get("write_claim_guard") or {}).get("triggered") for turn in case["turns"]):
        signals.append("Response guard intervened")
    return tuple(dict.fromkeys(signals))


def load_reviews(*, roots: list[Path], notes: Path | None) -> list[CaseReview]:
    annotations = {}
    for entry in json.loads(notes.read_text()) if notes else []:
        key = (entry["batch"], entry["model"], entry["case"])
        if key in annotations:
            raise ValueError(f"Duplicate review annotation: {key}")
        annotations[key] = entry
    reviews = []
    for index, (batch, path, case) in enumerate(saved_cases(roots=roots)):
        annotation = annotations.get((batch, case["model"], case["case"]), {})
        reviews.append(CaseReview(
            anchor=f"case-{index}", batch=batch, path=path, payload=case,
            test=recorded_result(path=path, case_id=case["case"], roots=roots),
            actual=tuple(len(turn["comments"]) for turn in case["turns"]),
            signals=review_signals(case=case), findings=tuple(annotation.get("findings", [])),
            finding_source=annotation.get("source", ""),
        ))
    if not reviews:
        raise ValueError("No conversation.json evidence found")
    order = {root.name: index for index, root in enumerate(roots)}
    return sorted(reviews, key=lambda row: (-order[row.batch], STATUSES.index(row.test.status),
                                          row.payload["case"], row.payload["model"]))


def message_panel(*, message: WireMessage, number: int) -> str:
    content = ['<article class="message-card">',
               f'<h5>Message {number} <span class="role-label">role={escape(message.role)}</span></h5>']
    if message.tool_call_id:
        content.append(f'<p class="caption">Response to call: {escape(message.tool_call_id)}</p>')
    content.append(block(message.content) if message.content is not None else '<p class="caption">No text content.</p>')
    if message.reasoning:
        content += ['<h5>Reasoning returned by API</h5>', block(message.reasoning)]
    for call in message.tool_calls:
        content += [f'<h5>Parsed tool call: {escape(call.function.name)}</h5>', block(call.function.arguments)]
    return "".join(content) + '</article>'


def inference_panel(*, call: InferenceRecord, number: int, anchor: str,
                    profiles: dict[str, TemplateProfile]) -> str:
    profile = profiles.get(call.request.model)
    rendered = render_prompt(request=call.request, profile=profile)
    body = [f'<details class="inference-panel" id="{anchor}-inference-{number}">',
            f'<summary>Inference {number}: {escape(call.request.model)} / {escape(call.path)}, {call.elapsed_ms} ms</summary>',
            '<div class="inference-body"><h4>IN / Prompt with model role markers</h4>',
            f'<p class="caption">{escape(rendered.explanation)}</p>']
    if rendered.text is not None:
        body.append('<pre class="rendered-prompt">' + escape(rendered.text) + '</pre>')
    if profile:
        body.append(details(title="Template source and fingerprint", body=block(
            f"{profile.source}\nSHA256: {profile.sha256}\n{profile.provenance}")))
    messages = "".join(message_panel(message=message, number=index)
                       for index, message in enumerate(call.request.messages, start=1))
    body.append(details(title=f"API messages in order ({len(call.request.messages)})", body=messages))
    if call.request.tools:
        tools = "".join(f'<h5>{escape(tool.function.name)}</h5><p>{escape(tool.function.description)}</p>'
                        + block(tool.function.parameters) for tool in call.request.tools)
        body.append(details(title=f"Available tool definitions ({len(call.request.tools)})", body=tools))
    body.append('<h4>OUT / API response before iSPEC guards</h4>')
    if call.response and call.response.choices:
        for index, choice in enumerate(call.response.choices, start=1):
            body.append(f'<p class="caption">Choice {index}; finish reason: {escape(choice.finish_reason or "unavailable")}; '
                        f'parsed tool calls: {len(choice.message.tool_calls)}</p>')
            body.append(message_panel(message=choice.message, number=index))
        body.append('<p class="caption">Tool-shaped text in content is still text. Only the parsed tool calls above are API tool requests. '
                    'Original generation tokens before the runtime parser were not captured.</p>')
    else:
        body.append(block(call.error or 'No response choices captured.'))
    body.append(details(title="Original API JSON", body='<h4>Exact API request (before chat-template rendering)</h4>'
                        + block(call.request.model_dump(exclude_unset=True))
                        + '<h4>Raw model response (before response guards)</h4>'
                        + block(call.response.model_dump(exclude_unset=True) if call.response else call.error)))
    return "".join(body) + '</div></details>'


def case_panel(*, review: CaseReview, profiles: dict[str, TemplateProfile]) -> str:
    case, calls = review.payload, review.payload["tool_calls"]
    wire_path = review.path.with_name("wire.json")
    capture = WireCapture.read(path=wire_path) if wire_path.exists() else None
    body = [f'<details class="case-panel" id="{review.anchor}"><summary>',
            badge(label=review.test.status.upper(), kind=review.test.status),
            f' <strong>{escape(case["case"])}</strong> <span>{escape(case["model"])} / {escape(review.batch)}</span>',
            '</summary><div class="case-body"><a href="#case-index">Back to case index</a>',
            '<div class="outcomes"><div><h3>Recorded test</h3>', block(review.test.detail),
            f'<p class="path">JUnit: {escape(str(review.test.source or "Unavailable"))}</p></div>',
            '<div><h3>Behavioral review</h3>']
    if review.findings:
        body += ["<ul>" + "".join(f"<li>{escape(note)}</li>" for note in review.findings) + "</ul>",
                 f'<p>Review source: {escape(review.finding_source)}</p>']
    else:
        body.append("<p>No manual semantic verdict recorded.</p>")
    if review.signals:
        body.append("<h4>Automatic review signals</h4><ul>"
                    + "".join(f"<li>{escape(signal)}</li>" for signal in review.signals) + "</ul>")
    body += ['</div></div>', f'<p class="path">Evidence: {escape(str(review.path))}</p>',
             block({"expected_comments_by_turn": case["expected_comment_counts"],
                    "actual_comments_by_turn": review.actual})]
    if case.get("negative_checks"):
        body.append(details(title="Negative approval coverage (actual vs diagnostic)", body=(
            '<p>Each entry identifies its source. Diagnostic checks use the actual pre-turn draft and '
            'presentation but never execute their tickets or feed results into the conversation.</p>'
            + block(case["negative_checks"]))))
    if capture:
        body.append(f'<p><a href="#{review.anchor}-prompts" data-open-trace>Read prompts and replies with role tags ({len(capture.calls)} inferences)</a></p>')
    for turn_index, turn in enumerate(case["turns"]):
        metadata = turn["metadata"]
        workflow = metadata.get("project_comment_workflow") or {}
        body += [f'<section class="turn"><h3>Turn {turn_index + 1}</h3><div class="exchange"><div><h4>User</h4>',
                 block(turn["user"]), '</div><div><h4>Delivered assistant reply</h4>', block(turn["assistant"]), '</div></div>',
                 details(title="Actual tool calls and results", body=block([call for call in calls if call["turn"] == turn_index])),
                 details(title="Stored drafts and comments (database state, not chat history)", body=block({"drafts": turn["drafts"], "comments": turn["comments"]})),
                 details(title="Approval, write outcome and response guard", body=block({
                     "save_checks": workflow.get("save_checks", []), "write_outcome": workflow.get("write_outcome"),
                     "server_presentation": workflow.get("presentation"),
                     "write_claim_guard": metadata.get("write_claim_guard"), "elapsed_ms": turn["elapsed_ms"],
                 }))]
        presentation = workflow.get("presentation")
        if presentation:
            body.append('<p class="caption">Server-rendered ' + escape(str(presentation["kind"]))
                        + ' response. This delivered wording is not raw model output; see the inference trace.</p>')
        if capture:
            matching = [number for number, call in enumerate(capture.calls, start=1)
                        if (call.request.latest_user_text or '').strip() == turn["user"].strip()]
            links = ' / '.join(f'<a href="#{review.anchor}-inference-{number}" data-open-trace>Inference {number}</a>' for number in matching)
            body.append('<p>Prompts with this latest user message: ' + (links or 'No exact match; see the complete timeline.') + '</p>')
        body.append('</section>')
    if capture:
        body.append(f'<section id="{review.anchor}-prompts"><h3>Prompt / response timeline</h3>'
                    '<p>Every captured inference is included, including routing and approval checks. '
                    'Per-turn links match the latest user text; repeated identical replies can match more than one turn.</p>')
        body.extend(inference_panel(call=call, number=number, anchor=review.anchor, profiles=profiles)
                    for number, call in enumerate(capture.calls, start=1))
        body.append('</section>')
    else:
        body.append("<p>Wire capture unavailable; delivered text alone cannot reconstruct raw model behavior.</p>")
    return "".join(body) + '</div></details>'


def render_review(*, roots: list[Path], notes: Path | None = None,
                  profiles: dict[str, TemplateProfile] | None = None) -> str:
    reviews = load_reviews(roots=roots, notes=notes)
    rows = []
    for review in reviews:
        case = review.payload
        search = " ".join((review.batch, case["model"], case["case"], *review.signals, *review.findings)).lower()
        label = "Findings" if review.findings else "Review signals" if review.signals else "Unreviewed"
        counts_match = list(review.actual) == case["expected_comment_counts"]
        rows.append(
            f'<tr data-case="{review.anchor}" data-result="{review.test.status}" data-batch="{escape(review.batch)}" '
            f'data-model="{escape(case["model"])}" data-attention="{str(review.needs_review).lower()}" data-search="{escape(search)}">'
            f'<th scope="row"><a href="#{review.anchor}" data-open-case>{escape(case["case"])}</a></th>'
            f'<td>{escape(review.batch)}</td><td>{escape(case["model"])}</td><td>'
            + badge(label=review.test.status.upper(), kind=review.test.status) + '</td><td>'
            + ("Match" if counts_match else "Mismatch") + '</td><td>'
            + badge(label=label, kind="review" if review.needs_review else "unknown")
            + '<small>' + escape("; ".join(review.signals)) + '</small></td></tr>'
        )
    batches = []
    batch_names = list(dict.fromkeys(review.batch for review in reviews))
    models = sorted({review.payload["model"] for review in reviews})
    for batch in batch_names:
        for model in models:
            group = [r for r in reviews if r.batch == batch and r.payload["model"] == model]
            if not group:
                continue
            counts = Counter(r.test.status for r in group)
            summary = " / ".join(f"{counts[status]} {status}" for status in ("pass", "fail", "error", "skipped", "unknown") if counts[status])
            batches.append(f'<button class="batch-card" data-pick-batch="{escape(batch)}" data-pick-model="{escape(model)}">'
                           f'<small>{escape(batch)}</small><strong>{escape(model)}</strong><span>{summary}</span></button>')
    counts = Counter(review.test.status for review in reviews)

    def options(values):
        return "".join(f'<option value="{escape(value)}">{escape(value)}</option>' for value in values)

    return Template((ASSETS / "page.html").read_text()).substitute(
        style=(ASSETS / "style.css").read_text(), script=(ASSETS / "navigation.js").read_text(),
        total=len(reviews), passed=counts["pass"], failed=counts["fail"], errors=counts["error"],
        skipped=counts["skipped"], unknown=counts["unknown"],
        flagged_passes=sum(r.test.status == "pass" and r.needs_review for r in reviews),
        batches="".join(batches), batch_options=options(batch_names), model_options=options(models),
        rows="".join(rows), panels="".join(case_panel(review=r, profiles=profiles or {}) for r in reviews),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--notes", type=Path, help="Optional JSON array of manually reviewed behavioral findings")
    parser.add_argument("--template-map", type=Path, help="Explicit model-to-local-template mapping with SHA256 fingerprints")
    parser.add_argument("--overwrite", action="store_true", help="Replace the generated HTML only")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".html":
        parser.error("--output must be an .html file")
    html = render_review(roots=args.roots, notes=args.notes, profiles=load_profiles(path=args.template_map))
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        handle.write(html)
    print(args.output)


if __name__ == "__main__":
    main()
