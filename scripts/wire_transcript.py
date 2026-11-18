"""Render saved HTTP conversation evidence without running a model."""
from difflib import unified_diff
import json
from pathlib import Path
import sys
import textwrap


def pretty(value):
    if not isinstance(value, str):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return "\n".join(textwrap.fill(line, width=100, break_long_words=False,
                                   replace_whitespace=False) for line in value.split("\n"))


def render(wire):
    data = json.loads(wire.read_text())
    lines = ["# " + data.get("nodeid", "Conversation"), ""]
    previous = {}
    for index, call in enumerate(data["calls"], 1):
        path = call["path"]
        if path == "/tokenize":
            lines.append(f"Token-budget preflight (wire call {index}); no model reply.\n")
            continue
        req = call["request"]
        router = "primary" in req.get("structured_outputs", {}).get("json", {}).get("properties", {})
        stage = "Router" if router else path
        lines.append(f"## Wire call {index}: {stage}")
        msgs = req.get("messages", [])
        oldsys, oldmsgs = previous.get(stage, (None, []))
        system = "\n\n".join(pretty(m.get("content")) for m in msgs if m["role"] == "system")
        diff = system if oldsys is None else "\n".join(unified_diff(oldsys.splitlines(), system.splitlines(), lineterm=""))
        lines.extend(["### System", diff or "(unchanged)", "### Offered tools"])
        for spec in req.get("tools") or []:
            f = spec["function"]
            lines.append(f"- **{f['name']}**: " + pretty(f.get("description","")) + "\n  Arguments: " + ", ".join(f.get("parameters",{}).get("properties",{})))
        if not req.get("tools"):
            lines.append("(none)")
        turns = [m for m in msgs if m["role"] != "system"]
        prefix = 0
        while prefix < min(len(oldmsgs),len(turns)) and turns[prefix] == oldmsgs[prefix]:
            prefix += 1
        lines.append(f"### Messages ({prefix} unchanged at start)")
        for m in turns[prefix:]:
            lines.extend([f"**{m['role']}**", pretty(m.get("content") or "")])
            if m.get("tool_calls"):
                lines.append(pretty(m["tool_calls"]))
        previous[stage] = system, turns
        if not msgs:
            lines.append(pretty(req))
        lines.append("### Model output")
        for ch in (call.get("response") or {}).get("choices", []):
            msg = ch["message"]
            lines.append(pretty(msg.get("content") or ""))
            if msg.get("tool_calls"):
                lines.append(pretty(msg["tool_calls"]))
        if not (call.get("response") or {}).get("choices"):
            lines.append(pretty(call.get("response") or call.get("error")))
    lines.append("## Delivered replies and executor evidence")
    for filename in ("directory.json", "conversation.json"):
        evidence = wire.with_name(filename)
        if evidence.exists():
            record = json.loads(evidence.read_text())
            for result in record.get("calls", record.get("tool_calls", [])):
                lines.append(pretty(result))
            for turn in record.get("turns", []):
                lines.extend(["**User**", pretty(turn.get("user")), "**Delivered**", pretty(turn.get("assistant"))])
            if "reply" in record:
                lines.append(pretty(record["reply"]))
            break
    else:
        lines.append("No delivered-reply record captured; model output above is not proof of delivery.")
    wire.with_name("transcript.md").write_text("\n\n".join(lines)+"\n")


if __name__ == "__main__":
    for root in map(Path, sys.argv[1:]):
        for wire in root.rglob("wire.json"):
            render(wire)
