"""Export component-aware usage JSONL to a standalone, read-only graph viewer.

No inference, DB writes, network requests, or raw prompt/context text in the export.
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path

from ispec.prompt.composition import clean_prompt_composition


ROOT = Path(__file__).resolve().parents[1]
MAX_INPUT_BYTES = 16 * 1024 * 1024


def read_calls(source: Path, *, limit: int = 100) -> tuple[list[dict], int]:
    if not 1 <= limit <= 500:
        raise ValueError("Choose between 1 and 500 calls")
    if source.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Choose a bounded log extract of at most 16 MiB")
    calls, skipped = deque(maxlen=limit), 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            composition = (
                clean_prompt_composition(event.get("prompt_composition"))
                if isinstance(event, dict)
                else None
            )
            if composition is None:
                skipped += 1
                continue
            call = {"source_line": line_number, "prompt_composition": composition}
            for key in (
                "ts_utc",
                "session_id",
                "message_id",
                "provider",
                "model",
                "stage",
                "surface",
                "ok",
                "elapsed_ms",
            ):
                value = event.get(key)
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if value is None or type(value) in (str, int, float, bool):
                    call[key] = value if not isinstance(value, str) else value[:256]
            calls.append(call)
    return list(calls), skipped


def export_view(source: Path, output: Path, *, limit: int = 100) -> dict:
    calls, skipped = read_calls(source, limit=limit)
    data = json.dumps(
        {"calls": calls, "skipped": skipped}, ensure_ascii=True, allow_nan=False
    ).replace("<", "\\u003c")
    template = (ROOT / "docs/prompts/composition-viewer.html").read_text()
    html = template.replace("/* COMPOSITION_DATA */ {}", data)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(html)
    return {"output": str(output), "calls": len(calls), "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(export_view(args.source, args.output, limit=args.limit)))


if __name__ == "__main__":
    main()
