"""Offline numeric/date/identifier review of saved synthetic gate evidence."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ispec.assistant.draft_fact_review import review_draft_facts


def review_evidence(*, root: Path) -> list[dict]:
    results = []
    paths = sorted({path.resolve() for path in root.rglob("conversation.json")})
    for path in paths:
        conversation = json.loads(path.read_text())
        sources = []
        previous_ids = set()
        for index, turn in enumerate(conversation["turns"], start=1):
            sources.append(turn["user"])
            for draft in turn["drafts"]:
                if draft["id"] in previous_ids:
                    continue
                previous_ids.add(draft["id"])
                review = review_draft_facts(draft=draft["text"], sources=tuple(sources))
                results.append({"case": conversation["case"], "turn": index, "draft": draft["text"],
                                "sources": list(sources), "review": asdict(review)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"advisory_only": True, "source": str(args.evidence), "results": review_evidence(root=args.evidence)}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
