"""Resolve pinned runtime chat templates from a checkout or installed package."""

import argparse
import hashlib
from pathlib import Path


TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates" / "chat"
OLMO3_TEMPLATE_SHA256 = "69c9c09e966a5e36f61ddfff4c655e10138178ad291ac830f4008925e6ac55f4"


def chat_template_path(*, name: str) -> Path:
    """Return the absolute template path after verifying its reviewed bytes."""
    if name != "olmo3":
        raise ValueError(f"Unknown packaged chat template: {name}")
    path = TEMPLATE_ROOT / "olmo3.j2"
    if hashlib.sha256(path.read_bytes()).hexdigest() != OLMO3_TEMPLATE_SHA256:
        raise ValueError("OLMo chat template differs from its reviewed SHA256")
    return path


def main() -> None:
    """Print a verified path for --chat-template without starting a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["olmo3"])
    args = parser.parse_args()
    print(chat_template_path(name=args.name))


if __name__ == "__main__":
    main()
