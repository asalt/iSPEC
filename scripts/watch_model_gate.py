"""Wait for gate receipt changes or runner exit using Linux kernel events.

Attach this command to the Codex tool session to receive readiness/failure output
immediately. A disconnected session can reattach using the same output folder.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import time


def watch(*, folder: Path, timeout: float) -> int:
    """Print state transitions and wake immediately if the recorded runner exits."""
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    runner_fd = None
    previous = None
    deadline = time.monotonic() + timeout
    try:
        # Receipt writers close their file after a complete JSON write.
        if libc.inotify_add_watch(fd, os.fsencode(folder), 0x8 | 0x80) < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        while True:
            try:
                receipt = json.loads((folder / "receipt.json").read_text())
            except json.JSONDecodeError:
                receipt = {}
            state = receipt.get("state")
            if state and state != previous:
                print(json.dumps({
                    "state": state, "error": receipt.get("error"),
                    "test_exit_code": receipt.get("test_exit_code"),
                }), flush=True)
                previous = state
            if receipt.get("ended_at"):
                return 0 if state == "finished" else 1
            if runner_fd is None and receipt.get("runner_pid"):
                try:
                    runner_fd = os.pidfd_open(receipt["runner_pid"])
                except ProcessLookupError:
                    print("Runner exited without a completion receipt", flush=True)
                    return 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("Wait limit reached; runner has not reported completion", flush=True)
                return 124
            ready, _, _ = select.select(
                [fd] + ([runner_fd] if runner_fd is not None else []), [], [], remaining,
            )
            if fd in ready:
                os.read(fd, 65536)
            if runner_fd in ready:
                final = json.loads((folder / "receipt.json").read_text())
                print(json.dumps({
                    "state": final.get("state"), "error": final.get("error"),
                    "test_exit_code": final.get("test_exit_code"),
                    "completion_receipt": bool(final.get("ended_at")),
                }), flush=True)
                return 0 if final.get("state") == "finished" else 1
    finally:
        os.close(fd)
        if runner_fd is not None:
            os.close(runner_fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    raise SystemExit(watch(folder=args.folder.resolve(), timeout=args.timeout))
