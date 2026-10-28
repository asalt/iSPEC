"""Print a nonsecret, read-only runtime inventory using the selected Python.

Run with each environment's interpreter, locally or via SSH stdin. This is a
compatibility receipt, not a binary backup or an environment/credential dump.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = ("vllm", "torch", "transformers", "triton", "gguf", "ollama",
            "huggingface-hub", "safetensors", "numpy", "xformers")
COMMANDS = {
    "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
    "gcc": ["gcc", "--version"],
    "gxx": ["g++", "--version"],
    "nvcc": ["nvcc", "--version"],
    "ollama": ["ollama", "--version"],
}


def command_version(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        return {"exit_code": result.returncode, "output": (result.stdout + result.stderr).strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def snapshot(*, label, checkpoint=None, probe_cuda=False):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "schema_version": 1, "label": label,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "system": {"os": platform.system(), "release": platform.release(),
                   "architecture": platform.machine(), "libc": platform.libc_ver()},
        "packages": packages,
        "commands": {key: command_version(argv) for key, argv in COMMANDS.items()},
        "limitations": "No process environments, credentials, package indexes, or full machine image. Installed compilers are not proof of wheel build provenance.",
    }
    if checkpoint is not None:
        root = Path(checkpoint)
        files = {}
        for name in ("config.json", "chat_template.jinja", "model.safetensors.index.json"):
            path = root / name
            if path.is_file():
                files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        config = json.loads((root / "config.json").read_text())
        result["checkpoint"] = {"path": str(root), "revision_directory": root.name,
                                "dtype": config.get("dtype", config.get("torch_dtype")),
                                "architectures": config.get("architectures"), "files": files,
                                "weights": [{"name": p.name, "bytes": p.stat().st_size,
                                             "cache_blob": p.resolve().name}
                                            for p in sorted(root.glob("*.safetensors"))]}
    if probe_cuda:
        try:
            import torch
            probe = {"cuda_runtime": torch.version.cuda, "available": torch.cuda.is_available()}
            if probe["available"]:
                probe["capability"] = torch.cuda.get_device_capability()
                probe["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
                probe["fp16_add"] = float((torch.ones(1, device="cuda", dtype=torch.float16) + 1).item())
            result["cuda_probe"] = probe
        except Exception as exc:
            result["cuda_probe"] = {"error": type(exc).__name__}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Nonsecret host/environment description")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe-cuda", action="store_true", help="Allocate one FP16 GPU element to check execution")
    args = parser.parse_args()
    print(json.dumps(snapshot(label=args.label, checkpoint=args.checkpoint, probe_cuda=args.probe_cuda), indent=2))


if __name__ == "__main__":
    main()
