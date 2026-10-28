"""Replay prepared chat requests using local cached weights; no server or tools.

Standalone Python 3.10 compatible so an older CUDA-compatible environment can
evaluate model content without altering the backend's production environment.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if not args.model_path.is_dir() or args.max_tokens <= 0:
        parser.error("An existing model directory and positive token budget are required")
    if args.output.exists():
        parser.error("Choose a new output path; preserve earlier trials")
    os.environ.setdefault("USE_TF", "0")
    import torch
    import transformers
    from transformers.utils import import_utils

    # This text-only replay does not need the optional torchvision binary.
    # Disable its discovery in this process, not in the shared installation.
    import_utils._torchvision_available = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing an accidental CPU run")
    torch.set_num_threads(8)
    torch.manual_seed(0)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    info = {"model_path": str(args.model_path), "torch": torch.__version__,
            "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
            "dtype": "float16", "load_seconds": time.monotonic() - started,
            "constrained_json": False, "tool_execution": False,
            "optional_torchvision_disabled_for_text_only_replay": True}
    print(json.dumps(info), flush=True)
    results = []
    for request in json.loads(args.requests.read_text()):
        messages = request["messages"]
        budget = request.get("max_tokens", args.max_tokens)
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError("Positive per-request output budget required")
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to("cuda")
        if inputs["input_ids"].shape[-1] + budget > 8192:
            raise ValueError("Request exceeds the comparison context budget")
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        tokens = output[0, inputs["input_ids"].shape[-1]:]
        result = {"id": request["id"], "elapsed_ms": int((time.monotonic()-started)*1000),
                  "output_tokens": len(tokens), "content": tokenizer.decode(tokens, skip_special_tokens=True),
                  "input_tokens": inputs["input_ids"].shape[-1], "max_tokens": budget,
                  "ok": True, "finish_reason": "stop" if int(tokens[-1]) == tokenizer.eos_token_id else "length",
                  "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                  "sent": False, "saved": False}
        if request.get("expected_decision") is not None:
            result["expected_decision"] = request["expected_decision"]
            try:
                decision = json.loads(result["content"])
                result["matches_expected"] = decision.get("decision") == request["expected_decision"]
            except (ValueError, AttributeError):
                result["matches_expected"] = False
        results.append(result)
        args.output.write_text(json.dumps({"runtime": info, "results": results}, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
