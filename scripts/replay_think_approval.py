"""Local-only Think replay of the frozen y1 requests. No controller or write tools."""

import argparse
from bisect import bisect_right
from itertools import accumulate
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from jsonschema import Draft202012Validator
from transformers import AutoTokenizer

from scripts import approval_action_trial as trial
from scripts import replay_approval_wording as wording


ROUND2 = trial.ROOT / "tests/behavioral/local/approval-wording-round2-4ebd366"
PROFILE = trial.ROOT.parent / "configs/models/olmo7b-think-vllm.env"
CHECKPOINT = Path("/home/alex/amms06/mnt/e/ispec-model-cache/hub/models--allenai--Olmo-3-7B-Think/"
                  "snapshots/d97e442d7cc678210054dbcc9b440894d62c89a4")


def inspect_tokens(tokenizer, ids):
    """Count whole generated tokens preceding the first closing tag, excluding its boundary tokens."""
    pieces = tokenizer.convert_ids_to_tokens(ids)
    encoded = "".join(pieces)
    offset = encoded.find("</think>")
    if offset < 0:
        return None, []
    ends = list(accumulate(map(len, pieces)))
    before = bisect_right(ends, offset)
    after = bisect_right(ends, offset + len("</think>") - 1) + 1
    return before, ids[before:after]


def infer(*, port, request, tokenizer, validator):
    post = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                  data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.monotonic()
    with opener.open(post, timeout=210) as response:
        result = json.load(response)
    elapsed = time.monotonic() - start
    choice, = result["choices"]
    message = choice["message"]
    reason = message.get("reasoning") or message.get("reasoning_content") or ""
    content = message.get("content") or ""
    try:
        answer = json.loads(content)
    except ValueError:
        answer = None
    valid = choice["finish_reason"] == "stop" and validator.is_valid(answer) and not message.get("tool_calls")
    usage = result["usage"]
    ids = choice["token_ids"]
    count, boundary = inspect_tokens(tokenizer, ids)
    return dict(label=answer["label"] if valid else None, confidence=answer["confidence"] if valid else None,
                valid=valid, finish=choice["finish_reason"], seconds=elapsed, usage=usage,
                reasoning_tokens=count, delimiter_ids=boundary, generated_ids=len(ids),
                # Preserve third-party model output separately from delivered answers for forensic review.
                parsed_reasoning=reason, final_content=content, separated=bool(reason and content),
                prompt_ids=result["prompt_token_ids"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output.resolve()
    occupancy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if occupancy.strip():
        raise RuntimeError("Local GPU has a compute process; refusing startup.")
    out.mkdir(parents=True, exist_ok=False)
    profile = trial.gate.read_profile(PROFILE)
    rows = trial.read(ROUND2 / "inputs.json")
    canonical = {v["key"]: v["request"] for v in wording.records(ROUND2 / "replay.jsonl") if v["arm"] == "y1"}
    assert len(canonical) == len(rows) == 24 and sum(r["called"] for r in rows) == 10
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    env = {
        "PATH": "/home/alex/venvs/py3.12/bin:/usr/bin:/bin", "HOME": str(out / "home"), "LANG": "C.UTF-8",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        "VLLM_CACHE_ROOT": str(out / "vllm-cache"), "CUDA_CACHE_PATH": str(out / "cuda-cache"),
        "TRITON_CACHE_DIR": str(out / "triton-cache"),
    }
    receipt = {"services": [], "model": profile["ISPEC_VLLM_MODEL"], "checkpoint": str(CHECKPOINT),
               "profile_sha256": trial.gate.digest(PROFILE), "template_sha256": trial.gate.digest(CHECKPOINT / "chat_template.jinja"),
               "inputs_sha256": trial.gate.digest(ROUND2 / "inputs.json"), "data_origin": "synthetic_evaluation",
               "reasoning_parser": "olmo3", "total_output_budget": 2048, "production": False}
    def command(fd, port):
        args = [trial.gate.GPU_PYTHON, str(trial.ROOT / "scripts/run_model_regression.py"), "--vllm-child", str(fd),
                "--model", str(CHECKPOINT), "--served-model-name", profile["ISPEC_VLLM_MODEL"], "--host", "127.0.0.1",
                "--port", str(port), "--dtype", profile["VLLM_DTYPE"], "--max-model-len", profile["VLLM_MAX_MODEL_LEN"],
                "--max-num-seqs", "1", "--gpu-memory-utilization", "0.8", "--enforce-eager",
                "--attention-backend", "TRITON_ATTN", "--generation-config", "vllm",
                "--enable-auto-tool-choice", "--tool-call-parser", profile["ISPEC_VLLM_TOOL_CALL_PARSER"],
                "--reasoning-parser", profile["ISPEC_VLLM_REASONING_PARSER"], "--no-enable-log-requests"]
        receipt["server_command"] = args
        return args

    with trial.endpoint(out, "think", command, env, 18030, receipt) as (port, _):
        with (out / "results.jsonl").open("x", buffering=1) as log:
            def run(row, off=False):
                req = {**canonical[row["key"]], "model": profile["ISPEC_VLLM_MODEL"],
                       "max_tokens": 2048, "return_token_ids": True}
                if off:
                    req["chat_template_kwargs"] = {"enable_thinking": False}
                observed = infer(port=port, request=req, tokenizer=tokenizer,
                                 validator=Draft202012Validator(req["structured_outputs"]["json"]))
                print(json.dumps(dict(key=row["key"], off=off, called=row["called"], wanted=row["wanted"],
                                      **observed)), file=log)
                return observed

            for index, row in enumerate(rows):
                observed = run(row)
                if index == 0:
                    print("First inference complete:", row["key"], observed["finish"], flush=True)
                    probe = run(row, off=True)
                    same = observed["prompt_ids"] == probe["prompt_ids"]
                    receipt["off_probe"] = dict(key=row["key"], identical_prompt_ids=same,
                                                emitted_reasoning=bool(probe["parsed_reasoning"]),
                                                same_final=probe["final_content"] == observed["final_content"])
                    off_effective = not probe["parsed_reasoning"] and probe["valid"] and not same
                    receipt["full_off_arm"] = off_effective
                elif off_effective:
                    run(row, off=True)


if __name__ == "__main__":
    main()
