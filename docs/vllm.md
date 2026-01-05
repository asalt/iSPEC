# vLLM setup (WIP)

iSPEC can talk to a local vLLM server via the OpenAI-compatible Chat Completions API.

## Start vLLM

Example launch (from your vLLM environment):

```bash
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

vllm serve allenai/Llama-3.1-Tulu-3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --enable-chunked-prefill
```

Tuning notes:

- If VRAM is tight, try lowering `--max-model-len`.
- If you expect only a couple concurrent users, try `--max-num-seqs 4`.
- If you want smoother latency, try `--max-num-batched-tokens 2048` (4096 often yields better throughput).
- If you need the desktop GUI responsive, try `--gpu-memory-utilization 0.80–0.85`.

## Configure iSPEC to use vLLM

In `iSPEC/.env` (or your environment):

```bash
ISPEC_ASSISTANT_PROVIDER=vllm
ISPEC_VLLM_URL=http://127.0.0.1:8000
ISPEC_VLLM_MODEL=allenai/Llama-3.1-Tulu-3-8B
```

## Prompt customization (optional)

Prefer appending small tweaks (keeps the built-in tool list + response format rules intact):

```bash
ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA_PATH=docs/prompts/ispec-extra.txt
```

For a full system prompt replacement:

```bash
ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH=docs/prompts/ispec-system.txt
```

If your vLLM server requires an API key, set:

```bash
ISPEC_VLLM_API_KEY=...
```
