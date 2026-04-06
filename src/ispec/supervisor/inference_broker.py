from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from ispec.assistant.service import AssistantReply, generate_reply
from ispec.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class InferenceRequest:
    """A single blocking LLM call to be executed by the inference thread."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stage: str = "answer"
    vllm_extra_body: dict[str, Any] | None = None
    observability_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class InferenceJob:
    job_id: str
    command_id: int
    request: InferenceRequest


@dataclass(frozen=True)
class InferenceResult:
    job_id: str
    command_id: int
    reply: AssistantReply


class InferenceBroker:
    """Runs model inference on a dedicated thread (single-lane).

    Main thread submits `InferenceJob`s via `submit(...)` and receives results
    via `poll_result(...)` or `drain_results(...)`.

    Important invariant:
    - The inference thread must never touch SQLite or other shared state.
    """

    def __init__(self) -> None:
        self._in_q: queue.Queue[InferenceJob | None] = queue.Queue()
        self._out_q: queue.Queue[InferenceResult] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="supervisor-inference", daemon=True)

    @property
    def thread(self) -> threading.Thread:
        return self._thread

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()

    def stop(self, *, join_seconds: float = 2.0) -> None:
        self._stop.set()
        try:
            # Unblock the queue.get() call.
            self._in_q.put_nowait(None)
        except Exception:
            pass
        try:
            self._thread.join(timeout=max(0.0, float(join_seconds)))
        except Exception:
            pass

    def submit(self, *, command_id: int, request: InferenceRequest) -> str:
        job_id = uuid.uuid4().hex
        self._in_q.put(InferenceJob(job_id=job_id, command_id=int(command_id), request=request))
        return job_id

    def poll_result(self) -> InferenceResult | None:
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def drain_results(self, *, limit: int = 50) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for _ in range(max(0, int(limit))):
            item = self.poll_result()
            if item is None:
                break
            results.append(item)
        return results

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                continue
            try:
                req = job.request
                reply = generate_reply(
                    messages=req.messages,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    stage=req.stage,  # type: ignore[arg-type]
                    vllm_extra_body=req.vllm_extra_body,
                    observability_context={
                        "surface": "supervisor_task",
                        "command_id": int(job.command_id),
                        "stage": req.stage,
                        **(req.observability_context or {}),
                    },
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("Inference worker crashed during generate_reply: %s", error)
                reply = AssistantReply(
                    content=f"Assistant error: {error}",
                    provider="supervisor_inference_worker",
                    model=None,
                    meta={"error": repr(exc)},
                    ok=False,
                    error=error,
                )
            try:
                self._out_q.put(InferenceResult(job_id=job.job_id, command_id=int(job.command_id), reply=reply))
            except Exception:
                # Last resort: swallow. The supervisor loop will treat this as a timeout and retry.
                logger.exception("Inference worker failed to publish result (job_id=%s)", job.job_id)

