"""Thread-based task queue with optional concurrency and error reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Task:
    """Represent a unit of work to be executed."""

    func: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = field(default=None, init=False)

    def run(self) -> None:
        self.func(*self.args, **self.kwargs)


class TaskQueue:
    """Simple task queue running in background threads.

    Tasks are processed by a configurable number of worker threads. The
    default is one worker which preserves the previous sequential behaviour.
    Errors raised by tasks are stored on the task object for later
    inspection.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._queue: Queue[Task | None] = Queue()
        self._stop_event = Event()
        self._max_workers = max_workers
        self._threads: List[Thread] = []
        self._started = False

    def start(self) -> None:
        """Start the worker threads if not already running."""

        if self._started:
            return
        self._stop_event.clear()
        for _ in range(self._max_workers):
            thread = Thread(target=self._worker, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._started = True

    def stop(self) -> None:
        """Signal the worker threads to exit and wait for them."""

        self._stop_event.set()
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            if thread.is_alive():
                thread.join()
        self._threads.clear()
        self._started = False

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Task:
        """Submit a callable to be executed by the queue."""

        task = Task(func, args, kwargs)
        self._queue.put(task)
        return task

    def join(self) -> None:
        """Block until all tasks have been processed."""

        self._queue.join()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.1)
            except Empty:
                continue
            if task is None:
                self._queue.task_done()
                break
            try:
                task.run()
            except Exception as exc:
                task.error = exc
            finally:
                self._queue.task_done()
