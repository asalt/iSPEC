"""Thread-based task queue for processing work items sequentially."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Event, Thread
from typing import Any, Callable, Dict, Tuple


@dataclass
class Task:
    """Represent a unit of work to be executed."""

    func: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def run(self) -> None:
        self.func(*self.args, **self.kwargs)


class TaskQueue:
    """Simple task queue running in a dedicated thread.

    The queue uses a background thread to execute submitted tasks one after
    the other. It can be started and stopped to control processing.
    """

    def __init__(self) -> None:
        self._queue: Queue[Task | None] = Queue()
        self._stop_event = Event()
        self._thread = Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        """Start the worker thread if not already running."""

        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        """Signal the worker thread to exit and wait for it."""

        self._stop_event.set()
        self._queue.put(None)
        self._thread.join()

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Submit a callable to be executed by the queue."""

        self._queue.put(Task(func, args, kwargs))

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
                break
            try:
                task.run()
            finally:
                self._queue.task_done()
