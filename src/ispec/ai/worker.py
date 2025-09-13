"""Background worker integrating chat sessions, task queue and backend API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .api import put_response
from .chat import ChatSession
from .llm import generate_response
from .task_queue import TaskQueue


@dataclass
class ChatWorker:
    """Process user messages using a background task queue."""

    backend_url: Optional[str] = None
    session: ChatSession = field(default_factory=ChatSession)
    queue: TaskQueue = field(default_factory=TaskQueue)

    def start(self) -> None:
        self.queue.start()

    def stop(self) -> None:
        self.queue.stop()

    def enqueue(self, content: str) -> None:
        """Queue a user message for processing."""

        self.queue.add_task(self._process_message, content)

    # internal --------------------------------------------------------------
    def _process_message(self, content: str) -> None:
        self.session = self.session.add_user_message(content)
        self.session = generate_response(self.session)
        if self.backend_url:
            put_response(self.backend_url, {"response": self.session.messages[-1].content})
