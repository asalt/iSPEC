"""AI helpers for interacting with LLMs and tensor utilities.

This package provides skeleton implementations for chat sessions
that may later be wired to a large language model as well as simple
numerical helpers that operate on Python lists as lightweight tensors.
"""

from .chat import ChatMessage, ChatSession
from .llm import generate_response, handle_user_message
from .tensor_ops import matmul, transpose
from .task_queue import TaskQueue
from .worker import ChatWorker

__all__ = [
    "ChatMessage",
    "ChatSession",
    "ChatWorker",
    "TaskQueue",
    "generate_response",
    "handle_user_message",
    "matmul",
    "transpose",
]
