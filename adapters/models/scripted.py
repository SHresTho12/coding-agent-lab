"""Deterministic chat client for testing the repair workflow."""

from collections import deque
from typing import Any


class ScriptedClient:
    """Return prewritten replies in the order they were supplied."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = deque(replies)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Return the next reply or raise when the script is exhausted."""
        if not self._replies:
            raise RuntimeError("ScriptedClient has no replies remaining")
        return self._replies.popleft()
