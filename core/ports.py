"""Interfaces used by the repair workflow."""

from typing import Any, Protocol

from .models import TestResult


class ChatClient(Protocol):
    """Return model replies for a conversation."""

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Return the next reply to the supplied messages."""
        ...


class RepairToolsPort(Protocol):
    """Provide access to workspace files and test execution."""

    def read_file(self, path: str) -> str:
        """Return the contents of a workspace file."""
        ...

    def edit_file(self, path: str, content: str) -> str:
        """Replace a workspace file's contents and return an observation."""
        ...

    def run_tests(self) -> TestResult:
        """Run the workspace tests and return their outcome."""
        ...


class TraceRecorder(Protocol):
    """Record public repair events when a recorder is supplied."""

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Record a named event and its payload."""
        ...
