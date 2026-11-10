"""Assemble the repair service and deterministic demonstration clients."""

import json
import os
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from adapters.models.scripted import ScriptedClient
from adapters.models.ollama import OllamaClient
from adapters.sandbox.repair_tools import RepairTools
from adapters.sandbox.runner import EVALS_ROOT, create_workspace
from core.repair_agent import RepairAgent
from core.repair_service import RepairService

# Each replacement addresses the bug in its supplied fixture.
REPAIRS = {
    "arithmetic": ("calculator.py", "return left - right", "return left + right"),
    "balanced_brackets": ("brackets.py", "if not stack:", "if not stack or stack[-1] != pairs[character]:"),
    "median": ("statistics_tools.py", "    return ordered[middle]",
               "    if len(ordered) % 2 == 0:\n        return (ordered[middle - 1] + ordered[middle]) / 2\n    return ordered[middle]"),
    "multi_function": ("text_tools.py", "normalize_name(name).upper()", "normalize_name(name).title()"),
    "unique_words": ("word_tools.py", "sorted(set(text.lower().split()))", "list(dict.fromkeys(text.lower().split()))"),
}


def _credential(name: str) -> str:
    """Read a credential from the environment or the project .env file."""
    value = os.environ.get(name, "")
    if value:
        return value
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and key.strip() == name:
                try:
                    values = shlex.split(value, comments=True)
                except ValueError:
                    raise ValueError(f"Invalid {name} entry in .env") from None
                return values[0] if len(values) == 1 else ""
    return ""


def build_service(
    provider: str = "scripted",
    on_event: Callable[[dict[str, Any]], None] | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    research: bool = False,
    observe: bool = False,
) -> RepairService:
    """Wire fresh clients and workspace tools for each selected repair."""
    if provider not in {"scripted", "ollama"}:
        raise ValueError(f"Unsupported provider: {provider}")

    token = _credential("GITHUB_TOKEN") if research else ""
    credentials = {}
    if observe:
        names = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
        credentials = {name: _credential(name) for name in names}
        missing = [name for name, value in credentials.items() if not value.strip()]
        if missing:
            raise ValueError("Missing Langfuse credentials: " + ", ".join(missing))

    def create_agent(case_name: str, workspace: Path) -> RepairAgent:
        recorder = None
        if observe:
            from adapters.observability.langfuse import LangfuseTrajectoryRecorder

            recorder = LangfuseTrajectoryRecorder(
                credentials["LANGFUSE_PUBLIC_KEY"], credentials["LANGFUSE_SECRET_KEY"],
                credentials["LANGFUSE_BASE_URL"],
            )
        research_port = None
        if research:
            from adapters.research.github_research import GitHubResearch

            research_port = GitHubResearch(token)
        if provider == "ollama":
            client = OllamaClient(model=model, temperature=temperature)
            return RepairAgent(client, RepairTools(workspace), research_port, recorder)
        filename, old, new = REPAIRS[case_name]
        original = (workspace / filename).read_text(encoding="utf-8")
        if original.count(old) != 1:
            raise ValueError("Fixture does not match the prepared scripted repair")
        replies = [json.dumps(action) for action in (
            {"tool": "read_file", "path": filename, "summary": "Inspect the source"},
            {"tool": "run_tests", "summary": "Reproduce the failure"},
            {"tool": "edit_file", "path": filename, "content": original.replace(old, new),
             "summary": "Apply the focused repair"},
            {"tool": "run_tests", "summary": "Verify the repair"},
        )]
        return RepairAgent(ScriptedClient(replies), RepairTools(workspace), research_port, recorder)

    return RepairService(EVALS_ROOT, tuple(REPAIRS), create_workspace, create_agent, on_event)
