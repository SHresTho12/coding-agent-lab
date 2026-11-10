"""Assemble the repair service and deterministic demonstration clients."""

import json
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


def build_service(
    provider: str = "scripted",
    on_event: Callable[[dict[str, Any]], None] | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> RepairService:
    """Wire fresh clients and workspace tools for each selected repair."""
    if provider not in {"scripted", "ollama"}:
        raise ValueError(f"Unsupported provider: {provider}")

    def create_agent(case_name: str, workspace: Path) -> RepairAgent:
        if provider == "ollama":
            client = OllamaClient(model=model, temperature=temperature)
            return RepairAgent(client, RepairTools(workspace))
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
        return RepairAgent(ScriptedClient(replies), RepairTools(workspace))

    return RepairService(EVALS_ROOT, tuple(REPAIRS), create_workspace, create_agent, on_event)
