"""Shared data models for the repair workflow."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TestResult:
    """Capture the outcome and output of a test run."""

    __test__ = False

    passed: bool
    output: str
    returncode: int


@dataclass
class ModelAction:
    """Describe a tool action requested by the model."""

    tool: str
    arguments: dict[str, Any]
    public_summary: str | None = None


@dataclass
class RepairResult:
    """Describe the outcome and ordered trace of a repair attempt."""

    passed: bool
    trace: list[dict[str, Any]]
    changed_files: list[str]
    workspace: str
    last_test_result: TestResult | None
