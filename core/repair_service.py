"""Select an evaluation case and coordinate its disposable repair workspace."""

from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import Any

from .models import RepairResult
from .repair_agent import RepairAgent


class RepairService:
    """Connect validated case selection to injected workspace and agent factories."""

    def __init__(
        self,
        evals_root: Path,
        projects: tuple[str, ...],
        create_workspace: Callable[[str], Path],
        create_agent: Callable[[str, Path], RepairAgent],
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.evals_root = evals_root.resolve()
        self.projects = projects
        self.create_workspace = create_workspace
        self.create_agent = create_agent
        self.on_event = on_event

    def repair(self, selected_project: str) -> RepairResult:
        """Validate an evals/project selection and repair a fresh copy."""
        selection = Path(selected_project)
        if selection.anchor or PureWindowsPath(selected_project).anchor:
            raise ValueError("Select a relative evals/project path")
        if len(selection.parts) != 2 or selection.parts[0] != "evals":
            raise ValueError("Select one project directly under evals/")
        case_name = selection.parts[1]
        source = self.evals_root / case_name
        if case_name not in self.projects or not source.is_dir() or source.resolve().parent != self.evals_root:
            raise ValueError("Unknown or unsafe evaluation project")
        workspace = self.create_workspace(case_name)
        files = sorted(path.relative_to(workspace).as_posix()
                       for path in workspace.rglob("*") if path.is_file())
        agent = self.create_agent(case_name, workspace)
        return agent.repair(case_name, files, str(workspace), on_event=self.on_event)
