"""Gradio controls for repairing and downloading supplied evaluation cases."""

from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZIP_DEFLATED, ZipFile

import gradio as gr

from core.repair_service import RepairService


PROJECTS = [
    "arithmetic",
    "balanced_brackets",
    "median",
    "multi_function",
    "unique_words",
]


def _repair(service: RepairService, selected: str) -> tuple[str, str, str, str | None]:
    """Run the service and prepare plain-text results and a workspace archive."""
    if selected not in PROJECTS:
        return "Select one of the supplied projects.", "", "", None
    try:
        result = service.repair(f"evals/{selected}")
        workspace = Path(result.workspace).resolve(strict=True)
        trace = "\n\n".join(
            "\n".join(f"{key}: {value}" for key, value in event.items())
            for event in result.trace
        )
        changed = []
        for relative in result.changed_files:
            path = (workspace / relative).resolve(strict=True)
            if not path.is_relative_to(workspace) or not path.is_file():
                raise ValueError("Changed file is outside the repaired workspace")
            changed.append(f"{relative}\n{path.read_text(encoding='utf-8')}")

        files = sorted(workspace.rglob("*"))
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                raise ValueError("Workspace contains an unsafe archive entry")
            if not path.is_file() and not path.is_dir():
                raise ValueError("Workspace contains an unsupported archive entry")
        archive = Path(mkdtemp(prefix="coding-agent-lab-download-")) / f"{selected}.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as download:
            for path in files:
                download.write(path, path.relative_to(workspace).as_posix())

        outcome = "Repair passed" if result.passed else "Repair failed"
        test = result.last_test_result
        output = "No test result available"
        if test is not None:
            output = f"Test exit code: {test.returncode}\n{test.output}"
        return trace, "\n\n".join(changed) or "No changed files", f"{outcome}\n{output}", str(archive)
    except (OSError, ValueError, RuntimeError) as exc:
        return f"Repair could not complete: {exc}", "", "", None


def create_demo(service: RepairService) -> gr.Blocks:
    """Build a page using an already configured repair service."""
    with gr.Blocks(title="Code Repair Lab", analytics_enabled=False) as demo:
        gr.Markdown("# Code Repair Lab\nSelect a supplied project and start a repair.")
        project = gr.Dropdown(choices=PROJECTS, value="arithmetic", label="Project")
        start = gr.Button("Start repair", variant="primary")
        trace = gr.Textbox(label="Visible trace", lines=12, interactive=False)
        changed = gr.Textbox(label="Changed relative files and text", lines=12, interactive=False)
        output = gr.Textbox(label="Final test output", lines=10, interactive=False)
        download = gr.File(label="Download repaired workspace ZIP", interactive=False)
        start.click(
            fn=lambda selected: _repair(service, selected),
            inputs=project,
            outputs=[trace, changed, output, download],
            concurrency_limit=1,
            api_name=False,
        )
    return demo
