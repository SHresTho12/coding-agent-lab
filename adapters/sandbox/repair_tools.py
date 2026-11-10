"""Workspace file operations exposed to the repair agent."""

import os
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path, PureWindowsPath
from typing import Iterator
from uuid import uuid4

from core.models import TestResult

from . import runner


def safe_path(workspace: str | Path, requested_path: str) -> Path:
    """Resolve a relative file path and require its target to stay in the workspace."""
    relative = Path(requested_path)
    windows_path = PureWindowsPath(requested_path)
    if relative.anchor or windows_path.anchor:
        raise ValueError("Use a workspace-relative path")
    if os.name == "nt" and any(":" in part or PureWindowsPath(part).is_reserved()
                               for part in windows_path.parts):
        raise ValueError("Use a regular workspace file path")

    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes the workspace")
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return path
    if stat.S_ISDIR(mode):
        raise IsADirectoryError(path)
    if not stat.S_ISREG(mode):
        raise ValueError("Path must refer to a regular file")
    return path


@contextmanager
def _file_location(workspace: Path, path: Path) -> Iterator[tuple[Path | str, int | None]]:
    """Pin parent directories without following links on POSIX.

    Windows uses canonical paths and assumes no concurrent host-side path swaps.
    The workspace and its ancestors must be controlled by the application.
    """
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        yield path, None
        return

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        parent = os.open(workspace, flags)
        stack.callback(os.close, parent)
        for part in path.relative_to(workspace).parts[:-1]:
            parent = os.open(part, flags, dir_fd=parent)
            stack.callback(os.close, parent)
        yield path.name, parent


class RepairTools:
    """Read and replace workspace files and delegate test execution to Docker."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = runner.validate_workspace(workspace)
        self.changed_files: list[str] = []

    def read_file(self, path: str) -> str:
        """Return the UTF-8 contents of a workspace file."""
        target = safe_path(self.workspace, path)
        with _file_location(self.workspace, target) as (name, parent):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            fd = os.open(name, flags, dir_fd=parent)
            with os.fdopen(fd, "r", encoding="utf-8") as file:
                info = os.fstat(file.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("Only regular files without hard links can be read")
                return file.read()

    def edit_file(self, path: str, content: str) -> str:
        """Atomically replace or create a UTF-8 file within an existing directory."""
        target = safe_path(self.workspace, path)
        data = content.encode("utf-8")
        with _file_location(self.workspace, target) as (name, parent):
            temporary = target.with_name(f".edit-{uuid4().hex}")
            if parent is not None:
                temporary = temporary.name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            fd = os.open(temporary, flags, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as file:
                    file.write(data)
                    if parent is not None:
                        os.fchmod(file.fileno(), 0o644)
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            except BaseException:
                os.unlink(temporary, dir_fd=parent)
                raise

        relative = target.relative_to(self.workspace).as_posix()
        if relative not in self.changed_files:
            self.changed_files.append(relative)
        return f"Updated {relative}"

    def run_tests(self) -> TestResult:
        """Run the workspace's tests through the Docker runner."""
        return runner.run_tests(self.workspace)
