"""Copy evaluation cases and run their tests inside a restricted container."""

import shutil
import stat
import subprocess
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from core.models import TestResult


EVALS_ROOT = Path(__file__).resolve().parents[2] / "evals"
IMAGE = "coding-agent-lab-runner"
DEFAULT_TIMEOUT = 30.0


def _workspace_root() -> Path:
    root = Path.home().resolve() / ".cache" / "coding-agent-lab"
    if root.resolve() != root:
        raise ValueError("The workspace cache must not be redirected by symlinks")
    return root


def _check_source_entries(directory: str, names: list[str]) -> list[str]:
    """Reject links and special files before copytree processes a directory."""
    for name in names:
        entry = Path(directory) / name
        info = entry.lstat()
        is_reparse_point = getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        if is_reparse_point or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"Unsafe evaluation case entry: {entry}")
    return []


def create_workspace(case: str | Path, evals_root: str | Path = EVALS_ROOT) -> Path:
    """Copy a case path (absolute or relative to evals_root) into a fresh workspace."""
    root = Path(evals_root).resolve(strict=True)
    source = (root / case).resolve(strict=True)
    if source == root or not source.is_relative_to(root) or not source.is_dir():
        raise ValueError("Select a case directory inside the evaluation root")

    cache = _workspace_root()
    if cache.is_relative_to(source) or source.is_relative_to(cache):
        raise ValueError("The evaluation case and workspace cache must be separate")
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    workspace = Path(mkdtemp(prefix="case-", dir=cache))
    try:
        shutil.copytree(source, workspace, dirs_exist_ok=True, ignore=_check_source_entries)
        # The container's unprivileged UID needs read access to the copied files.
        workspace.chmod(0o755)
        for path in workspace.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
    except Exception:
        shutil.rmtree(workspace)
        raise
    return workspace


def validate_workspace(workspace: str | Path) -> Path:
    """Require a disposable workspace directly under the home-directory cache."""
    path = Path(workspace).absolute()
    resolved = path.resolve(strict=True)
    if resolved != path or path.parent != _workspace_root() or not path.is_dir():
        raise ValueError("Expected a disposable workspace inside the home-directory cache")
    return path


def run_tests(workspace: str | Path, *, timeout: float = DEFAULT_TIMEOUT) -> TestResult:
    """Run the fixed pytest command in Docker and capture its combined output."""
    try:
        path = validate_workspace(workspace)
        if any(char in str(path) for char in ',\r\n"'):
            raise ValueError("Workspace path contains unsupported Docker mount characters")
    except (OSError, RuntimeError, ValueError) as exc:
        return TestResult(False, f"Invalid workspace: {exc}", 125)

    name = f"coding-agent-lab-{uuid4().hex}"
    command = [
        "docker", "run", "--rm", "--name", name, "--pull=never",
        "--network=none", "--user=10001:10001", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--memory=256m", "--memory-swap=256m",
        "--cpus=1", "--pids-limit=64", "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=32m",
        "--mount", f"type=bind,source={path},target=/workspace,readonly",
        "--workdir", "/workspace", "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        IMAGE, "python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except FileNotFoundError:
        return TestResult(False, "Docker executable not found", 127)
    except OSError as exc:
        return TestResult(False, f"Unable to start Docker: {exc}", 125)
    except subprocess.TimeoutExpired as exc:
        output = exc.output or b""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        message = f"Docker test execution timed out after {timeout:g} seconds"
        try:
            cleanup = subprocess.run(
                ["docker", "rm", "-f", name], capture_output=True,
                encoding="utf-8", errors="replace", timeout=5, check=False,
            )
            if cleanup.returncode:
                message += f"\nDocker cleanup reported: {cleanup.stderr.strip()}"
        except (OSError, subprocess.TimeoutExpired) as cleanup_error:
            message += f"\nCould not confirm Docker cleanup: {cleanup_error}"
        return TestResult(False, f"{message}\n{output}".rstrip(), 124)

    output = completed.stdout
    if completed.returncode < 0 or completed.returncode >= 125:
        output = f"Docker container execution failed (exit {completed.returncode})\n{output}"
    return TestResult(completed.returncode == 0, output, completed.returncode)
