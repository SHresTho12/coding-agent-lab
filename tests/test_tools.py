"""Test sandbox boundaries with synthetic files and mocked Docker calls."""

import os
import subprocess

import pytest

from adapters.sandbox import runner
from adapters.sandbox.repair_tools import RepairTools, safe_path
from core.models import TestResult


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    source = tmp_path / "evals" / "example"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "sample.txt").write_text("café", encoding="utf-8")
    return source


@pytest.fixture
def workspace(case):
    return runner.create_workspace(case.name, case.parent)


def test_create_workspace_copies_case_into_unique_home_cache_dirs(case, workspace, tmp_path):
    other = runner.create_workspace(case, case.parent)
    assert workspace != other
    assert workspace.parent == other.parent == tmp_path / ".cache" / "coding-agent-lab"
    assert (other / "nested" / "sample.txt").read_bytes() == (
        case / "nested" / "sample.txt"
    ).read_bytes()


def test_workspace_creation_rejects_source_outside_evals(case, tmp_path):
    sibling = tmp_path / "evals-other" / "example"
    sibling.mkdir(parents=True)
    for path in (sibling, "../evals-other/example", case.parent):
        with pytest.raises(ValueError, match="inside the evaluation root"):
            runner.create_workspace(path, case.parent)


def make_symlink(link, target):
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")


def test_workspace_creation_rejects_source_symlink(case, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    make_symlink(case / "link.txt", outside)
    with pytest.raises(ValueError, match="Unsafe evaluation case entry"):
        runner.create_workspace(case, case.parent)
    assert outside.read_text(encoding="utf-8") == "secret"


def test_reads_utf8_nested_file_and_rejects_missing_file(workspace):
    tools = RepairTools(workspace)
    assert safe_path(workspace, "nested/sample.txt") == workspace / "nested" / "sample.txt"
    assert tools.read_file("nested/sample.txt") == "café"
    with pytest.raises(FileNotFoundError):
        tools.read_file("missing.txt")


def test_absolute_paths_and_directories_are_rejected(workspace):
    tools = RepairTools(workspace)
    with pytest.raises(ValueError, match="workspace-relative"):
        tools.read_file(str(workspace / "nested" / "sample.txt"))
    with pytest.raises(IsADirectoryError):
        tools.read_file("nested")


def test_traversal_to_sibling_with_shared_prefix_is_rejected(workspace):
    sibling = workspace.with_name(f"{workspace.name}-other")
    sibling.mkdir()
    secret = sibling / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    relative = f"../{sibling.name}/secret.txt"
    tools = RepairTools(workspace)
    with pytest.raises(ValueError, match="escapes"):
        tools.read_file(relative)
    with pytest.raises(ValueError, match="escapes"):
        tools.edit_file(relative, "overwrite")
    assert secret.read_text(encoding="utf-8") == "secret"
    assert tools.changed_files == []


def test_external_symlink_is_rejected(case, workspace):
    original = case / "nested" / "sample.txt"
    make_symlink(workspace / "link.txt", original)
    tools = RepairTools(workspace)
    with pytest.raises(ValueError, match="escapes"):
        tools.read_file("link.txt")
    with pytest.raises(ValueError, match="escapes"):
        tools.edit_file("link.txt", "overwrite")
    assert original.read_text(encoding="utf-8") == "café"


def test_hard_link_cannot_expose_or_change_external_file(case, workspace):
    original = case / "nested" / "sample.txt"
    try:
        os.link(original, workspace / "link.txt")
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Hard links unavailable: {exc}")
    tools = RepairTools(workspace)
    with pytest.raises(ValueError, match="hard links"):
        tools.read_file("link.txt")
    tools.edit_file("link.txt", "updated")
    assert tools.read_file("link.txt") == "updated"
    assert original.read_text(encoding="utf-8") == "café"


def test_edit_replaces_workspace_copy_and_records_canonical_path_once(case, workspace):
    tools = RepairTools(workspace)
    assert tools.edit_file("nested/sample.txt", "réparé") == "Updated nested/sample.txt"
    tools.edit_file("nested/../nested/sample.txt", "réparé deux fois")
    assert tools.read_file("nested/sample.txt") == "réparé deux fois"
    assert (case / "nested" / "sample.txt").read_text(encoding="utf-8") == "café"
    assert tools.changed_files == ["nested/sample.txt"]


def test_edit_can_create_file_but_not_parent_directories(workspace):
    tools = RepairTools(workspace)
    tools.edit_file("nested/new.txt", "new")
    assert tools.read_file("nested/new.txt") == "new"
    with pytest.raises(FileNotFoundError):
        tools.edit_file("missing/new.txt", "new")
    assert tools.changed_files == ["nested/new.txt"]


def test_repair_tools_delegates_tests(workspace, monkeypatch):
    expected = TestResult(True, "1 passed", 0)

    def run(workspace_arg):
        assert workspace_arg == workspace
        return expected

    monkeypatch.setattr(runner, "run_tests", run)
    assert RepairTools(workspace).run_tests() is expected


@pytest.mark.parametrize(
    ("code", "output"),
    [(0, "1 passed"), (1, "1 failed"), (125, "Cannot connect to the Docker daemon"),
     (125, "No such image: coding-agent-lab-runner"), (127, "python not found")],
)
def test_docker_command_and_exit_status(case, workspace, monkeypatch, code, output):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs.get("shell", False) is False
        assert kwargs["timeout"] == 3
        assert kwargs["stderr"] == subprocess.STDOUT
        return subprocess.CompletedProcess(command, code, stdout=output)

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.run_tests(workspace, timeout=3)
    assert result.passed is (code == 0)
    assert result.returncode == code
    assert output in result.output
    assert ("Docker container execution failed" in result.output) is (code >= 125)

    command, = calls
    assert isinstance(command, list)
    assert command[:2] == ["docker", "run"]
    assert {
        "--rm", "--network=none", "--user=10001:10001", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--read-only", "--memory=256m",
        "--cpus=1", "--pids-limit=64", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
    }.issubset(command)
    assert command.count("--mount") == 1
    assert command[command.index("--mount") + 1] == (
        f"type=bind,source={workspace},target=/workspace,readonly"
    )
    assert str(case) not in " ".join(command)
    assert command[-8:] == [
        runner.IMAGE, "python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
    ]


def test_timeout_attempts_to_remove_named_container(workspace, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial output")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.run_tests(workspace, timeout=1)
    assert result == TestResult(False, "Docker test execution timed out after 1 seconds\npartial output", 124)
    start, cleanup = calls
    assert cleanup == ["docker", "rm", "-f", start[start.index("--name") + 1]]


def test_missing_docker_reports_failure(workspace, monkeypatch):
    def run(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.run_tests(workspace) == TestResult(False, "Docker executable not found", 127)


def test_original_case_cannot_be_mounted_or_edited(case, monkeypatch):
    def run(*args, **kwargs):
        pytest.fail("Docker must not start for an invalid workspace")

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.run_tests(case)
    assert not result.passed
    assert "Invalid workspace" in result.output
    with pytest.raises(ValueError, match="disposable workspace"):
        RepairTools(case)
