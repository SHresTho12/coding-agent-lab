"""Verify observability with fake clients and in-memory repair tools."""

from unittest.mock import MagicMock, call, patch

import pytest

from adapters.models.scripted import ScriptedClient
from adapters.observability.langfuse import LangfuseTrajectoryRecorder
from core.models import TestResult
from core.repair_agent import RepairAgent


def repair(recorder=None):
    tools = MagicMock()
    tools.read_file.return_value = "PRIVATE SOURCE"
    tools.edit_file.return_value = "Updated code.py"
    tools.run_tests.side_effect = [
        TestResult(False, "PRIVATE TEST OUTPUT", 1), TestResult(True, "PRIVATE TEST OUTPUT", 0),
    ]
    client = ScriptedClient([
        '{"tool":"read_file","path":"code.py","summary":"Inspect source"}',
        '{"tool":"run_tests"}',
        '{"tool":"edit_file","path":"code.py","content":"PRIVATE REPLACEMENT"}',
        '{"tool":"run_tests"}',
    ])
    return RepairAgent(client, tools, recorder=recorder).repair("example", ["code.py"], "/workspace")


def test_event_order_payloads_and_flush():
    client = MagicMock()
    recorder = LangfuseTrajectoryRecorder("public", "secret", "https://example.test", client=client)
    with patch.object(recorder, "record", wraps=recorder.record) as record:
        result = repair(recorder)
    assert result == repair()
    assert [call.args[0] for call in record.call_args_list] == [event["event"] for event in result.trace]
    calls = client.mock_calls
    assert calls[0].kwargs["as_type"] == "agent"
    root = client.start_observation.return_value
    assert root.start_observation.call_count == 8
    assert root.start_observation.return_value.end.call_count == 8
    assert calls[-2] == call.start_observation().end()
    assert calls[-1] == call.flush()
    assert root.update.call_args.kwargs["output"]["changed_files"] == ["code.py"]
    exported = str(calls)
    for forbidden in ("PRIVATE SOURCE", "PRIVATE TEST OUTPUT", "PRIVATE REPLACEMENT", "secret"):
        assert forbidden not in exported
    assert "'returncode': 1" in exported


@pytest.mark.parametrize("failure", ["start", "child", "update", "end", "flush"])
def test_sdk_failures_preserve_result_and_flush(failure, caplog):
    client = MagicMock()
    root = client.start_observation.return_value
    operations = {
        "start": client.start_observation, "child": root.start_observation,
        "update": root.update, "end": root.end, "flush": client.flush,
    }
    operations[failure].side_effect = RuntimeError("SECRET MUST NOT BE LOGGED")
    recorder = LangfuseTrajectoryRecorder("public", "secret", "https://example.test", client=client)
    assert repair(recorder) == repair()
    client.flush.assert_called_once()
    assert "repair will continue" in caplog.text
    assert "SECRET MUST NOT BE LOGGED" not in caplog.text


def test_constructor_failure_is_best_effort(caplog):
    with patch("adapters.observability.langfuse.Langfuse", side_effect=RuntimeError("secret")):
        recorder = LangfuseTrajectoryRecorder("public", "secret", "https://example.test")
    assert repair(recorder) == repair()
    assert "initialization failed" in caplog.text


def test_credentials_loaded_and_validated_before_workspace_creation(tmp_path, monkeypatch):
    import bootstrap
    monkeypatch.setattr(bootstrap, "__file__", str(tmp_path / "bootstrap.py"))
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with patch("bootstrap.create_workspace") as workspace:
        with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
            bootstrap.build_service(observe=True)
        workspace.assert_not_called()
    (tmp_path / ".env").write_text(
        'LANGFUSE_PUBLIC_KEY="public"\nLANGFUSE_SECRET_KEY="secret"\n'
        'LANGFUSE_BASE_URL="https://example.test"\n'
    )
    with patch("adapters.observability.langfuse.LangfuseTrajectoryRecorder") as recorder, patch("bootstrap.RepairTools"):
        bootstrap.build_service("ollama").create_agent("example", tmp_path)
        recorder.assert_not_called()
        bootstrap.build_service("ollama", observe=True).create_agent("example", tmp_path)
        recorder.assert_called_once_with("public", "secret", "https://example.test")
