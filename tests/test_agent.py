"""Exercise the repair loop with scripted replies and in-memory tools."""

import pytest

from adapters.models.scripted import ScriptedClient
from core.models import RepairResult, TestResult
from core.repair_agent import RepairAgent


READ = '{"tool":"read_file","path":"code.py","summary":"Inspect code"}'
RUN = '{"tool":"run_tests"}'
EDIT = '{"tool":"edit_file","path":"code.py","content":"fixed"}'


class RecordingClient(ScriptedClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.histories = []

    def chat(self, messages):
        self.histories.append([message.copy() for message in messages])
        return super().chat(messages)


class FakeTools:
    def __init__(self, results=(), failures=()):
        self.results = iter(results)
        self.failures = set(failures)
        self.calls = []

    def read_file(self, path):
        self.calls.append(("read_file", path))
        if "read_file" in self.failures:
            raise FileNotFoundError(path)
        return "original"

    def edit_file(self, path, content):
        self.calls.append(("edit_file", path, content))
        if "edit_file" in self.failures:
            raise PermissionError(path)
        return "Updated code.py"

    def run_tests(self):
        self.calls.append(("run_tests",))
        if "run_tests" in self.failures:
            raise RuntimeError("runner unavailable")
        return next(self.results)


def attempt(replies, tools):
    client = RecordingClient(replies)
    result = RepairAgent(client, tools).repair("example", ["code.py"], "/workspace")
    assert isinstance(result, RepairResult)
    assert result.trace[0]["event"] == "repair_started"
    assert result.trace[-1]["event"] == "repair_finished"
    return result, client


def test_success_and_conversation_history():
    final = TestResult(True, "passed", 0)
    tools = FakeTools([TestResult(False, "failed", 1), final])
    result, client = attempt([READ, RUN, EDIT, RUN], tools)
    assert result.passed
    assert result.last_test_result is final
    assert result.changed_files == ["code.py"]
    assert result.workspace == "/workspace"
    assert [call[0] for call in tools.calls] == ["read_file", "run_tests", "edit_file", "run_tests"]
    assert {entry["event"] for entry in result.trace} == {
        "repair_started", "model_response", "tool_result", "repair_finished",
    }
    assert [message["role"] for message in client.histories[-1]] == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant", "user",
    ]
    assert client.histories[1][2]["content"] == READ
    assert "original" in client.histories[1][3]["content"]


@pytest.mark.parametrize("reply", [
    "invalid", '[]', '{}', RUN + RUN, "text " + RUN,
    '```json\n' + RUN + '\n```',
    '{"tool":"run_tests","tool":"read_file"}',
    '{"tool":"shell"}', '{"tool":"read_file"}',
    '{"tool":"run_tests","path":"code.py"}',
    '{"tool":"edit_file","path":"code.py","content":42}',
    '{"tool":"run_tests","summary":null}',
    '{"tool":"run_tests","summary":NaN}',
])
def test_invalid_response_calls_no_tools(reply):
    tools = FakeTools()
    result, client = attempt([reply, READ], tools)
    assert not result.passed
    assert tools.calls == []
    assert len(client.histories) == 1


def test_early_edit_is_rejected_then_repair_continues():
    tools = FakeTools([TestResult(False, "failed", 1), TestResult(True, "passed", 0)])
    result, client = attempt([READ, EDIT, RUN, EDIT, RUN], tools)
    assert result.passed
    assert [call[0] for call in tools.calls].count("edit_file") == 1
    assert any(entry.get("status") == "policy_error" for entry in result.trace)
    assert "Policy error" in client.histories[2][-1]["content"]


def test_budget_stops_after_eight_calls():
    result, client = attempt([READ] * 10, FakeTools())
    assert not result.passed
    assert len(client.histories) == 8
    assert "budget" in result.trace[-1]["reason"]


def test_eighth_decision_can_verify_repair():
    tools = FakeTools([TestResult(False, "failed", 1), TestResult(True, "passed", 0)])
    result, client = attempt([READ] * 5 + [RUN, EDIT, RUN], tools)
    assert result.passed
    assert len(client.histories) == 8


def test_passing_baseline_stops_without_claiming_repair():
    baseline = TestResult(True, "passed", 0)
    tools = FakeTools([baseline])
    result, client = attempt([RUN, READ], tools)
    assert not result.passed
    assert result.last_test_result is baseline
    assert len(client.histories) == 1
    assert "No reproducible failure" in result.trace[-1]["reason"]


def test_client_exhaustion_returns_failed_result():
    result, client = attempt([], FakeTools())
    assert not result.passed
    assert result.last_test_result is None
    assert "Client error" in result.trace[-1]["reason"]
    assert len(client.histories) == 1


@pytest.mark.parametrize("failed_tool", ["read_file", "run_tests", "edit_file"])
def test_tool_errors_are_visible_and_do_not_supply_success_evidence(failed_tool):
    tools = FakeTools([TestResult(False, "failed", 1)], [failed_tool])
    result, client = attempt([READ, RUN, EDIT, READ], tools)
    assert not result.passed
    assert result.changed_files == []
    assert any(entry.get("status") == "tool_error" for entry in result.trace)
    assert any("Tool error" in message["content"] for message in client.histories[-1])
    if failed_tool != "edit_file":
        assert not any(call[0] == "edit_file" for call in tools.calls)


def test_infrastructure_failure_does_not_authorize_edit():
    tools = FakeTools([TestResult(False, "Docker unavailable", 125)])
    result, _ = attempt([READ, RUN, EDIT], tools)
    assert not result.passed
    assert not any(call[0] == "edit_file" for call in tools.calls)


def test_reusing_agent_starts_a_fresh_attempt():
    tools = FakeTools([TestResult(False, "failed", 1), TestResult(True, "passed", 0)])
    client = ScriptedClient([READ, RUN, EDIT, RUN, EDIT])
    agent = RepairAgent(client, tools)
    first = agent.repair("first", ["code.py"], "/workspace")
    second = agent.repair("second", ["code.py"], "/workspace")
    assert first.passed and not second.passed
    assert second.changed_files == []
    assert second.last_test_result is None
    assert any(entry.get("status") == "policy_error" for entry in second.trace)


def test_live_progress_matches_stored_trace_and_preserves_decisions(capsys):
    from main import print_event, print_summary

    replies = [READ, EDIT, RUN, EDIT, RUN]
    outcomes = [TestResult(False, "first line\nsecond line", 1), TestResult(True, "passed", 0)]
    baseline, _ = attempt(replies, FakeTools(outcomes))
    tools = FakeTools(outcomes)
    observed = []

    def emit(event):
        # The callback runs between decisions, including before tool dispatch.
        if event["event"] == "repair_started":
            assert tools.calls == []
        if event["event"] == "model_response" and event["decision"] == 1:
            assert tools.calls == []
        observed.append(event.copy())
        print_event(event)
        assert capsys.readouterr().out
        event["event"] = "modified by observer"

    result = RepairAgent(ScriptedClient(replies), tools).repair(
        "example", ["code.py"], "/workspace", on_event=emit,
    )
    assert result == baseline
    assert observed == result.trace
    for event in observed:
        print_event(event)
    print_summary(result)
    output = capsys.readouterr().out
    stages = [line.split()[0] for line in output.splitlines()
              if line.startswith(("START ", "ACTION ", "RESULT ", "FINISHED "))]
    assert stages == ["START"] + ["ACTION", "RESULT"] * 5 + ["FINISHED"]
    assert "ACTION  read_file — Inspect code" in output
    assert "RESULT  edit_file — rejected" in output
    assert "    first line\n    second line" in output
    assert "    code.py" in output
    assert "Final sandboxed test status: passed" in output


def test_progress_handles_errors_and_failed_summary(capsys):
    from main import print_event, print_summary

    agent = RepairAgent(ScriptedClient([READ]), FakeTools(failures=["read_file"]))
    result = agent.repair("example", ["code.py"], "/workspace", on_event=print_event)
    print_summary(result)
    output = capsys.readouterr().out
    assert "RESULT  read_file — error" in output
    assert "    Tool error:" in output
    assert "FINISHED  failure" in output
    assert "No files changed" in output
    assert "Final sandboxed test status: not executed" in output
    assert "Reason: Client error:" in output


def test_broken_presentation_does_not_change_repair_result():
    replies = [READ, RUN, EDIT, RUN]
    outcomes = [TestResult(False, "failed", 1), TestResult(True, "passed", 0)]
    baseline, _ = attempt(replies, FakeTools(outcomes))

    def broken_callback(event):
        raise RuntimeError("display unavailable")

    result = RepairAgent(ScriptedClient(replies), FakeTools(outcomes)).repair(
        "example", ["code.py"], "/workspace", on_event=broken_callback,
    )
    assert result == baseline
