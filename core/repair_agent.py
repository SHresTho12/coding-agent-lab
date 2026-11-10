"""Coordinate a bounded repair attempt through the core interfaces."""

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .models import ModelAction, RepairResult, TestResult
from .ports import ChatClient, RepairToolsPort, ResearchPort, TraceRecorder
from .prompts import repair_request, system_prompt, tool_result_message


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys")
        result[key] = value
    return result


def _parse_action(raw: str, research_enabled: bool = False) -> ModelAction:
    data = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or not all(isinstance(value, str) for value in data.values()):
        raise ValueError("Expected one JSON object with string values")
    fields = {
        "read_file": {"tool", "path"},
        "run_tests": {"tool"},
        "edit_file": {"tool", "path", "content"},
    }
    if research_enabled:
        fields["search_github"] = {"tool", "query"}
    required = fields.get(data.get("tool"))
    if required is None or not required <= data.keys() or data.keys() - required - {"summary"}:
        raise ValueError("Unknown tool, missing required fields, or unexpected fields")
    return ModelAction(
        data["tool"],
        {key: data[key] for key in required - {"tool"}},
        data.get("summary"),
    )


class RepairAgent:
    """Allow up to eight model decisions to reproduce, edit, and verify a repair."""

    def __init__(self, client: ChatClient, tools: RepairToolsPort, research: ResearchPort | None = None, recorder: TraceRecorder | None = None) -> None:
        self.client = client
        self.tools = tools
        self.research = research
        self.recorder = recorder

    def repair(
        self, case_name: str, files: list[str], workspace: str, *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> RepairResult:
        """Run one independent attempt and return its outcome and public trace."""
        messages = [
            {"role": "system", "content": system_prompt(self.research is not None)},
            {"role": "user", "content": repair_request(case_name, files)},
        ]
        trace: list[dict[str, Any]] = []

        def record(event: dict[str, Any]) -> None:
            """Store an event and notify presentation without affecting the attempt."""
            trace.append(event)
            if self.recorder is not None:
                public = {key: event[key] for key in ("case", "tool", "summary", "passed") if key in event}
                if event["event"] in {"model_response", "tool_result"}:
                    public["step"] = decision
                if event["event"] == "tool_result":
                    public["outcome"] = event["status"]
                    if event["tool"] == "run_tests" and event["status"] == "ok":
                        public["passed"] = last_test_result.passed
                        public["returncode"] = last_test_result.returncode
                if event["event"] == "repair_finished":
                    public["changed_files"] = changed_files.copy()
                    public["outcome"] = "success" if passed else "failure"
                try:
                    self.recorder.record(event["event"], public)
                except Exception:
                    logging.getLogger(__name__).warning("Trace recorder failed; repair will continue")
            if on_event is not None:
                try:
                    on_event(event.copy())
                except Exception:
                    # A presentation failure must not change repair decisions.
                    pass

        record({"event": "repair_started", "case": case_name})
        changed_files: list[str] = []
        last_test_result: TestResult | None = None
        read_succeeded = False
        failing_test_seen = False
        research_requests = 0
        edited = False
        passed = False
        reason = "Decision budget exhausted"

        for decision in range(1, 9):
            try:
                raw = self.client.chat(messages)
            except Exception as exc:
                reason = f"Client error: {exc}"
                break
            try:
                action = _parse_action(raw, self.research is not None)
            except (ValueError, TypeError, RecursionError):
                record({"event": "model_response", "decision": decision, "valid": False})
                reason = "Invalid model response"
                break

            messages.append({"role": "assistant", "content": raw})
            record({
                "event": "model_response", "decision": decision,
                "tool": action.tool, "summary": action.public_summary, "valid": True,
            })
            status = "ok"
            if action.tool == "edit_file" and not (read_succeeded and failing_test_seen):
                status = "policy_error"
                observation = "Policy error: editing requires a successful read and a failing test."
            elif action.tool == "search_github" and not (read_succeeded and failing_test_seen):
                status = "policy_error"
                observation = "Policy error: research requires a successful local read and a failing test."
            else:
                try:
                    if action.tool == "read_file":
                        observation = self.tools.read_file(action.arguments["path"])
                        read_succeeded = True
                    elif action.tool == "edit_file":
                        observation = self.tools.edit_file(**action.arguments)
                        edited = True
                        path = action.arguments["path"]
                        if path not in changed_files:
                            changed_files.append(path)
                    elif action.tool == "search_github":
                        query = action.arguments["query"]
                        if not query.strip() or len(query) > 180 or len(query.splitlines()) != 1 or "\n" in query or "\r" in query:
                            raise ValueError("Research query must be one nonempty line of at most 180 characters")
                        if research_requests >= 3:
                            raise ValueError("Research request budget exhausted")
                        research_requests += 1
                        observation = json.dumps(self.research.search_github(query))
                    else:
                        last_test_result = self.tools.run_tests()
                        observation = json.dumps(asdict(last_test_result))
                        # Pytest exit 1 represents test failures, unlike runner errors.
                        if not last_test_result.passed and last_test_result.returncode == 1:
                            failing_test_seen = True
                        if last_test_result.passed and last_test_result.returncode == 0:
                            passed = edited
                            reason = "Repair verified" if passed else "No reproducible failure; tests passed before an edit"
                except Exception as exc:
                    status = "tool_error"
                    observation = f"Tool error: {exc}"
            record({
                "event": "tool_result", "tool": action.tool,
                "status": status, "output": observation,
            })
            messages.append({"role": "user", "content": tool_result_message(action.tool, observation)})
            if status == "ok" and action.tool == "run_tests" and last_test_result.passed and last_test_result.returncode == 0:
                break

        record({"event": "repair_finished", "passed": passed, "reason": reason})
        return RepairResult(passed, trace, changed_files, workspace, last_test_result)
