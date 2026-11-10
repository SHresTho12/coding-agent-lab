"""Command-line entry point for repairing supplied evaluation cases."""

import argparse
import json
import sys

from bootstrap import build_service


def print_event(event: dict) -> None:
    """Render a public trace event immediately for the terminal."""
    kind = event["event"]
    if kind == "repair_started":
        print(f"START  {event['case']}", flush=True)
    elif kind == "model_response":
        if not event.get("valid", False):
            print("ACTION  Invalid model response rejected", flush=True)
            return
        summary = " ".join((event.get("summary") or "").split())
        label = event["tool"]
        if summary:
            label += f" — {summary}"
        print(f"ACTION  {label}", flush=True)
    elif kind == "tool_result":
        status = {
            "ok": "completed", "policy_error": "rejected", "tool_error": "error",
        }[event["status"]]
        output = event["output"]
        if event["tool"] == "run_tests" and event["status"] == "ok":
            test = json.loads(output)
            status = "passed" if test["passed"] else "failed"
            status += f" (exit {test['returncode']})"
            output = test["output"]
        print(f"RESULT  {event['tool']} — {status}", flush=True)
        for line in output.splitlines():
            print(f"    {line}", flush=True)
    elif kind == "repair_finished":
        status = "success" if event["passed"] else "failure"
        print(f"FINISHED  {status} — {event['reason']}", flush=True)


def print_summary(result) -> None:
    """Summarize the existing repair result without interpreting model replies."""
    print(f"\nOverall repair status: {'success' if result.passed else 'failure'}")
    print("Changed files")
    for path in result.changed_files:
        print(f"    {path}")
    if not result.changed_files:
        print("    No files changed")
    test = result.last_test_result
    status = "not executed" if test is None else ("passed" if test.passed else "failed")
    print(f"Final sandboxed test status: {status}")
    if not result.passed:
        print(f"Reason: {result.trace[-1]['reason']}")
    print(f"Workspace: {result.workspace}")


def main(argv: list[str] | None = None) -> int:
    """Run the selected repair and return an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    repair = commands.add_parser("repair")
    repair.add_argument("project")
    repair.add_argument("--provider", choices=["scripted", "ollama"], default="scripted")
    repair.add_argument("--model", help="Ollama model (defaults to OLLAMA_MODEL or qwen2.5-coder:7b)")
    repair.add_argument("--temperature", type=float, default=0.0)
    commands.add_parser("ui", help="Open the scripted repair page")
    args = parser.parse_args(argv)

    if args.command == "ui":
        from adapters.gradio_ui import create_demo

        create_demo(build_service("scripted")).launch(
            server_name="127.0.0.1", share=False, inbrowser=False,
        )
        return 0

    try:
        result = build_service(
            args.provider, print_event, model=args.model, temperature=args.temperature,
        ).repair(args.project)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Repair could not start: {exc}", file=sys.stderr)
        return 1
    print_summary(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
