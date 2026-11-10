"""Verify optional research without network access."""

import json
from unittest.mock import MagicMock, patch

import pytest

from adapters.models.scripted import ScriptedClient
from adapters.research.github_research import GitHubResearch
from core.models import TestResult
from core.repair_agent import RepairAgent

READ = '{"tool":"read_file","path":"code.py"}'
RUN = '{"tool":"run_tests"}'
SEARCH = '{"tool":"search_github","query":"Python median even length"}'
URL = "https://github.com/example/public/blob/main/code.py"


def run(replies, research):
    tools = MagicMock()
    tools.read_file.return_value = "source"
    tools.run_tests.return_value = TestResult(False, "failed", 1)
    return RepairAgent(ScriptedClient(replies), tools, research).repair("case", ["code.py"], "/workspace")


def test_disabled_and_early_research():
    research = MagicMock()
    disabled = run([SEARCH], None)
    assert disabled.trace[-1]["reason"] == "Invalid model response"
    early = run([SEARCH], research)
    research.search_github.assert_not_called()
    assert any(event.get("status") == "policy_error" for event in early.trace)


def test_allowed_url_and_request_budget():
    research = MagicMock()
    research.search_github.return_value = {"sources": [{"url": URL}]}
    result = run([READ, RUN] + [SEARCH] * 4, research)
    assert research.search_github.call_count == 3
    assert any(URL in event.get("output", "") for event in result.trace)
    assert any("budget" in event.get("output", "") for event in result.trace)
    assert not result.passed


@pytest.mark.parametrize("query", ["", " ", "x\ny", "x\r", "x" * 181])
def test_bad_queries_never_call_network_or_port(query):
    with patch("adapters.research.github_research.HTTPSConnection") as connection:
        with pytest.raises(ValueError):
            GitHubResearch("secret").search_github(query)
        connection.assert_not_called()
    research = MagicMock()
    run([READ, RUN, json.dumps({"tool": "search_github", "query": query})], research)
    research.search_github.assert_not_called()


def test_public_evidence_and_total_result_budget():
    public = {"repository": {"private": False, "full_name": "example/public"},
              "path": "code.py", "html_url": URL,
              "text_matches": [{"fragment": "x" * 600}]}
    private = dict(public, repository={"private": True, "full_name": "private/repo"})
    with patch("adapters.research.github_research.HTTPSConnection") as factory:
        response = factory.return_value.getresponse.return_value
        response.status = 200
        response.read.return_value = json.dumps({"items": [private, public, public, public]}).encode()
        adapter = GitHubResearch("secret")
        result = adapter.search_github("median")
        assert len(result["sources"]) == 3
        assert all(source["url"] == URL and len(source["snippet"]) == 500 for source in result["sources"])
        assert "retrieved_at" in result
        assert adapter.search_github("another")["sources"] == []
        assert factory.call_count == 1
        assert factory.return_value.request.call_args.args[0] == "GET"


def test_failed_requests_count_and_do_not_leak_token():
    with patch("adapters.research.github_research.HTTPSConnection") as factory:
        factory.return_value.request.side_effect = RuntimeError("secret")
        adapter = GitHubResearch("secret")
        for _ in range(3):
            with pytest.raises(RuntimeError) as error:
                adapter.search_github("median")
            assert "secret" not in str(error.value)
        with pytest.raises(ValueError, match="budget"):
            adapter.search_github("median")
        assert factory.call_count == 3


def test_composition_loads_dotenv_only_when_enabled(tmp_path, monkeypatch):
    import bootstrap
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(bootstrap, "__file__", str(tmp_path / "bootstrap.py"))
    (tmp_path / ".env").write_text('GITHUB_TOKEN="local-test-token"\n')
    with patch("adapters.research.github_research.GitHubResearch") as adapter, patch("bootstrap.RepairTools"):
        bootstrap.build_service("ollama").create_agent("median", tmp_path)
        adapter.assert_not_called()
        bootstrap.build_service("ollama", research=True).create_agent("median", tmp_path)
        adapter.assert_called_once_with("local-test-token")


def test_terminal_reports_disabled_and_unused_research(capsys):
    from main import print_summary

    result = run([READ], None)
    print_summary(result)
    assert "GitHub research: disabled" in capsys.readouterr().out
    print_summary(result, research_enabled=True)
    assert "enabled but unused" in capsys.readouterr().out


def test_terminal_reports_search_outcomes_and_source_urls(capsys):
    from main import print_event, print_summary

    research = MagicMock()
    research.search_github.return_value = {
        "query": "Python median even length", "retrieved_at": "2026-11-10",
        "sources": [{"repository": "example/public", "path": "code.py", "url": URL, "snippet": "example"}],
    }
    result = run([SEARCH, READ, RUN, SEARCH], research)
    for event in result.trace:
        print_event(event)
    print_summary(result, research_enabled=True)
    output = capsys.readouterr().out
    assert "Query: Python median even length" in output
    assert URL in output
    assert "1 searches completed, 1 sources returned, 1 rejected, 0 errors" in output


def test_terminal_does_not_count_errors_as_completed_searches(capsys):
    from main import print_summary

    research = MagicMock()
    research.search_github.side_effect = RuntimeError("Search unavailable")
    result = run([READ, RUN, SEARCH], research)
    print_summary(result, research_enabled=True)
    assert "0 searches completed, 0 sources returned, 0 rejected, 1 errors" in capsys.readouterr().out
