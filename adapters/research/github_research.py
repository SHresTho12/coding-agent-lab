"""Bounded public evidence from GitHub code search."""

import json
from datetime import datetime, timezone
from http.client import HTTPSConnection
from urllib.parse import urlencode, urlsplit


class GitHubResearch:
    """Use one instance per repair to limit requests and returned sources."""

    def __init__(self, token: str) -> None:
        if not token or any(character in token for character in "\r\n"):
            raise ValueError("A valid GITHUB_TOKEN is required for research")
        self._token = token
        self._requests = 0
        self._results = 0

    def search_github(self, query: str) -> dict:
        """Return up to three public source snippets across this repair."""
        if not isinstance(query, str) or not query.strip() or len(query) > 180 or len(query.splitlines()) != 1 or "\r" in query or "\n" in query:
            raise ValueError("Research query must be one nonempty line of at most 180 characters")
        if self._requests >= 3:
            raise ValueError("Research request budget exhausted")
        evidence = {"query": query, "retrieved_at": datetime.now(timezone.utc).isoformat(), "sources": []}
        if self._results >= 3:
            return evidence
        self._requests += 1
        connection = HTTPSConnection("api.github.com", timeout=20)
        try:
            connection.request(
                "GET", "/search/code?" + urlencode({"q": query, "per_page": 3}),
                headers={"Authorization": f"Bearer {self._token}",
                         "Accept": "application/vnd.github.text-match+json",
                         "User-Agent": "coding-agent-lab"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"GitHub code search failed (HTTP {response.status})")
            payload = response.read(1_000_001)
            if len(payload) > 1_000_000:
                raise ValueError("Oversized response")
            items = json.loads(payload)["items"]
            for item in items:
                repository = item["repository"]
                if repository.get("private") is not False:
                    continue
                url = item["html_url"]
                parsed = urlsplit(url)
                if parsed.scheme != "https" or parsed.netloc != "github.com":
                    continue
                snippet = "\n".join(match.get("fragment", "") for match in item.get("text_matches", []))[:500]
                source = {"repository": repository["full_name"], "path": item["path"],
                          "url": url, "snippet": snippet}
                if not all(isinstance(value, str) for value in source.values()):
                    continue
                # Never expose credentials, including unexpected server echoes.
                source = {key: value.replace(self._token, "[redacted]") for key, value in source.items()}
                evidence["sources"].append(source)
                self._results += 1
                if self._results == 3:
                    break
            return evidence
        except Exception:
            raise RuntimeError("GitHub code search could not retrieve public evidence") from None
        finally:
            connection.close()
