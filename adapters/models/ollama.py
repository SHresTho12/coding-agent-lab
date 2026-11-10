"""Chat client for the local Ollama HTTP API."""

import json
import math
import os
from http.client import HTTPConnection
from typing import Any


class OllamaClient:
    """Return reply text from a locally installed Ollama model."""

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL") or "qwen2.5-coder:7b"
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and non-negative")
        self.temperature = temperature

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Send prepared messages and return content without interpreting the action."""
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        })
        connection = HTTPConnection("127.0.0.1", 11434, timeout=120)
        try:
            connection.request("POST", "/api/chat", body.encode("utf-8"),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                raise RuntimeError(
                    f"Ollama returned HTTP {response.status}: "
                    f"{payload.decode('utf-8', errors='replace')}"
                )
            try:
                envelope = json.loads(payload)
                content = envelope["message"]["content"]
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError("Ollama returned an invalid chat response") from exc
            if not isinstance(content, str):
                raise RuntimeError("Ollama reply content must be text")
            return content
        finally:
            connection.close()
