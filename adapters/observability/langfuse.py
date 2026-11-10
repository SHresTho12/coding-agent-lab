"""Best-effort public repair trajectories for Langfuse Cloud."""

import logging

from langfuse import Langfuse

logger = logging.getLogger(__name__)


class LangfuseTrajectoryRecorder:
    """Record one agent observation with closed model and tool children."""

    def __init__(self, public_key: str, secret_key: str, base_url: str, *, client=None) -> None:
        self._client = client
        self._root = None
        if self._client is None:
            try:
                self._client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)
            except Exception:
                logger.warning("Langfuse initialization failed; repair will continue")

    def record(self, event: str, payload: dict) -> None:
        """Export only approved public fields and contain all SDK failures."""
        if self._client is None:
            return
        allowed = {"case", "step", "tool", "summary", "passed", "returncode", "changed_files", "outcome"}
        public = {key: value for key, value in payload.items() if key in allowed}
        try:
            if event == "repair_started":
                self._root = self._client.start_observation(
                    name="repair", as_type="agent", metadata=public,
                )
            elif event in {"model_response", "tool_result"} and self._root is not None:
                child = self._root.start_observation(
                    name=event, as_type="tool" if event == "tool_result" else "span",
                    metadata=public,
                )
                child.end()
            elif event == "repair_finished" and self._root is not None:
                try:
                    self._root.update(output=public)
                finally:
                    self._root.end()
        except Exception:
            logger.warning("Langfuse recording failed for %s; repair will continue", event)
        finally:
            if event == "repair_finished":
                self._root = None
                try:
                    self._client.flush()
                except Exception:
                    logger.warning("Langfuse flush failed; repair will continue")
