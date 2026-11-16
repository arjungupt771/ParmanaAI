"""
ParmanaAI - Multi-Agent Base Interface

Abstract base class for all nodes in the multi-agent graph.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Dict, Optional

from backend.agents.multi_agent.state import SharedAgentState


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Standard interface for an autonomous agent node.
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def run(self, state: SharedAgentState) -> SharedAgentState:
        """
        Wrapper handling timing, state logging, and error containment.
        """
        if not self.enabled:
            state.log_event(self.name, "SKIPPED", {"reason": "Agent disabled"})
            return state

        start_time = time.perf_counter()
        try:
            state.log_event(self.name, "STARTED", {})
            state = self.execute(state)
            duration_ms = (time.perf_counter() - start_time) * 1000
            state.agent_latencies_ms[self.name] = round(duration_ms, 2)
            state.log_event(self.name, "COMPLETED", {"duration_ms": round(duration_ms, 2)})
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            state.agent_latencies_ms[self.name] = round(duration_ms, 2)
            logger.exception("Agent %s failed: %s", self.name, exc)
            state.errors.append({
                "agent": self.name,
                "error": str(exc),
                "type": type(exc).__name__,
            })
            state.log_event(self.name, "FAILED", {"error": str(exc)})
            state = self.handle_failure(state, exc)

        return state

    @abstractmethod
    def execute(self, state: SharedAgentState) -> SharedAgentState:
        """Core agent logic."""
        pass

    def handle_failure(self, state: SharedAgentState, error: Exception) -> SharedAgentState:
        """Graceful degradation logic on agent exception."""
        return state
