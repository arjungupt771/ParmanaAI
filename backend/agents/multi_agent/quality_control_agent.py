"""
ParmanaAI - Multi-Agent Quality Control Agent Node

Validates agent output contracts, checks for contradictory inter-agent findings,
monitors latencies and token budgets, and ensures response integrity.
"""

from typing import Any, Dict

from backend.agents.multi_agent.agent_base import BaseAgent
from backend.agents.multi_agent.state import SharedAgentState


class QualityControlAgent(BaseAgent):
    """
    Supervises system consistency, schema validity, and contradiction checks.
    """

    def __init__(self, name: str = "QualityControlAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    def execute(self, state: SharedAgentState) -> SharedAgentState:
        checks: Dict[str, bool] = {}

        # 1. Schema integrity checks
        checks["has_answer"] = bool(state.generated_answer and state.generated_answer.strip())
        checks["has_valid_hallucination_score"] = (0.0 <= state.hallucination_score <= 1.0)
        checks["has_valid_relevance_score"] = (0.0 <= state.relevance_score <= 1.0)

        # 2. Contradiction Detection between Generation and Verification
        contradicted_claims = [
            c for c in state.verified_claims
            if c.get("combined_verdict") == "CONTRADICTED" or c.get("verdict") == "CONTRADICTED"
        ]
        has_internal_contradiction = len(contradicted_claims) > 0
        checks["no_severe_contradictions"] = not has_internal_contradiction

        if has_internal_contradiction and not state.is_refusal:
            # Append transparent clinical warning notice if any generated claim is contradicted by evidence
            state.log_event(
                self.name,
                "CONTRADICTION_FLAGGED",
                {"contradicted_claims_count": len(contradicted_claims)}
            )
            # Add advisory notice if not already present
            notice = "\n\n> Note: Verification detected that one or more claims in this answer contradict the retrieved clinical evidence."
            if notice not in state.generated_answer:
                state.generated_answer += notice

        # 3. Latency & Resource Accounting
        total_pipeline_time_ms = sum(state.agent_latencies_ms.values())
        state.agent_latencies_ms["total_pipeline_ms"] = round(total_pipeline_time_ms, 2)
        checks["within_latency_budget"] = total_pipeline_time_ms < 15000.0

        state.quality_checks = checks
        return state
