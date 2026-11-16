"""
ParmanaAI - Evidence Analysis Agent Node

Performs evidence quality assessment, deduplication, cross-encoder reranking,
source diversity enforcement, and topic mismatch detection.
"""

from typing import Any, Dict, List, Optional

from backend.agents.multi_agent.agent_base import BaseAgent
from backend.agents.multi_agent.state import SharedAgentState
from backend.reranking.evidence_selector import EvidenceSelector
from backend.reranking.base import RerankConfig


class EvidenceAnalysisAgent(BaseAgent):
    """
    Reranks candidate evidence and enforces quality and source diversity.
    """

    def __init__(
        self,
        selector: Optional[EvidenceSelector] = None,
        name: str = "EvidenceAnalysisAgent",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled)
        self.selector = selector or EvidenceSelector(config=RerankConfig(max_evidence_budget=3))

    def execute(self, state: SharedAgentState) -> SharedAgentState:
        if state.is_emergency or not state.retrieved_candidates:
            state.selected_evidence = []
            state.abstained = True
            state.abstention_reason = (
                "Emergency query escalated to medical emergency protocol."
                if state.is_emergency
                else "No candidate evidence retrieved."
            )
            return state

        outcome = self.selector.select(
            query=state.user_query,
            candidates=state.retrieved_candidates,
        )

        state.selected_evidence = outcome.selected_evidence
        state.rejected_evidence = outcome.rejected_evidence
        state.evidence_diagnostics = outcome.selection_diagnostics
        state.abstained = outcome.abstained
        state.abstention_reason = outcome.abstention_reason

        return state
