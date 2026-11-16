"""
ParmanaAI - Centralized Multi-Agent Orchestration Graph

Coordinates the execution graph across all specialized clinical agents:
QueryUnderstanding -> Retrieval -> EvidenceAnalysis -> Generation -> Verification -> QualityControl.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.agents.multi_agent.state import SharedAgentState
from backend.agents.multi_agent.query_understanding_agent import QueryUnderstandingAgent
from backend.agents.multi_agent.retrieval_agent import RetrievalAgent
from backend.agents.multi_agent.evidence_analysis_agent import EvidenceAnalysisAgent
from backend.agents.multi_agent.generation_agent_node import GenerationAgentNode
from backend.agents.multi_agent.verification_agent_node import VerificationAgentNode
from backend.agents.multi_agent.quality_control_agent import QualityControlAgent


logger = logging.getLogger(__name__)


class OrchestratorGraph:
    """
    Centralized execution graph orchestrator.
    """

    def __init__(
        self,
        query_agent: Optional[QueryUnderstandingAgent] = None,
        retrieval_agent: Optional[RetrievalAgent] = None,
        analysis_agent: Optional[EvidenceAnalysisAgent] = None,
        generation_agent: Optional[GenerationAgentNode] = None,
        verification_agent: Optional[VerificationAgentNode] = None,
        qc_agent: Optional[QualityControlAgent] = None,
    ):
        self.query_agent = query_agent or QueryUnderstandingAgent()
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.analysis_agent = analysis_agent or EvidenceAnalysisAgent()
        self.generation_agent = generation_agent or GenerationAgentNode()
        self.verification_agent = verification_agent or VerificationAgentNode()
        self.qc_agent = qc_agent or QualityControlAgent()

    def run(
        self,
        user_query: str,
        external_context: Optional[str] = None,
    ) -> SharedAgentState:
        """
        Execute the deterministic multi-agent workflow graph.
        """
        state = SharedAgentState(
            user_query=user_query,
            external_context=external_context,
        )

        # 1. Query Understanding Agent
        state = self.query_agent.run(state)

        # 2. Retrieval Agent
        state = self.retrieval_agent.run(state)

        # 3. Evidence Analysis & Reranking Agent
        state = self.analysis_agent.run(state)

        # 4. Generation Agent
        state = self.generation_agent.run(state)

        # 5. Verification Agent
        state = self.verification_agent.run(state)

        # 6. Quality Control Agent
        state = self.qc_agent.run(state)

        return state

    def run_to_dict(
        self,
        user_query: str,
        external_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute and return backward-compatible dictionary for API and frontend.
        """
        state = self.run(user_query, external_context=external_context)

        # Serialize evidence for frontend
        serialized_evidence = []
        for e in state.selected_evidence:
            prov = e.get("provenance") or {}
            serialized_evidence.append({
                "evidence_id": e.get("evidence_id"),
                "text": e.get("text", ""),
                "document_id": e.get("document_id") or prov.get("document_id"),
                "chunk_id": e.get("chunk_id") or prov.get("chunk_id"),
                "source_type": e.get("source_type", "unknown"),
                "topic": e.get("topic", ""),
                "match_type": e.get("match_type", "unknown"),
                "retrieval_score": e.get("score"),
                "rerank_score": e.get("rerank_score"),
                "origin": prov.get("origin", "unknown"),
                "provenance": prov,
                "verified": prov.get("verified", False),
                "verification_status": "assessed_per_claim",
            })

        return {
            "execution_id": state.execution_id,
            "answer": state.generated_answer,
            "claims": state.verified_claims,
            "evidence": serialized_evidence,
            "hallucination_score": state.hallucination_score,
            "relevance_score": state.relevance_score,
            "query_mode": state.query_mode,
            "question_type": state.question_type,
            "is_emergency": state.is_emergency,
            "is_refusal": state.is_refusal,
            "abstained": state.abstained,
            "abstention_reason": state.abstention_reason,
            "execution_trace": state.execution_trace,
            "agent_latencies_ms": state.agent_latencies_ms,
            "quality_checks": state.quality_checks,
        }
