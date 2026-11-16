"""
ParmanaAI - Retrieval Agent Node

Executes hybrid (dense + sparse) retrieval and external-context ingestion,
preserving document provenance.
"""

from typing import Any, Dict, List, Optional

from backend.agents.multi_agent.agent_base import BaseAgent
from backend.agents.multi_agent.state import SharedAgentState
from backend.agents.evidence_normalizer import normalize_evidence
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.semantic_retriever import retrieve as semantic_retrieve


class RetrievalAgent(BaseAgent):
    """
    Coordinates multi-strategy evidence retrieval.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        name: str = "RetrievalAgent",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled)
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()

    def execute(self, state: SharedAgentState) -> SharedAgentState:
        # If emergency or safety refusal, skip retrieval
        if state.is_emergency or state.is_refusal:
            state.retrieved_candidates = []
            return state

        # Handle external context if supplied
        if state.external_context and state.external_context.strip():
            raw_text = state.external_context.strip()
            words = raw_text.split()
            chunk_size = 250
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

            raw_candidates = [
                {
                    "text": chunk,
                    "source_type": "external_context_unverified",
                    "match_type": "unverified",
                    "score": None,
                    "provenance": {
                        "origin": "external_context",
                        "verified": False,
                        "source_type": "unknown",
                    }
                }
                for chunk in chunks
            ]
            state.retrieval_strategy = "external_context"
        else:
            source_filter = "guideline" if state.query_mode == "FACTUAL" else "peer_reviewed"
            search_query = state.expanded_query or state.user_query

            try:
                raw_candidates = self.hybrid_retriever.retrieve(
                    search_query,
                    source_filter=source_filter,
                    top_k=8,
                )
            except Exception as exc:
                state.retrieval_error = str(exc)
                raw_candidates = None

            if not raw_candidates and state.query_mode == "FACTUAL":
                # Fallback to peer_reviewed abstracts
                try:
                    raw_candidates = self.hybrid_retriever.retrieve(
                        search_query,
                        source_filter="peer_reviewed",
                        top_k=8,
                    )
                except Exception:
                    raw_candidates = None

            if not raw_candidates:
                raw_candidates = semantic_retrieve(
                    state.user_query,
                    source_filter=source_filter,
                    top_k=5,
                )
                state.retrieval_strategy = "dense_semantic_fallback"
            else:
                state.retrieval_strategy = "hybrid_rrf"

        origin = "external_context" if state.external_context else "chroma_vector_store"
        normalized = normalize_evidence(raw_candidates or [], origin=origin)
        state.retrieved_candidates = normalized

        return state
