"""
ParmanaAI - Multi-Agent Shared Execution State

Standardized state contract passed across all agents in the orchestration graph.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SharedAgentState:
    """
    Central shared state object capturing all intermediate representations,
    agent execution traces, and diagnostics across the pipeline.
    """
    user_query: str
    external_context: Optional[str] = None
    execution_id: str = field(
        default_factory=lambda: f"exec-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    )

    # 1. Query Understanding
    query_mode: str = "FACTUAL"  # "FACTUAL" | "INFERENTIAL" | "EMERGENCY" | "REFUSAL"
    question_type: str = "factual"  # "factual", "inferential", "numerical", "comparative", "emergency"
    entities: Dict[str, List[Any]] = field(default_factory=dict)
    expanded_query: str = ""
    is_ambiguous: bool = False
    is_emergency: bool = False

    # 2. Retrieval
    retrieved_candidates: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_strategy: str = "hybrid_rrf"
    retrieval_error: Optional[str] = None

    # 3. Evidence Analysis & Reranking
    selected_evidence: List[Dict[str, Any]] = field(default_factory=list)
    rejected_evidence: List[Dict[str, Any]] = field(default_factory=list)
    evidence_diagnostics: Dict[str, Any] = field(default_factory=dict)
    abstained: bool = False
    abstention_reason: Optional[str] = None

    # 4. Generation
    generated_answer: str = ""
    is_refusal: bool = False
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

    # 5. Verification
    claim_records: List[Dict[str, Any]] = field(default_factory=list)
    verified_claims: List[Dict[str, Any]] = field(default_factory=list)
    hallucination_score: float = 0.0
    relevance_score: float = 0.0

    # 6. Quality Control & Observability
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    agent_latencies_ms: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    quality_checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def log_event(self, agent_name: str, event_type: str, details: Dict[str, Any]) -> None:
        self.execution_trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "event": event_type,
            "details": details,
        })
