"""
ParmanaAI - Reranking Base Configuration and Schemas
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RerankConfig:
    """Configuration for cross-encoder reranking and evidence selection."""
    max_evidence_budget: int = 3
    min_rerank_score: float = 0.25
    max_chunks_per_document: int = 2
    near_duplicate_threshold: float = 0.82
    min_content_length: int = 30
    preserve_minority_perspectives: bool = True
    enable_quality_filter: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class EvidenceSelectionOutcome:
    """Outcome of evidence selection pipeline."""
    selected_evidence: List[Dict[str, Any]]
    rejected_evidence: List[Dict[str, Any]]
    selection_diagnostics: Dict[str, Any] = field(default_factory=dict)
    abstained: bool = False
    abstention_reason: Optional[str] = None
