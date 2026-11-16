"""
ParmanaAI - Retrieval Base Architecture

Defines unified retrieval interfaces, configuration, and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalConfig:
    """Configuration for retrieval components."""
    top_k: int = 5
    threshold: float = 0.35
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60
    source_filter: Optional[str] = None  # "guideline" | "peer_reviewed" | None
    topic_filter: Optional[str] = None
    enable_query_expansion: bool = True
    enable_relevance_validation: bool = True
    min_relevance_threshold: float = 0.20


@dataclass
class RetrievalResult:
    """Standardized retrieval result representation."""
    evidence_id: str
    text: str
    document_id: str
    chunk_id: str
    source_type: str = "unknown"
    topic: str = ""
    match_type: str = "semantic"  # "dense" | "sparse" | "hybrid" | "exact"
    score: float = 0.0
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "topic": self.topic,
            "match_type": self.match_type,
            "score": round(self.score, 4),
            "retrieval_metadata": self.retrieval_metadata,
            "provenance": self.provenance,
        }


class BaseRetriever(ABC):
    """Abstract interface for all retrieval strategies."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()

    @abstractmethod
    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Execute retrieval query and return standardized normalized evidence dicts.
        """
        pass
