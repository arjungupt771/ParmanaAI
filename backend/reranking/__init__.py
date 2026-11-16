"""
ParmanaAI - Reranking and Evidence Selection Module
"""

from backend.reranking.base import RerankConfig, EvidenceSelectionOutcome
from backend.reranking.cross_encoder import CrossEncoderReranker
from backend.reranking.quality_assessor import assess_passage_quality
from backend.reranking.deduplicator import deduplicate_evidence, group_evidence_by_document
from backend.reranking.diversity_selector import select_diverse_evidence
from backend.reranking.evidence_selector import EvidenceSelector

__all__ = [
    "RerankConfig",
    "EvidenceSelectionOutcome",
    "CrossEncoderReranker",
    "assess_passage_quality",
    "deduplicate_evidence",
    "group_evidence_by_document",
    "select_diverse_evidence",
    "EvidenceSelector",
]
