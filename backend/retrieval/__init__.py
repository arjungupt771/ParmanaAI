"""
ParmanaAI - Retrieval Module
"""

from backend.retrieval.base import BaseRetriever, RetrievalConfig, RetrievalResult
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.sparse_retriever import BM25Retriever, tokenize_medical_text
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.query_expander import expand_query, extract_query_entities
from backend.retrieval.relevance_guard import (
    validate_query_evidence_relevance,
    filter_relevant_evidence,
)

__all__ = [
    "BaseRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    "DenseRetriever",
    "BM25Retriever",
    "tokenize_medical_text",
    "HybridRetriever",
    "expand_query",
    "extract_query_entities",
    "validate_query_evidence_relevance",
    "filter_relevant_evidence",
]
