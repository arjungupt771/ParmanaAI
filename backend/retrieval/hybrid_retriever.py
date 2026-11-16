"""
ParmanaAI - Hybrid Retriever with Reciprocal Rank Fusion (RRF)

Combines Dense Semantic and Sparse BM25 retrieval, normalizes scoring,
deduplicates passages, applies controlled query expansion, and validates
query-to-evidence relevance.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from backend.retrieval.base import BaseRetriever, RetrievalConfig
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.sparse_retriever import BM25Retriever
from backend.retrieval.query_expander import expand_query
from backend.retrieval.relevance_guard import filter_relevant_evidence


logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    cleaned = " ".join(text.lower().split())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]


class HybridRetriever(BaseRetriever):
    """
    Combines dense and sparse retrieval strategies via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        super().__init__(config)
        self.dense_retriever = dense_retriever or DenseRetriever(self.config)
        self.sparse_retriever = sparse_retriever or BM25Retriever(config=self.config)

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval pipeline:
        1. Query understanding & controlled expansion.
        2. Candidate retrieval from dense and sparse retrievers.
        3. Reciprocal Rank Fusion (RRF) deduplication and score blending.
        4. Query-to-evidence relevance validation.
        """
        cfg = config or self.config
        top_k = kwargs.get("top_k", cfg.top_k)
        rrf_k = kwargs.get("rrf_k", cfg.rrf_k)
        dense_weight = kwargs.get("dense_weight", cfg.dense_weight)
        sparse_weight = kwargs.get("sparse_weight", cfg.sparse_weight)
        enable_expansion = kwargs.get("enable_query_expansion", cfg.enable_query_expansion)
        enable_validation = kwargs.get("enable_relevance_validation", cfg.enable_relevance_validation)

        # 1. Query expansion
        if enable_expansion:
            query_info = expand_query(query)
            search_query = query_info["expanded_query"]
        else:
            search_query = query

        # Retrieve top candidate pool (2x top_k) from each strategy
        candidate_k = max(top_k * 3, 10)
        dense_candidates = self.dense_retriever.retrieve(
            search_query,
            top_k=candidate_k,
            source_filter=kwargs.get("source_filter", cfg.source_filter),
        )
        sparse_candidates = self.sparse_retriever.retrieve(
            search_query,
            top_k=candidate_k,
            source_filter=kwargs.get("source_filter", cfg.source_filter),
        )

        # 2. Reciprocal Rank Fusion
        fused_pool: Dict[str, Dict[str, Any]] = {}

        # Process dense candidates
        for rank, item in enumerate(dense_candidates, start=1):
            key = item.get("chunk_id") or _text_hash(item.get("text", ""))
            rrf_contrib = dense_weight / (rrf_k + rank)

            if key not in fused_pool:
                fused_pool[key] = {
                    "item": item,
                    "rrf_score": rrf_contrib,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "dense_score": item.get("score"),
                    "sparse_score": None,
                }
            else:
                fused_pool[key]["rrf_score"] += rrf_contrib
                fused_pool[key]["dense_rank"] = rank
                fused_pool[key]["dense_score"] = item.get("score")

        # Process sparse candidates
        for rank, item in enumerate(sparse_candidates, start=1):
            key = item.get("chunk_id") or _text_hash(item.get("text", ""))
            rrf_contrib = sparse_weight / (rrf_k + rank)

            if key not in fused_pool:
                fused_pool[key] = {
                    "item": item,
                    "rrf_score": rrf_contrib,
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "dense_score": None,
                    "sparse_score": item.get("score"),
                }
            else:
                fused_pool[key]["rrf_score"] += rrf_contrib
                fused_pool[key]["sparse_rank"] = rank
                fused_pool[key]["sparse_score"] = item.get("score")

        if not fused_pool:
            return []

        # 3. Assemble unified results
        max_rrf = max(v["rrf_score"] for v in fused_pool.values()) if fused_pool else 1.0

        fused_results = []
        for key, entry in fused_pool.items():
            base_item = entry["item"]
            normalized_fused_score = min(1.0, entry["rrf_score"] / max_rrf)

            match_type = "hybrid"
            if entry["dense_rank"] is not None and entry["sparse_rank"] is None:
                match_type = "dense"
            elif entry["sparse_rank"] is not None and entry["dense_rank"] is None:
                match_type = "sparse"

            result_item = {
                **base_item,
                "match_type": match_type,
                "score": round(normalized_fused_score, 4),
                "retrieval_metadata": {
                    "strategy": "hybrid_rrf",
                    "rrf_score": round(entry["rrf_score"], 6),
                    "normalized_rrf_score": round(normalized_fused_score, 4),
                    "dense_rank": entry["dense_rank"],
                    "sparse_rank": entry["sparse_rank"],
                    "dense_score": entry["dense_score"],
                    "sparse_score": entry["sparse_score"],
                }
            }
            fused_results.append(result_item)

        # Sort by fused score descending
        fused_results.sort(key=lambda x: x["score"], reverse=True)

        # 4. Relevance validation (Topic/condition mismatch filtering)
        if enable_validation:
            accepted, rejected = filter_relevant_evidence(
                query=query,
                evidence_list=fused_results,
                min_score=cfg.min_relevance_threshold,
            )
            # If no accepted passages remain, abstain gracefully
            final_candidates = accepted if accepted else []
        else:
            final_candidates = fused_results

        return final_candidates[:top_k]
