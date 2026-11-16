"""
ParmanaAI - End-to-End Evidence Selection Strategy

Orchestrates cross-encoder reranking, quality assessment, deduplication,
source diversity constraints, and diagnostic rejection tracking.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.reranking.base import EvidenceSelectionOutcome, RerankConfig
from backend.reranking.cross_encoder import CrossEncoderReranker
from backend.reranking.quality_assessor import assess_passage_quality
from backend.reranking.deduplicator import deduplicate_evidence
from backend.reranking.diversity_selector import select_diverse_evidence


logger = logging.getLogger(__name__)


class EvidenceSelector:
    """
    Comprehensive Evidence Selection Pipeline implementing Phase 6 requirements.
    """

    def __init__(
        self,
        config: Optional[RerankConfig] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.config = config or RerankConfig()
        self.reranker = reranker or CrossEncoderReranker(self.config)

    def select(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        config: Optional[RerankConfig] = None,
    ) -> EvidenceSelectionOutcome:
        """
        Execute full evidence selection pipeline.
        """
        cfg = config or self.config
        diagnostics: Dict[str, Any] = {
            "initial_candidate_count": len(candidates),
            "rejections": {},
        }
        all_rejected: List[Dict[str, Any]] = []

        if not candidates or not query.strip():
            return EvidenceSelectionOutcome(
                selected_evidence=[],
                rejected_evidence=[],
                selection_diagnostics=diagnostics,
                abstained=True,
                abstention_reason="Empty query or no candidate evidence provided.",
            )

        # 1. Evidence Quality Assessment
        qualified = []
        for cand in candidates:
            if cfg.enable_quality_filter:
                quality = assess_passage_quality(cand, query, min_length=cfg.min_content_length)
                if not quality["acceptable"]:
                    reason = "; ".join(quality["reasons"])
                    diagnostics["rejections"].setdefault("low_quality", []).append(cand.get("chunk_id"))
                    all_rejected.append({**cand, "rejection_reason": f"low_quality: {reason}"})
                    continue
            qualified.append(cand)

        if not qualified:
            return EvidenceSelectionOutcome(
                selected_evidence=[],
                rejected_evidence=all_rejected,
                selection_diagnostics=diagnostics,
                abstained=True,
                abstention_reason="All candidates failed basic evidence quality criteria.",
            )

        # 2. Evidence Deduplication (Exact and Near-Duplicates)
        deduped, dup_rejected = deduplicate_evidence(
            qualified,
            near_duplicate_threshold=cfg.near_duplicate_threshold,
        )
        for d in dup_rejected:
            diagnostics["rejections"].setdefault("duplicate", []).append(d.get("chunk_id"))
            all_rejected.append(d)

        # 3. Cross-Encoder Reranking
        reranked = self.reranker.rerank(query, deduped, config=cfg)

        # 4. Filter by minimum rerank threshold
        above_threshold = []
        for cand in reranked:
            if cand["rerank_score"] >= cfg.min_rerank_score:
                above_threshold.append(cand)
            else:
                diagnostics["rejections"].setdefault("below_threshold", []).append(cand.get("chunk_id"))
                all_rejected.append({
                    **cand,
                    "rejection_reason": f"rerank_score_{cand['rerank_score']}_below_{cfg.min_rerank_score}",
                })

        # Fallback if too few candidates pass threshold: keep top-ranked qualified candidate
        if not above_threshold and reranked:
            logger.info("No candidates passed threshold %s; falling back to best reranked candidate", cfg.min_rerank_score)
            above_threshold = [reranked[0]]

        # 5. Source Diversity and Budget Allocation
        selected, div_rejected = select_diverse_evidence(
            above_threshold,
            max_budget=cfg.max_evidence_budget,
            max_chunks_per_document=cfg.max_chunks_per_document,
            preserve_minority_perspectives=cfg.preserve_minority_perspectives,
        )
        for r in div_rejected:
            diagnostics["rejections"].setdefault("diversity_or_budget", []).append(r.get("chunk_id"))
            all_rejected.append(r)

        diagnostics["final_selected_count"] = len(selected)
        diagnostics["total_rejected_count"] = len(all_rejected)

        return EvidenceSelectionOutcome(
            selected_evidence=selected,
            rejected_evidence=all_rejected,
            selection_diagnostics=diagnostics,
            abstained=len(selected) == 0,
            abstention_reason="No qualifying evidence available." if len(selected) == 0 else None,
        )
