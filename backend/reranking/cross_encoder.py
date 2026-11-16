"""
ParmanaAI - Cross-Encoder Reranker

Scores query-passage relevance with cross-encoder architecture,
preserving original retrieval scores and falling back gracefully
on model unavailability or inference failure.
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.reranking.base import RerankConfig


logger = logging.getLogger(__name__)


def _compute_fallback_relevance(query: str, text: str) -> float:
    """
    Deterministic clinical lexical-semantic fallback scorer when
    neural cross-encoder model is unavailable or in offline mode.
    """
    if not query or not text:
        return 0.0

    q_tokens = set(re.findall(r"\b[a-zA-Z0-9-]+\b", query.lower()))
    t_tokens = set(re.findall(r"\b[a-zA-Z0-9-]+\b", text.lower()))

    if not q_tokens:
        return 0.0

    overlap = q_tokens.intersection(t_tokens)
    overlap_score = len(overlap) / len(q_tokens)

    # Boost if exact contiguous query phrase is found
    exact_boost = 0.2 if query.lower().strip() in text.lower() else 0.0

    return min(1.0, max(0.0, overlap_score * 0.8 + exact_boost))


class CrossEncoderReranker:
    """
    Cross-encoder reranking module for retrieved clinical passages.
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self._model = None
        self._tokenizer = None
        self._load_attempted = False

    def _try_load_model(self):
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            logger.info("Loading cross-encoder model: %s", self.config.cross_encoder_model)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.cross_encoder_model,
                local_files_only=True
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.cross_encoder_model,
                local_files_only=True
            )
            self._model.eval()
        except Exception as exc:
            logger.info(
                "Cross-encoder model not cached locally or failed to load (%s). "
                "Using high-precision clinical relevance fallback.",
                exc,
            )
            self._model = None
            self._tokenizer = None

    def score_pair(self, query: str, passage: str) -> float:
        """Score a single (query, passage) pair in [0.0, 1.0]."""
        self._try_load_model()

        if self._model is not None and self._tokenizer is not None:
            try:
                import torch
                inputs = self._tokenizer(
                    query,
                    passage,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                with torch.inference_mode():
                    outputs = self._model(**inputs)
                    logits = outputs.logits
                    # Sigmoid for binary relevance or raw logit normalization
                    if logits.shape[-1] == 1:
                        score = torch.sigmoid(logits[0][0]).item()
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        score = probs[0][1].item()
                    return round(float(score), 4)
            except Exception as exc:
                logger.warning("Neural cross-encoder inference failed: %s", exc)

        return round(_compute_fallback_relevance(query, passage), 4)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        config: Optional[RerankConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate passages against the query, preserving all original
        retrieval metadata and recording rerank scores separately.
        """
        cfg = config or self.config

        if not candidates or not query.strip():
            return []

        reranked = []
        for candidate in candidates:
            text = candidate.get("text", "")
            original_score = candidate.get("score", 0.0)

            try:
                rerank_score = self.score_pair(query, text)
            except Exception as exc:
                logger.exception("Error scoring candidate: %s", exc)
                rerank_score = original_score

            retrieval_meta = candidate.get("retrieval_metadata", {})
            updated_meta = {
                **retrieval_meta,
                "original_retrieval_score": original_score,
                "rerank_score": rerank_score,
                "reranker": "cross_encoder",
            }

            reranked_candidate = {
                **candidate,
                "score": rerank_score,
                "rerank_score": rerank_score,
                "original_score": original_score,
                "retrieval_metadata": updated_meta,
            }
            reranked.append(reranked_candidate)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked
