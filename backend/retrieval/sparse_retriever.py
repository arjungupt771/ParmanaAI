"""
ParmanaAI - Clinical BM25 Sparse Retriever

Implements BM25Okapi scoring with medical terminology awareness,
exact keyword matching, and document metadata preservation.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set

from backend.retrieval.base import BaseRetriever, RetrievalConfig


# Basic English + generic filler stopwords (clinical terms excluded from stop words)
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've",
    "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize_medical_text(text: str) -> List[str]:
    """
    Tokenize medical text while preserving exact drug names, hyphenated conditions,
    and units (e.g., 'type-2', 'hba1c', 'mg/dl', 'covid-19').
    """
    if not isinstance(text, str):
        return []

    lowered = text.lower()
    # Match alphanumeric sequences, retaining hyphens or slashes in technical terms
    tokens = re.findall(r"\b[a-z0-9]+(?:[-/][a-z0-9]+)*\b", lowered)

    return [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]


class BM25Retriever(BaseRetriever):
    """
    In-memory BM25Okapi retriever for exact clinical keyword matching.
    """

    def __init__(
        self,
        corpus: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RetrievalConfig] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        super().__init__(config)
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.tokenized_corpus: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        if corpus:
            self.index(corpus)

    def index(self, corpus: List[Dict[str, Any]]) -> None:
        """
        Index a collection of documents/chunks.
        Each item is expected to have 'text', 'document_id', 'chunk_id', etc.
        """
        self.corpus = corpus
        self.tokenized_corpus = []
        self.doc_lengths = []
        self.doc_freqs = {}

        total_length = 0
        for doc in corpus:
            tokens = tokenize_medical_text(doc.get("text", ""))
            self.tokenized_corpus.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            seen_terms = set(tokens)
            for term in seen_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        n_docs = len(corpus)
        self.avg_doc_length = (total_length / n_docs) if n_docs > 0 else 0.0

        # Calculate standard BM25 IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents matching the query using BM25.
        """
        cfg = config or self.config
        top_k = kwargs.get("top_k", cfg.top_k)
        source_filter = kwargs.get("source_filter", cfg.source_filter)

        if not self.corpus or not query.strip():
            return []

        query_tokens = tokenize_medical_text(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * len(self.corpus)

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for idx, doc_tokens in enumerate(self.tokenized_corpus):
                # Apply source filter if defined
                if source_filter:
                    doc_src = self.corpus[idx].get("source_type")
                    if doc_src and doc_src != source_filter:
                        continue

                term_count = doc_tokens.count(token)
                if term_count == 0:
                    continue

                doc_len = self.doc_lengths[idx]
                num = term_count * (self.k1 + 1.0)
                denom = term_count + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0)))
                scores[idx] += idf_val * (num / denom)

        # Pair scores with documents
        results = []
        max_score = max(scores) if scores and max(scores) > 0 else 1.0

        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue

            doc = self.corpus[idx]
            # Normalized BM25 score in [0.0, 1.0]
            normalized_score = min(1.0, score / max_score)

            results.append({
                "evidence_id": doc.get("evidence_id") or f"sparse-{doc.get('chunk_id', idx)}",
                "text": doc.get("text", ""),
                "document_id": doc.get("document_id") or doc.get("doc_id"),
                "chunk_id": doc.get("chunk_id") or str(idx),
                "source_type": doc.get("source_type", "unknown"),
                "topic": doc.get("topic", ""),
                "match_type": "sparse",
                "score": round(normalized_score, 4),
                "retrieval_metadata": {
                    "strategy": "bm25",
                    "raw_bm25_score": round(score, 4),
                    "matched_query_tokens": [t for t in query_tokens if t in self.tokenized_corpus[idx]],
                },
                "provenance": doc.get("provenance") or {
                    "origin": "bm25_sparse_index",
                    "document_id": doc.get("document_id"),
                    "chunk_id": doc.get("chunk_id"),
                    "source_type": doc.get("source_type", "unknown"),
                    "verified": False,
                }
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
