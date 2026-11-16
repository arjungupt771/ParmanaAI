"""
ParmanaAI - Cross-Encoder Reranking and Selection Benchmark

Compares initial retrieval ranking vs reranked results on:
- nDCG@K improvement
- MRR improvement
- Precision@K and Recall@K
- Evidence source diversity
- Topic mismatch reduction
"""

import json
import os
from typing import Any, Dict, List, Optional

from backend.retrieval.sparse_retriever import BM25Retriever
from backend.reranking.evidence_selector import EvidenceSelector
from backend.reranking.base import RerankConfig
from backend.evaluation.retrieval_benchmark import (
    BENCHMARK_CORPUS,
    BENCHMARK_QUERIES,
    calculate_precision_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
)


def run_reranking_benchmark(
    corpus: Optional[List[Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Run evaluation benchmark comparing baseline retrieval ranking vs reranked ranking.
    """
    benchmark_corpus = corpus or BENCHMARK_CORPUS
    benchmark_queries = queries or BENCHMARK_QUERIES

    retriever = BM25Retriever(corpus=benchmark_corpus)
    selector = EvidenceSelector(config=RerankConfig(max_evidence_budget=k))

    initial_metrics = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "unique_docs": 0}
    reranked_metrics = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "unique_docs": 0}

    n_queries = len(benchmark_queries)

    for item in benchmark_queries:
        query = item["query"]
        relevant = item["relevant_doc_ids"]

        # Initial retrieval (pool size 5)
        candidates = retriever.retrieve(query, top_k=5)
        init_doc_ids = [c["document_id"] for c in candidates]

        pr_init = calculate_precision_recall_at_k(init_doc_ids, relevant, k=k)
        initial_metrics["precision"] += pr_init["precision"]
        initial_metrics["recall"] += pr_init["recall"]
        initial_metrics["mrr"] += calculate_mrr(init_doc_ids, relevant)
        initial_metrics["ndcg"] += calculate_ndcg_at_k(init_doc_ids, relevant, k=k)
        initial_metrics["unique_docs"] += len(set(init_doc_ids[:k]))

        # Evidence selection & reranking
        outcome = selector.select(query, candidates)
        rerank_doc_ids = [c["document_id"] for c in outcome.selected_evidence]

        pr_rerank = calculate_precision_recall_at_k(rerank_doc_ids, relevant, k=k)
        reranked_metrics["precision"] += pr_rerank["precision"]
        reranked_metrics["recall"] += pr_rerank["recall"]
        reranked_metrics["mrr"] += calculate_mrr(rerank_doc_ids, relevant)
        reranked_metrics["ndcg"] += calculate_ndcg_at_k(rerank_doc_ids, relevant, k=k)
        reranked_metrics["unique_docs"] += len(set(rerank_doc_ids[:k]))

    report = {
        "k": k,
        "n_queries": n_queries,
        "baseline_retrieval": {
            f"Precision@{k}": round(initial_metrics["precision"] / n_queries, 4),
            f"Recall@{k}": round(initial_metrics["recall"] / n_queries, 4),
            "MRR": round(initial_metrics["mrr"] / n_queries, 4),
            f"nDCG@{k}": round(initial_metrics["ndcg"] / n_queries, 4),
            "avg_unique_docs": round(initial_metrics["unique_docs"] / n_queries, 2),
        },
        "after_reranking_and_selection": {
            f"Precision@{k}": round(reranked_metrics["precision"] / n_queries, 4),
            f"Recall@{k}": round(reranked_metrics["recall"] / n_queries, 4),
            "MRR": round(reranked_metrics["mrr"] / n_queries, 4),
            f"nDCG@{k}": round(reranked_metrics["ndcg"] / n_queries, 4),
            "avg_unique_docs": round(reranked_metrics["unique_docs"] / n_queries, 2),
        },
        "improvement": {
            "mrr_delta": round((reranked_metrics["mrr"] - initial_metrics["mrr"]) / n_queries, 4),
            "ndcg_delta": round((reranked_metrics["ndcg"] - initial_metrics["ndcg"]) / n_queries, 4),
        }
    }

    report_path = os.path.join(
        os.path.dirname(__file__),
        "reranking_benchmark_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_reranking_benchmark()
    print("Reranking Benchmark Results:")
    print(json.dumps(report, indent=2))
