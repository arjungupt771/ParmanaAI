"""
ParmanaAI - Clinical Retrieval Evaluation Benchmark

Evaluates dense, sparse (BM25), and hybrid retrieval strategies
using standard IR metrics: Recall@K, Precision@K, MRR, and nDCG@K.
"""

import json
import math
import os
from typing import Any, Dict, List, Set

from backend.retrieval.base import RetrievalConfig
from backend.retrieval.sparse_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever


# Curated medical corpus for benchmark
BENCHMARK_CORPUS: List[Dict[str, Any]] = [
    {
        "document_id": "doc-htn-01",
        "chunk_id": "chunk-htn-01",
        "topic": "hypertension",
        "source_type": "guideline",
        "text": "Hypertension is defined as systolic blood pressure ≥ 140 mmHg or diastolic blood pressure ≥ 90 mmHg. First-line therapy includes ACE inhibitors, ARBs, and CCBs."
    },
    {
        "document_id": "doc-t2dm-01",
        "chunk_id": "chunk-t2dm-01",
        "topic": "diabetes",
        "source_type": "guideline",
        "text": "Type 2 diabetes mellitus is diagnosed when fasting blood glucose is ≥ 126 mg/dL or glycated hemoglobin (HbA1c) is ≥ 6.5%. Metformin is initial pharmacotherapy."
    },
    {
        "document_id": "doc-cardio-01",
        "chunk_id": "chunk-cardio-01",
        "topic": "cardiology",
        "source_type": "peer_reviewed",
        "text": "Aspirin 81 mg daily significantly reduces recurrent myocardial infarction (MI) in patients with established coronary artery disease (CAD)."
    },
    {
        "document_id": "doc-asthma-01",
        "chunk_id": "chunk-asthma-01",
        "topic": "pulmonology",
        "source_type": "guideline",
        "text": "Asthma exacerbations should be treated with inhaled short-acting beta-agonists (SABA) like albuterol and systemic corticosteroids for severe symptoms."
    },
    {
        "document_id": "doc-fever-01",
        "chunk_id": "chunk-fever-01",
        "topic": "fever",
        "source_type": "guideline",
        "text": "Fever is defined as body temperature ≥ 38.0°C. Acetaminophen or ibuprofen may be used for symptom relief, but underlying etiology must be evaluated."
    },
]

# Benchmark queries with ground truth relevant document IDs
BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "q1",
        "query": "What is the threshold and first-line treatment for hypertension (HTN)?",
        "relevant_doc_ids": {"doc-htn-01"}
    },
    {
        "query_id": "q2",
        "query": "What fasting glucose and HbA1c levels diagnose type 2 diabetes mellitus (T2DM)?",
        "relevant_doc_ids": {"doc-t2dm-01"}
    },
    {
        "query_id": "q3",
        "query": "Does low-dose aspirin prevent recurrent myocardial infarction in CAD?",
        "relevant_doc_ids": {"doc-cardio-01"}
    },
    {
        "query_id": "q4",
        "query": "What is the initial management for acute asthma exacerbation with albuterol?",
        "relevant_doc_ids": {"doc-asthma-01"}
    },
    {
        "query_id": "q5",
        "query": "At what body temperature is a clinical fever diagnosed?",
        "relevant_doc_ids": {"doc-fever-01"}
    },
]


def calculate_precision_recall_at_k(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int = 3,
) -> Dict[str, float]:
    """Calculate Precision@K and Recall@K."""
    top_k = retrieved_doc_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)

    precision = relevant_in_top_k / k if k > 0 else 0.0
    recall = (
        relevant_in_top_k / len(relevant_doc_ids)
        if len(relevant_doc_ids) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall}


def calculate_mrr(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: Set[str],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int = 3,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (nDCG@K)."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_doc_ids else 0.0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG (all relevant docs ranked at the top)
    idcg = 0.0
    for i in range(1, min(len(relevant_doc_ids), k) + 1):
        idcg += 1.0 / math.log2(i + 1)

    return (dcg / idcg) if idcg > 0.0 else 0.0


def run_retrieval_benchmark(
    corpus: Optional[List[Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None,
    dense_retriever: Optional[Any] = None,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Run evaluation benchmark comparing sparse BM25 and Hybrid retrieval.
    """
    benchmark_corpus = corpus or BENCHMARK_CORPUS
    benchmark_queries = queries or BENCHMARK_QUERIES

    sparse_retriever = BM25Retriever(corpus=benchmark_corpus)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=RetrievalConfig(top_k=k, enable_relevance_validation=True),
    )

    sparse_metrics = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    hybrid_metrics = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}

    n_queries = len(benchmark_queries)

    for item in benchmark_queries:
        query = item["query"]
        relevant = item["relevant_doc_ids"]

        # 1. Sparse BM25
        sparse_results = sparse_retriever.retrieve(query, top_k=k)
        sparse_doc_ids = [r["document_id"] for r in sparse_results]
        pr_sparse = calculate_precision_recall_at_k(sparse_doc_ids, relevant, k=k)
        sparse_metrics["precision"] += pr_sparse["precision"]
        sparse_metrics["recall"] += pr_sparse["recall"]
        sparse_metrics["mrr"] += calculate_mrr(sparse_doc_ids, relevant)
        sparse_metrics["ndcg"] += calculate_ndcg_at_k(sparse_doc_ids, relevant, k=k)

        # 2. Hybrid
        hybrid_results = hybrid_retriever.retrieve(query, top_k=k)
        hybrid_doc_ids = [r["document_id"] for r in hybrid_results]
        pr_hybrid = calculate_precision_recall_at_k(hybrid_doc_ids, relevant, k=k)
        hybrid_metrics["precision"] += pr_hybrid["precision"]
        hybrid_metrics["recall"] += pr_hybrid["recall"]
        hybrid_metrics["mrr"] += calculate_mrr(hybrid_doc_ids, relevant)
        hybrid_metrics["ndcg"] += calculate_ndcg_at_k(hybrid_doc_ids, relevant, k=k)

    # Average metrics
    report = {
        "k": k,
        "n_queries": n_queries,
        "sparse_bm25": {
            f"Precision@{k}": round(sparse_metrics["precision"] / n_queries, 4),
            f"Recall@{k}": round(sparse_metrics["recall"] / n_queries, 4),
            "MRR": round(sparse_metrics["mrr"] / n_queries, 4),
            f"nDCG@{k}": round(sparse_metrics["ndcg"] / n_queries, 4),
        },
        "hybrid_rrf": {
            f"Precision@{k}": round(hybrid_metrics["precision"] / n_queries, 4),
            f"Recall@{k}": round(hybrid_metrics["recall"] / n_queries, 4),
            "MRR": round(hybrid_metrics["mrr"] / n_queries, 4),
            f"nDCG@{k}": round(hybrid_metrics["ndcg"] / n_queries, 4),
        }
    }

    report_path = os.path.join(
        os.path.dirname(__file__),
        "retrieval_benchmark_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_retrieval_benchmark()
    print("Retrieval Benchmark Results:")
    print(json.dumps(report, indent=2))
