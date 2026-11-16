"""
ParmanaAI - Phase 5 Hybrid Retrieval Tests

Tests:
- Query understanding and controlled abbreviation expansion
- Sparse BM25 retrieval and medical tokenization
- Query-to-evidence relevance validation (hypertension vs diabetes mismatch)
- Hybrid Reciprocal Rank Fusion (RRF) deduplication and ranking
- Retrieval evaluation metrics (Precision@K, Recall@K, MRR, nDCG@K)
"""

import pytest

from backend.retrieval.base import RetrievalConfig
from backend.retrieval.sparse_retriever import BM25Retriever, tokenize_medical_text
from backend.retrieval.query_expander import expand_query, extract_query_entities
from backend.retrieval.relevance_guard import (
    validate_query_evidence_relevance,
    filter_relevant_evidence,
)
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.evaluation.retrieval_benchmark import (
    calculate_precision_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    run_retrieval_benchmark,
)


def test_medical_tokenization_preserves_clinical_terms():
    text = "Patient with Type-2 diabetes receives 500 mg/dL metformin for HbA1c control."
    tokens = tokenize_medical_text(text)
    assert "type-2" in tokens
    assert "diabetes" in tokens
    assert "mg/dl" in tokens
    assert "metformin" in tokens
    assert "hba1c" in tokens
    assert "for" not in tokens  # Stopword removed


def test_query_understanding_and_controlled_expansion():
    query = "What is the recommended medication for HTN in elderly patients?"
    info = expand_query(query)

    assert info["original_query"] == query
    assert "hypertension" in info["expanded_query"]
    entities = info["entities"]
    assert "older_adult" in entities["populations"]
    assert any(abbr[0] == "htn" for abbr in entities["abbreviations"])


def test_sparse_bm25_retrieval():
    corpus = [
        {
            "document_id": "doc-1",
            "chunk_id": "c1",
            "source_type": "guideline",
            "text": "Lisinopril is an ACE inhibitor used in hypertension therapy."
        },
        {
            "document_id": "doc-2",
            "chunk_id": "c2",
            "source_type": "guideline",
            "text": "Metformin reduces hepatic glucose production in type 2 diabetes mellitus."
        },
        {
            "document_id": "doc-3",
            "chunk_id": "c3",
            "source_type": "peer_reviewed",
            "text": "Atorvastatin lowers LDL cholesterol and reduces cardiovascular risk."
        }
    ]

    retriever = BM25Retriever(corpus=corpus)
    results = retriever.retrieve("Lisinopril for hypertension", top_k=2)

    assert len(results) > 0
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["match_type"] == "sparse"
    assert "raw_bm25_score" in results[0]["retrieval_metadata"]


def test_relevance_guard_detects_hypertension_vs_diabetes_mismatch():
    """
    Directly tests the hypertension-versus-diabetes topic mismatch issue.
    """
    query = "What is the target blood pressure in essential hypertension?"
    unrelated_diabetes_evidence = (
        "In patients with type 2 diabetes mellitus, intensive glycemic control "
        "targeting HbA1c below 6.5% reduces microvascular complications."
    )

    validation = validate_query_evidence_relevance(query, unrelated_diabetes_evidence)
    assert validation["mismatch_detected"] is True
    assert validation["relevance_level"] == "topic_mismatch"
    assert validation["is_relevant"] is False

    matching_htn_evidence = (
        "For adults with essential hypertension, the target blood pressure "
        "is systolic blood pressure below 130 mmHg and diastolic below 80 mmHg."
    )
    val_matching = validate_query_evidence_relevance(query, matching_htn_evidence)
    assert val_matching["is_relevant"] is True
    assert val_matching["relevance_level"] == "highly_relevant"


def test_hybrid_rrf_combines_and_deduplicates():
    corpus = [
        {
            "document_id": "doc-med-1",
            "chunk_id": "c1",
            "source_type": "guideline",
            "text": "Metformin is recommended as first-line therapy for type 2 diabetes."
        },
        {
            "document_id": "doc-med-2",
            "chunk_id": "c2",
            "source_type": "guideline",
            "text": "Empagliflozin provides cardiovascular benefit in type 2 diabetes."
        }
    ]

    sparse_retriever = BM25Retriever(corpus=corpus)

    # Mock dense retriever that returns candidate c1
    class MockDenseRetriever:
        def retrieve(self, query, top_k=5, **kwargs):
            return [{
                "evidence_id": "dense-c1",
                "text": corpus[0]["text"],
                "document_id": "doc-med-1",
                "chunk_id": "c1",
                "source_type": "guideline",
                "score": 0.88,
            }]

    hybrid = HybridRetriever(
        dense_retriever=MockDenseRetriever(),
        sparse_retriever=sparse_retriever,
        config=RetrievalConfig(top_k=2),
    )

    results = hybrid.retrieve("Metformin in type 2 diabetes", top_k=2)
    assert len(results) > 0
    # doc-med-1 was returned by both dense and sparse, so its match_type should be hybrid
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["match_type"] == "hybrid"
    assert "rrf_score" in results[0]["retrieval_metadata"]


def test_retrieval_evaluation_metrics():
    retrieved = ["doc-1", "doc-2", "doc-3"]
    relevant = {"doc-1", "doc-4"}

    pr = calculate_precision_recall_at_k(retrieved, relevant, k=3)
    assert pr["precision"] == pytest.approx(1.0 / 3.0)
    assert pr["recall"] == pytest.approx(0.5)

    mrr = calculate_mrr(retrieved, relevant)
    assert mrr == 1.0  # doc-1 is at rank 1

    ndcg = calculate_ndcg_at_k(retrieved, relevant, k=3)
    assert ndcg > 0.0


def test_run_retrieval_benchmark_reproducible():
    class FastMockDenseRetriever:
        def retrieve(self, query, **kwargs):
            return []

    report = run_retrieval_benchmark(dense_retriever=FastMockDenseRetriever(), k=3)
    assert report["k"] == 3
    assert "sparse_bm25" in report
    assert "hybrid_rrf" in report
    assert report["sparse_bm25"]["Recall@3"] > 0.0

