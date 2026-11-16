"""
ParmanaAI - Phase 14 Database & History Persistence Tests

Tests:
- Database schema initialization
- Atomic persistence of queries, claims, and evidence
- Full record retrieval with joined relations
- Query history pagination and aggregation statistics
"""

import os
import pytest

from backend.database.db import init_db
from backend.database.repository import QueryRepository


@pytest.fixture
def test_repo(tmp_path):
    db_file = str(tmp_path / "test_parmana.db")
    init_db(db_file)
    return QueryRepository(db_path=db_file)


def test_save_and_retrieve_query_with_claims_and_evidence(test_repo):
    request_id = "req-test-12345"
    user_query = "What is the recommended target blood pressure in diabetes?"
    scrubbed_query = user_query
    answer = "The recommended target blood pressure in diabetes is < 130/80 mmHg."

    claims = [
        {
            "claim_id": "CLM-001",
            "claim_text": "Target blood pressure in diabetes is < 130/80 mmHg.",
            "verdict": "SUPPORTED",
            "combined_verdict": "SUPPORTED",
            "primary_evidence_id": "ev-guideline-1",
            "nli_verdict": "entailment",
            "numerical_status": "NUMERICALLY_CONSISTENT",
            "is_inferential": False,
        }
    ]

    evidence = [
        {
            "evidence_id": "ev-guideline-1",
            "document_id": "doc-ada-2024",
            "chunk_id": "chunk-102",
            "source_type": "guideline",
            "score": 0.94,
            "text": "For individuals with diabetes and hypertension, target blood pressure is < 130/80 mmHg.",
            "provenance": {"org": "ADA", "year": 2024},
        }
    ]

    query_id = test_repo.save_query_result(
        request_id=request_id,
        user_query=user_query,
        scrubbed_query=scrubbed_query,
        answer=answer,
        query_mode="FACTUAL",
        is_emergency=False,
        is_refusal=False,
        hallucination_score=0.0,
        latency_ms=250.5,
        claims=claims,
        evidence=evidence,
    )

    assert query_id.startswith("qry-")

    # Fetch by query_id
    record = test_repo.get_query(query_id)
    assert record is not None
    assert record["request_id"] == request_id
    assert record["answer"] == answer
    assert record["hallucination_score"] == 0.0

    # Verify joined claims
    assert len(record["claims"]) == 1
    assert record["claims"][0]["claim_id"] == "CLM-001"
    assert record["claims"][0]["verdict"] == "SUPPORTED"

    # Verify joined evidence
    assert len(record["evidence"]) == 1
    assert record["evidence"][0]["document_id"] == "doc-ada-2024"
    assert record["evidence"][0]["provenance"]["org"] == "ADA"


def test_query_listing_and_statistics(test_repo):
    # Save a normal query
    test_repo.save_query_result(
        request_id="req-1",
        user_query="Q1",
        scrubbed_query="Q1",
        answer="A1",
        query_mode="FACTUAL",
        hallucination_score=0.1,
        latency_ms=100.0,
    )

    # Save an emergency query
    test_repo.save_query_result(
        request_id="req-2",
        user_query="Q2",
        scrubbed_query="Q2",
        answer="A2",
        query_mode="EMERGENCY",
        is_emergency=True,
        is_refusal=True,
        refusal_reason="EMERGENCY_TRIAGE",
        hallucination_score=0.0,
        latency_ms=20.0,
    )

    history = test_repo.list_queries(limit=10)
    assert len(history) == 2

    stats = test_repo.get_statistics()
    assert stats["total_queries"] == 2
    assert stats["emergency_count"] == 1
    assert stats["refusal_count"] == 1
    assert stats["avg_hallucination_score"] == 0.05
    assert stats["avg_latency_ms"] == 60.0
