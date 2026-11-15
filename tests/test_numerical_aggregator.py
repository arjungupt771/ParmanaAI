from backend.agents.numerical_aggregator import (
    CONFLICTING_EVIDENCE,
    aggregate_numerical_assessments,
)
from backend.agents.numerical_verdict import (
    NO_NUMERICAL_EVIDENCE,
    NUMERICALLY_CONSISTENT,
    NUMERICAL_DISCREPANCY,
)


def assessment(document_id, chunk_id, verdict):
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "assessment": {"verdict": verdict},
    }


def test_aggregates_consistent_and_discrepant_assessments():
    result = aggregate_numerical_assessments([
        assessment("doc-1", "chunk-1", NUMERICALLY_CONSISTENT),
        assessment("doc-1", "chunk-2", NUMERICALLY_CONSISTENT),
        assessment("doc-2", "chunk-1", NUMERICAL_DISCREPANCY),
    ])

    assert result["verdict"] == CONFLICTING_EVIDENCE
    assert result["source_assessment_count"] == 3
    assert result["unique_document_count"] == 2
    assert result["consistent_source_count"] == 2
    assert result["discrepant_source_count"] == 1


def test_empty_aggregation_includes_zero_counts():
    result = aggregate_numerical_assessments([])

    assert result["verdict"] == NO_NUMERICAL_EVIDENCE
    assert result["source_assessment_count"] == 0
    assert result["no_numerical_evidence_count"] == 0


def test_uses_chunk_identity_when_document_id_is_missing():
    result = aggregate_numerical_assessments([
        {
            "chunk_id": "chunk-1",
            "assessment": {"verdict": NUMERICALLY_CONSISTENT},
        },
        {
            "chunk_id": "chunk-2",
            "assessment": {"verdict": NUMERICALLY_CONSISTENT},
        },
    ])

    assert result["unique_document_count"] == 2
