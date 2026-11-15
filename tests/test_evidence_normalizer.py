
from backend.agents.evidence_normalizer import normalize_evidence


def test_normalizes_chroma_evidence():
    evidence = [{
        "text": "Fever is a clinical sign.",
        "source_type": "guideline",
        "topic": "fever",
        "document_id": "icmr_fever_guidelines.txt",
        "chunk_id": "guideline::icmr_fever_guidelines.txt::0",
        "match_type": "semantic",
        "score": 0.82
    }]

    result = normalize_evidence(
        evidence,
        origin="chroma_vector_store"
    )

    assert len(result) == 1

    record = result[0]

    assert record["document_id"] == "icmr_fever_guidelines.txt"
    assert record["chunk_id"] == "guideline::icmr_fever_guidelines.txt::0"
    assert record["source_type"] == "guideline"
    assert record["topic"] == "fever"
    assert record["provenance"]["origin"] == "chroma_vector_store"
    assert record["provenance"]["verified"] is False
    assert record["evidence_id"].startswith("EVD-")


def test_preserves_existing_metadata():
    evidence = [{
        "text": "Research evidence.",
        "source_type": "peer_reviewed",
        "doc_id": "paper-001",
        "id": "chunk-001",
        "metadata": {
            "publication_year": 2024
        },
        "provenance": {
            "origin": "pubmed",
            "verified": True,
            "doi": "10.1234/example"
        }
    }]

    result = normalize_evidence(evidence)

    record = result[0]

    assert record["document_id"] == "paper-001"
    assert record["chunk_id"] == "chunk-001"
    assert record["metadata"]["publication_year"] == 2024
    assert record["provenance"]["doi"] == "10.1234/example"
    assert record["provenance"]["verified"] is True


def test_generates_deterministic_fallback_identifiers():
    evidence = [{
        "text": "A passage without source identifiers."
    }]

    first = normalize_evidence(evidence)
    second = normalize_evidence(evidence)

    assert first[0]["evidence_id"] == second[0]["evidence_id"]
    assert first[0]["document_id"] == second[0]["document_id"]
    assert first[0]["chunk_id"] == second[0]["chunk_id"]


def test_same_document_keeps_same_document_identity():
    evidence = [
        {
            "text": "First passage from a guideline.",
            "document_id": "guideline-001",
            "chunk_id": "chunk-001",
            "source_type": "guideline"
        },
        {
            "text": "Second passage from the same guideline.",
            "document_id": "guideline-001",
            "chunk_id": "chunk-002",
            "source_type": "guideline"
        }
    ]

    result = normalize_evidence(evidence)

    assert result[0]["document_id"] == result[1]["document_id"]
    assert result[0]["chunk_id"] != result[1]["chunk_id"]
    assert result[0]["evidence_id"] != result[1]["evidence_id"]


def test_external_context_is_marked_unverified():
    evidence = [{
        "text": "Externally supplied medical passage.",
        "source_type": "external_context_unverified",
        "match_type": "unverified",
        "score": None,
        "provenance": {
            "origin": "external_context",
            "verified": False,
            "source_type": "unknown"
        }
    }]

    result = normalize_evidence(
        evidence,
        origin="external_context"
    )

    record = result[0]

    assert record["provenance"]["origin"] == "external_context"
    assert record["provenance"]["verified"] is False
    assert record["source_type"] == "external_context_unverified"
    assert record["score"] is None


def test_empty_and_invalid_records_are_skipped():
    evidence = [
        {"text": ""},
        {"text": "   "},
        None,
        {"text": "Valid evidence passage."}
    ]

    result = normalize_evidence(evidence)

    assert len(result) == 1
    assert result[0]["text"] == "Valid evidence passage."


def test_whitespace_is_normalized():
    evidence = [{
        "text": "  Fever   is \n a clinical   sign. "
    }]

    result = normalize_evidence(evidence)

    assert result[0]["text"] == "Fever is a clinical sign."


def test_does_not_mutate_original_evidence():
    evidence = [{
        "text": "Original evidence.",
        "source_type": "guideline"
    }]

    original = evidence[0].copy()

    normalize_evidence(evidence)

    assert evidence[0] == original
