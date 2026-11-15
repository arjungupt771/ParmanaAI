
"""
ParmanaAI - Evidence Normalization

Standardizes retrieved evidence while preserving source identity,
document identity, chunk identity and provenance.

This module does not determine whether a source is trustworthy.
It preserves available metadata without inventing missing details.
"""

import hashlib
from typing import Any


NORMALIZER_VERSION = "1.0"


def _clean_text(value: Any) -> str:
    """Convert valid text to a normalized string."""
    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def _stable_hash(value: str) -> str:
    """Generate a deterministic SHA-256 identifier."""
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def normalize_evidence(
    evidence_list: list[dict],
    origin: str = "unknown"
) -> list[dict]:
    """
    Normalize evidence records into a consistent schema.

    Existing metadata is preserved. Missing metadata is represented
    explicitly rather than inferred.

    Args:
        evidence_list: Retrieved evidence records.
        origin: Where the records entered the pipeline.

    Returns:
        A list of normalized evidence records.
    """

    normalized_records = []

    if not isinstance(evidence_list, list):
        return normalized_records

    for index, evidence in enumerate(evidence_list):

        if not isinstance(evidence, dict):
            continue

        text = _clean_text(evidence.get("text"))

        if not text:
            continue

        metadata = evidence.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        provenance = evidence.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}

        # Preserve identifiers already supplied by the retriever.
        document_id = (
            evidence.get("document_id")
            or evidence.get("doc_id")
            or metadata.get("doc_id")
            or provenance.get("document_id")
        )

        chunk_id = (
            evidence.get("chunk_id")
            or evidence.get("id")
            or provenance.get("chunk_id")
        )

        source_type = (
            evidence.get("source_type")
            or metadata.get("source_type")
            or "unknown"
        )

        topic = (
            evidence.get("topic")
            or metadata.get("topic")
            or ""
        )

        record_origin = (
            provenance.get("origin")
            or origin
        )

        # Content-derived fallback identity. This is not a
        # replacement for an actual document identifier.
        if not document_id:
            document_id = f"unknown-document-{_stable_hash(text)[:16]}"

        if not chunk_id:
            chunk_id = f"content-{_stable_hash(text)[:16]}"

        evidence_identity = (
            f"{record_origin}|{document_id}|{chunk_id}|{text}"
        )

        evidence_id = (
            evidence.get("evidence_id")
            or f"EVD-{_stable_hash(evidence_identity)[:20]}"
        )

        normalized_provenance = {
            **provenance,
            "origin": record_origin,
            "document_id": str(document_id),
            "chunk_id": str(chunk_id),
            "source_type": str(source_type),
            "verified": provenance.get("verified", False),
        }

        normalized = {
            **evidence,
            "evidence_id": str(evidence_id),
            "document_id": str(document_id),
            "chunk_id": str(chunk_id),
            "text": text,
            "source_type": str(source_type),
            "topic": str(topic),
            "match_type": evidence.get("match_type", "unknown"),
            "score": evidence.get("score"),
            "provenance": normalized_provenance,
            "normalizer_version": NORMALIZER_VERSION,
        }

        normalized_records.append(normalized)

    return normalized_records
