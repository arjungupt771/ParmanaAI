"""
ParmanaAI - Phase 6 Reranking & Evidence Selection Tests

Tests:
- Cross-encoder reranking and score preservation
- Evidence quality assessment
- Deduplication (exact and near-duplicates)
- Source diversity constraints across documents
- Evidence selection strategy with diagnostics
- Reranking benchmark
"""

import pytest

from backend.reranking.base import RerankConfig
from backend.reranking.cross_encoder import CrossEncoderReranker
from backend.reranking.quality_assessor import assess_passage_quality
from backend.reranking.deduplicator import deduplicate_evidence, group_evidence_by_document
from backend.reranking.diversity_selector import select_diverse_evidence
from backend.reranking.evidence_selector import EvidenceSelector
from backend.evaluation.reranking_benchmark import run_reranking_benchmark


def test_cross_encoder_reranking_preserves_retrieval_metadata():
    reranker = CrossEncoderReranker()
    candidates = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "ACE inhibitors such as lisinopril are standard treatment for essential hypertension.",
            "score": 0.45,
            "retrieval_metadata": {"strategy": "dense"}
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "text": "The patient presented with knee pain following physical trauma.",
            "score": 0.70,
            "retrieval_metadata": {"strategy": "sparse"}
        }
    ]

    reranked = reranker.rerank("hypertension medication lisinopril", candidates)
    assert len(reranked) == 2
    # c1 is much more clinically relevant to the query than c2
    assert reranked[0]["chunk_id"] == "c1"
    assert "rerank_score" in reranked[0]
    assert reranked[0]["retrieval_metadata"]["original_retrieval_score"] == 0.45
    assert reranked[0]["retrieval_metadata"]["reranker"] == "cross_encoder"


def test_evidence_quality_assessment_filters_boilerplate_and_fragments():
    valid_passage = {
        "text": "Patients with type 2 diabetes should undergo annual retinal examinations.",
        "source_type": "guideline",
    }
    short_fragment = {
        "text": "See table 1.",
        "source_type": "guideline",
    }
    boilerplate = {
        "text": "All rights reserved. Copyright 2024 by Medical Publishing Group.",
        "source_type": "unknown",
    }

    assert assess_passage_quality(valid_passage, "diabetes examination")["acceptable"] is True
    assert assess_passage_quality(short_fragment, "diabetes examination")["acceptable"] is False
    assert assess_passage_quality(boilerplate, "diabetes examination")["acceptable"] is False


def test_deduplication_removes_exact_and_near_duplicates():
    candidates = [
        {"chunk_id": "c1", "text": "Metformin reduces glycated hemoglobin (HbA1c) levels in type 2 diabetes."},
        {"chunk_id": "c2", "text": "Metformin reduces glycated hemoglobin (HbA1c) levels in type 2 diabetes."},  # Exact
        {"chunk_id": "c3", "text": "Metformin therapy reduces glycated hemoglobin (HbA1c) levels in type 2 diabetes."},  # Near dup
        {"chunk_id": "c4", "text": "Insulin therapy is required in severe glycemic decompensation."},  # Unique
    ]

    unique, removed = deduplicate_evidence(candidates, near_duplicate_threshold=0.80)
    assert len(unique) == 2
    assert unique[0]["chunk_id"] == "c1"
    assert unique[1]["chunk_id"] == "c4"
    assert len(removed) == 2


def test_source_diversity_limits_chunks_per_document():
    candidates = [
        {"chunk_id": "doc1-c1", "document_id": "doc1", "text": "Chunk 1 from study 1"},
        {"chunk_id": "doc1-c2", "document_id": "doc1", "text": "Chunk 2 from study 1"},
        {"chunk_id": "doc1-c3", "document_id": "doc1", "text": "Chunk 3 from study 1"},
        {"chunk_id": "doc2-c1", "document_id": "doc2", "text": "Chunk 1 from study 2"},
    ]

    # Max 2 chunks per doc, budget 3
    selected, rejected = select_diverse_evidence(
        candidates,
        max_budget=3,
        max_chunks_per_document=2,
    )
    doc1_count = sum(1 for c in selected if c["document_id"] == "doc1")
    doc2_count = sum(1 for c in selected if c["document_id"] == "doc2")

    assert doc1_count <= 2
    assert doc2_count == 1
    assert len(selected) == 3


def test_evidence_selector_pipeline_with_diagnostics():
    selector = EvidenceSelector(config=RerankConfig(max_evidence_budget=2, min_rerank_score=0.20))
    candidates = [
        {"chunk_id": "c1", "document_id": "doc1", "text": "Statins reduce cardiovascular mortality by lowering LDL."},
        {"chunk_id": "c2", "document_id": "doc1", "text": "Statins reduce cardiovascular mortality by lowering LDL."}, # Dup
        {"chunk_id": "c3", "document_id": "doc2", "text": "Short"}, # Low quality
    ]

    outcome = selector.select("statin LDL reduction", candidates)
    assert outcome.abstained is False
    assert len(outcome.selected_evidence) >= 1
    assert "duplicate" in outcome.selection_diagnostics["rejections"]
    assert "low_quality" in outcome.selection_diagnostics["rejections"]


def test_run_reranking_benchmark_execution():
    report = run_reranking_benchmark(k=3)
    assert report["k"] == 3
    assert "baseline_retrieval" in report
    assert "after_reranking_and_selection" in report
    assert "improvement" in report
