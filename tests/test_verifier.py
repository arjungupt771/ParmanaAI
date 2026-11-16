
import pytest

import backend.agents.verifier as verifier


def test_import_module():
    assert verifier is not None


def test_verify_claim_supported(monkeypatch):

    def mock_nli(premise, hypothesis):
        return {
            "entailment": 0.92,
            "contradiction": 0.02,
            "neutral": 0.06
        }

    monkeypatch.setattr(
        verifier,
        "predict_nli",
        mock_nli
    )

    evidence = [
        {
            "text": "Smoking increases lung cancer risk.",
            "source_type": "peer_reviewed"
        }
    ]

    result = verifier.verify_claim(
        "Smoking increases lung cancer risk.",
        evidence
    )

    assert result["verdict"] == "SUPPORTED"
    assert result["confidence"] == 0.92
    assert result["method"] == "deberta-nli"
    assert result["nli_verdict"] == "SUPPORTED"
    assert result["combined_verdict"] == "SUPPORTED"
    assert (
        result["numerical_assessment"]["verdict"]
        == "NO_NUMERICAL_EVIDENCE"
    )
    assert result["verification_components"] == ["nli", "numerical"]


def test_verify_claim_returns_evidence_assessments(monkeypatch):
    def mock_nli(premise, hypothesis):
        if "primary" in premise:
            return {
                "entailment": 0.94,
                "contradiction": 0.02,
                "neutral": 0.04,
            }

        return {
            "entailment": 0.83,
            "contradiction": 0.02,
            "neutral": 0.15,
        }

    monkeypatch.setattr(verifier, "predict_nli", mock_nli)

    result = verifier.verify_claim(
        "Smoking increases lung cancer risk.",
        [
            {
                "text": "The primary passage supports the claim.",
                "evidence_id": "evidence-primary",
                "source_type": "peer_reviewed",
                "match_type": "semantic",
                "score": 0.91,
                "provenance": {
                    "document_id": "document-primary",
                    "chunk_id": "chunk-primary",
                },
            },
            {
                "text": "Another passage supports the claim.",
                "evidence_id": "evidence-secondary",
                "document_id": "document-secondary",
                "chunk_id": "chunk-secondary",
                "provenance": "invalid",
            },
        ],
    )

    assert result["primary_evidence_id"] == "evidence-primary"
    assert len(result["evidence_assessments"]) == 2
    assert result["evidence_assessments"] == [
        {
            "candidate_index": 0,
            "evidence_id": "evidence-primary",
            "document_id": "document-primary",
            "chunk_id": "chunk-primary",
            "source_type": "peer_reviewed",
            "citation_match": "semantic",
            "score": 0.91,
            "citation": "The primary passage supports the claim.",
            "nli_verdict": "SUPPORTED",
            "nli_scores": {
                "entailment": 0.94,
                "contradiction": 0.02,
                "neutral": 0.04,
            },
            "contradiction_guard_passed": True,
        },
        {
            "candidate_index": 1,
            "evidence_id": "evidence-secondary",
            "document_id": "document-secondary",
            "chunk_id": "chunk-secondary",
            "source_type": "unknown",
            "citation_match": "unknown",
            "score": None,
            "citation": "Another passage supports the claim.",
            "nli_verdict": "SUPPORTED",
            "nli_scores": {
                "entailment": 0.83,
                "contradiction": 0.02,
                "neutral": 0.15,
            },
            "contradiction_guard_passed": True,
        },
    ]


def test_numerical_discrepancy_conflicts_with_supported_nli(monkeypatch):

    def mock_nli(premise, hypothesis):
        return {
            "entailment": 0.92,
            "contradiction": 0.02,
            "neutral": 0.06
        }

    monkeypatch.setattr(verifier, "predict_nli", mock_nli)

    result = verifier.verify_claim(
        "Blood pressure was 120 mmHg.",
        [{"text": "Blood pressure was 140 mmHg."}]
    )

    assert result["verdict"] == "SUPPORTED"
    assert result["nli_verdict"] == "SUPPORTED"
    assert (
        result["numerical_assessment"]["verdict"]
        == "NUMERICAL_DISCREPANCY"
    )
    assert result["combined_verdict"] == "CONFLICTING_SIGNALS"


def test_numerical_aggregation_uses_all_relevant_candidates():
    result = verifier.assess_numerical_across_candidates(
        claim="Blood pressure was 120 mmHg.",
        candidates=[
            {
                "verdict": "SUPPORTED",
                "scores": {"entailment": 0.9},
                "evidence": {
                    "text": "Blood pressure was 120 mmHg.",
                    "document_id": "guideline-1",
                    "chunk_id": "chunk-1",
                },
            },
            {
                "verdict": "CONTRADICTED",
                "scores": {"contradiction": 0.9},
                "evidence": {
                    "text": "Blood pressure was 140 mmHg.",
                    "document_id": "study-2",
                    "chunk_id": "chunk-2",
                },
            },
            {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "scores": {"neutral": 0.9},
                "evidence": {
                    "text": "Blood pressure was 999 mmHg.",
                    "document_id": "unrelated-3",
                    "chunk_id": "chunk-3",
                },
            },
        ],
    )

    assert result["verdict"] == verifier.CONFLICTING_EVIDENCE
    assert result["source_assessment_count"] == 2
    assert result["unique_document_count"] == 2
    assert result["consistent_source_count"] == 1
    assert result["discrepant_source_count"] == 1
    assert (
        verifier.combine_verification_signals("SUPPORTED", result)
        == "CONFLICTING_SIGNALS"
    )


def test_verify_claim_contradicted(monkeypatch):

    def mock_nli(premise, hypothesis):
        return {
            "entailment": 0.02,
            "contradiction": 0.94,
            "neutral": 0.04
        }

    monkeypatch.setattr(
        verifier,
        "predict_nli",
        mock_nli
    )

    evidence = [
        {
            "text": "The treatment does not reduce blood pressure."
        }
    ]

    result = verifier.verify_claim(
        "The treatment reduces blood pressure.",
        evidence
    )

    assert result["verdict"] == "CONTRADICTED"


def test_verify_claim_insufficient(monkeypatch):

    def mock_nli(premise, hypothesis):
        return {
            "entailment": 0.15,
            "contradiction": 0.10,
            "neutral": 0.75
        }

    monkeypatch.setattr(
        verifier,
        "predict_nli",
        mock_nli
    )

    evidence = [
        {
            "text": "The study examined unrelated dietary patterns."
        }
    ]

    result = verifier.verify_claim(
        "Smoking causes cardiovascular disease.",
        evidence
    )

    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_no_evidence():

    result = verifier.verify_claim(
        "Smoking causes disease.",
        []
    )

    assert result["verdict"] == "NO_EVIDENCE_PROVIDED"
    assert result["nli_verdict"] == "NO_EVIDENCE_PROVIDED"
    assert result["combined_verdict"] == "NO_EVIDENCE_PROVIDED"
    assert (
        result["numerical_assessment"]["verdict"]
        == "NO_NUMERICAL_EVIDENCE"
    )
    assert result["verification_components"] == []


def test_exact_threshold_is_supported(monkeypatch):

    def mock_nli(premise, hypothesis):
        return {
            "entailment": 0.75,
            "contradiction": 0.10,
            "neutral": 0.15
        }

    monkeypatch.setattr(
        verifier,
        "predict_nli",
        mock_nli
    )

    result = verifier.verify_claim(
        "A claim about a medical treatment.",
        [{"text": "Evidence about the treatment."}]
    )

    assert result["verdict"] == "SUPPORTED"


def test_verification_failure_does_not_fallback(monkeypatch):

    def failing_nli(premise, hypothesis):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(
        verifier,
        "predict_nli",
        failing_nli
    )

    with pytest.raises(RuntimeError):
        verifier.verify_claim(
            "A medical claim.",
            [{"text": "Some medical evidence."}]
        )


def test_hallucination_score_penalizes_contradiction():

    score = verifier.compute_hallucination_score([
        {"verdict": "SUPPORTED"},
        {"verdict": "CONTRADICTED"}
    ])

    assert score == 0.5


def test_hallucination_score_for_all_supported():

    score = verifier.compute_hallucination_score([
        {"verdict": "SUPPORTED"},
        {"verdict": "SUPPORTED"}
    ])

    assert score == 0.0


def test_hallucination_score_for_no_evidence():

    score = verifier.compute_hallucination_score([
        {"verdict": "NO_EVIDENCE_PROVIDED"}
    ])

    assert score == 1.0


def test_hallucination_score_prefers_combined_verdict():

    score = verifier.compute_hallucination_score([
        {
            "verdict": "SUPPORTED",
            "combined_verdict": "CONFLICTING_SIGNALS"
        }
    ])

    assert score == 0.75


def test_hallucination_score_falls_back_to_legacy_verdict():

    score = verifier.compute_hallucination_score([
        {"verdict": "SUPPORTED"}
    ])

    assert score == 0.0


def test_primary_evidence_selection_ignores_downgraded_contradiction(monkeypatch):
    """
    Ensure primary evidence is selected using accepted verification signals,
    not raw downgraded contradiction scores.
    """
    claim = "Metformin lowers blood glucose levels."

    def mock_nli(premise, hypothesis):
        if "agricultural" in premise:
            # Irrelevant passage that DeBERTa misclassifies as contradiction
            return {
                "entailment": 0.01,
                "contradiction": 0.98,
                "neutral": 0.01
            }
        # Legitimate supporting passage
        return {
            "entailment": 0.85,
            "contradiction": 0.02,
            "neutral": 0.13
        }

    monkeypatch.setattr(verifier, "predict_nli", mock_nli)

    evidence_list = [
        {
            "text": "The study examined agricultural soil and crop productivity in wheat fields.",
            "evidence_id": "ev-soil-irrelevant",
            "document_id": "doc-soil",
            "chunk_id": "chunk-1",
        },
        {
            "text": "Metformin therapy significantly lowers blood glucose levels in diabetic patients.",
            "evidence_id": "ev-metformin-supported",
            "document_id": "doc-metformin",
            "chunk_id": "chunk-1",
        }
    ]

    result = verifier.verify_claim(claim, evidence_list)

    assert result["verdict"] == "SUPPORTED"
    assert result["primary_evidence_id"] == "ev-metformin-supported"
    assert result["supporting_sources"] == 1

