
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
