"""
ParmanaAI - Phase 4.7 Safety and Verification Evaluation Suite

Tests:
- Medical negation
- Population mismatch
- Age and dosage mismatch
- Unit mismatch
- Conflicting studies
- Insufficient evidence
- Unrelated evidence
- Manually labeled verification benchmark
"""

import pytest
from unittest.mock import patch

from backend.agents.verifier import verify_claim, compute_hallucination_score
from backend.agents.numerical_verdict import assess_numerical_text, CONTEXT_MISMATCH, NUMERICAL_DISCREPANCY, NUMERICALLY_CONSISTENT
from backend.agents.numerical_aggregator import aggregate_numerical_assessments, CONFLICTING_EVIDENCE


def test_safety_medical_negation():
    """
    Ensure negation in claim is accurately assessed against affirmative evidence.
    """
    claim = "Aspirin does not reduce mortality in cardiovascular patients."
    evidence = [{
        "text": "Aspirin significantly reduces all-cause mortality in cardiovascular patients.",
        "evidence_id": "ev-asp-1",
        "document_id": "doc-asp-1",
        "chunk_id": "c1",
        "source_type": "peer_reviewed",
    }]

    # When NLI predicts contradiction between the negation claim and affirmative evidence
    with patch("backend.agents.verifier.predict_nli", return_value={"entailment": 0.02, "contradiction": 0.94, "neutral": 0.04}):
        result = verify_claim(claim, evidence)
        assert result["verdict"] == "CONTRADICTED"
        assert result["combined_verdict"] == "CONTRADICTED"


def test_safety_population_mismatch():
    """
    Ensure adult evidence is not considered equivalent to a pediatric claim.
    """
    claim = "The recommended dose for pediatric patients is 500 mg."
    evidence = "The recommended dose for adult patients is 500 mg."

    claim_mentions = [{"value": 500.0, "normalized_unit": "mg", "comparator": "none", "category": "mass", "statistic": None}]
    evidence_mentions = [{"value": 500.0, "normalized_unit": "mg", "comparator": "none", "category": "mass", "statistic": None}]

    assessment = assess_numerical_text(
        claim_text=claim,
        evidence_text=evidence,
        claim_mentions=claim_mentions,
        evidence_mentions=evidence_mentions,
    )
    assert assessment["verdict"] == CONTEXT_MISMATCH
    assert "populations" in assessment["context_comparison"]["mismatches"]


def test_safety_dosage_mismatch():
    """
    Ensure 50 mg vs 500 mg produces numerical discrepancy.
    """
    claim = "The daily dose is 500 mg."
    evidence = "The daily dose is 50 mg."

    claim_mentions = [{"value": 500.0, "normalized_unit": "mg", "comparator": "none", "category": "mass", "statistic": None}]
    evidence_mentions = [{"value": 50.0, "normalized_unit": "mg", "comparator": "none", "category": "mass", "statistic": None}]

    assessment = assess_numerical_text(
        claim_text=claim,
        evidence_text=evidence,
        claim_mentions=claim_mentions,
        evidence_mentions=evidence_mentions,
    )
    assert assessment["verdict"] == NUMERICAL_DISCREPANCY


def test_safety_unit_mismatch():
    """
    Ensure incompatible units (mg vs ml) return inconclusive.
    """
    claim = "The concentration was 50 mg."
    evidence = "The concentration was 50 ml."

    claim_mentions = [{"value": 50.0, "normalized_unit": "mg", "comparator": "none", "category": "mass", "statistic": None}]
    evidence_mentions = [{"value": 50.0, "normalized_unit": "ml", "comparator": "none", "category": "volume", "statistic": None}]

    assessment = assess_numerical_text(
        claim_text=claim,
        evidence_text=evidence,
        claim_mentions=claim_mentions,
        evidence_mentions=evidence_mentions,
    )
    assert assessment["verdict"] == "INCONCLUSIVE"


def test_safety_conflicting_independent_studies():
    """
    Ensure two independent studies with contradictory numerical outcomes
    produce CONFLICTING_EVIDENCE and CONFLICTING_SIGNALS.
    """
    claim = "Systolic blood pressure was 120 mmHg."
    evidence_list = [
        {
            "text": "In study A, mean systolic blood pressure was 120 mmHg.",
            "evidence_id": "ev-docA",
            "document_id": "doc-A",
            "chunk_id": "c1",
        },
        {
            "text": "In study B, mean systolic blood pressure was 140 mmHg.",
            "evidence_id": "ev-docB",
            "document_id": "doc-B",
            "chunk_id": "c1",
        }
    ]

    def mock_nli(premise, hypothesis):
        if "study A" in premise:
            return {"entailment": 0.90, "contradiction": 0.05, "neutral": 0.05}
        return {"entailment": 0.10, "contradiction": 0.85, "neutral": 0.05}

    with patch("backend.agents.verifier.predict_nli", side_effect=mock_nli):
        result = verify_claim(claim, evidence_list)
        assert result["numerical_assessment"]["verdict"] == CONFLICTING_EVIDENCE
        assert result["combined_verdict"] == "CONFLICTING_SIGNALS"
        assert result["conflict_details"]["has_conflict"] is True
        assert result["conflict_details"]["conflict_type"] == "cross_document"


def test_safety_unrelated_evidence_contradiction_guard():
    """
    Ensure irrelevant evidence is never marked as a real contradiction.
    """
    claim = "Smoking increases the risk of lung cancer."
    evidence_list = [{
        "text": "The soil pH was measured at 6.5 in the agricultural zone.",
        "evidence_id": "ev-soil",
        "document_id": "doc-soil",
        "chunk_id": "c1",
    }]

    # Even if NLI returns high contradiction due to domain confusion
    with patch("backend.agents.verifier.predict_nli", return_value={"entailment": 0.01, "contradiction": 0.96, "neutral": 0.03}):
        result = verify_claim(claim, evidence_list)
        # Relevance guard must downgrade to INSUFFICIENT_EVIDENCE
        assert result["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert result["combined_verdict"] == "INSUFFICIENT_EVIDENCE"


def test_safety_empty_evidence_semantics():
    """
    Ensure empty evidence produces NO_EVIDENCE_PROVIDED with 0 confidence and 1.0 hallucination penalty.
    """
    result = verify_claim("Any medical claim.", [])
    assert result["verdict"] == "NO_EVIDENCE_PROVIDED"
    assert result["confidence"] == 0.0
    assert result["combined_verdict"] == "NO_EVIDENCE_PROVIDED"
    assert compute_hallucination_score([result]) == 1.0


def test_verification_benchmark_suite():
    """
    Standard benchmark testing multiple clinical scenarios and their expected verdicts.
    """
    benchmark_cases = [
        {
            "scenario": "Direct Entailment",
            "claim": "Metformin reduces glycated hemoglobin in type 2 diabetes.",
            "evidence": [{"text": "Metformin therapy significantly reduces glycated hemoglobin (HbA1c) in type 2 diabetes.", "evidence_id": "e1"}],
            "mock_nli": {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03},
            "expected_verdict": "SUPPORTED"
        },
        {
            "scenario": "Direct Factual Contradiction",
            "claim": "Metformin increases blood glucose levels in diabetic patients.",
            "evidence": [{"text": "Metformin therapy significantly lowers blood glucose levels in diabetic patients.", "evidence_id": "e2"}],
            "mock_nli": {"entailment": 0.01, "contradiction": 0.95, "neutral": 0.04},
            "expected_verdict": "CONTRADICTED"
        },
        {
            "scenario": "Insufficient Evidence",
            "claim": "Vitamin C prevents all viral respiratory infections.",
            "evidence": [{"text": "Vitamin C was administered during clinical observation.", "evidence_id": "e3"}],
            "mock_nli": {"entailment": 0.20, "contradiction": 0.15, "neutral": 0.65},
            "expected_verdict": "INSUFFICIENT_EVIDENCE"
        }
    ]

    for case in benchmark_cases:
        with patch("backend.agents.verifier.predict_nli", return_value=case["mock_nli"]):
            res = verify_claim(case["claim"], case["evidence"])
            assert res["verdict"] == case["expected_verdict"], f"Failed scenario: {case['scenario']}"
