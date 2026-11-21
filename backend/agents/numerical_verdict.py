
"""
ParmanaAI - Numerical Verdict Engine

Phase 4.3.4

Combines numerical comparison and clinical context matching.

This module produces numerical assessment outcomes only.
It does not establish clinical truth or independently declare
a medical claim supported or contradicted.
"""

from typing import Any

from backend.agents.numerical_comparator import (
    compare_numerical_mentions,
)
from backend.agents.clinical_context import (
    compare_clinical_context,
    extract_clinical_context,
)


NUMERICALLY_CONSISTENT = "NUMERICALLY_CONSISTENT"
NUMERICAL_DISCREPANCY = "NUMERICAL_DISCREPANCY"
CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
INCONCLUSIVE = "INCONCLUSIVE"
NO_NUMERICAL_EVIDENCE = "NO_NUMERICAL_EVIDENCE"


CONSISTENT_STATUSES = {
    "equivalent",
    "within_range",
    "satisfies_constraint",
}

DISCREPANCY_STATUSES = {
    "mismatch",
    "outside_range",
    "violates_constraint",
    "disjoint_ranges",
}


def _result(
    verdict: str,
    reason: str,
    numerical_comparison: dict[str, Any] | None = None,
    context_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent structured numerical assessment."""

    return {
        "verdict": verdict,
        "reason": reason,
        "numerical_comparison": numerical_comparison,
        "context_comparison": context_comparison,
        "clinical_truth_established": False,
        "medical_contradiction_established": False,
    }


def _resolve_context(
    claim_context: dict[str, Any],
    evidence_context: dict[str, Any],
) -> str:
    """
    Resolve context status without treating missing information
    as an explicit mismatch.

    Returns:
        mismatch
        match
        unconfirmed
    """

    comparison = compare_clinical_context(
        claim_context,
        evidence_context,
    )

    if comparison["mismatches"]:
        return "mismatch"

    dimension_results = comparison["dimension_results"]

    matched_dimensions = [
        dimension
        for dimension, result in dimension_results.items()
        if result["status"] == "match"
    ]

    unknown_dimensions = [
        dimension
        for dimension, result in dimension_results.items()
        if result["status"] == "unknown"
    ]

    if matched_dimensions and not unknown_dimensions:
        return "match"

    return "unconfirmed"


def assess_numerical_claim(
    claim_mention: dict[str, Any] | None,
    evidence_mention: dict[str, Any] | None,
    claim_context: dict[str, Any] | None = None,
    evidence_context: dict[str, Any] | None = None,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    """
    Assess a claim's numerical mention against one evidence mention.

    Decision order:
    1. Validate numerical evidence.
    2. Compare explicit clinical context.
    3. Compare numerical values.
    4. Resolve a conservative numerical verdict.

    Missing context does not become a mismatch.
    Numerical discrepancy is not a medical contradiction.
    """

    if claim_mention is None and evidence_mention is None:
        return _result(
            NO_NUMERICAL_EVIDENCE,
            "Neither claim nor evidence contains a numerical mention.",
        )

    if claim_mention is None:
        return _result(
            NO_NUMERICAL_EVIDENCE,
            "The claim contains no numerical mention to verify.",
        )

    if evidence_mention is None:
        return _result(
            NO_NUMERICAL_EVIDENCE,
            "No numerical evidence is available for comparison.",
        )

    if not isinstance(claim_mention, dict):
        return _result(
            INCONCLUSIVE,
            "Claim numerical mention has an invalid structure.",
        )

    if not isinstance(evidence_mention, dict):
        return _result(
            INCONCLUSIVE,
            "Evidence numerical mention has an invalid structure.",
        )

    # Context may be supplied by an upstream extraction pipeline.
    # Otherwise, callers can provide text-derived context separately.
    claim_context = claim_context or {}
    evidence_context = evidence_context or {}

    if not isinstance(claim_context, dict):
        claim_context = {}

    if not isinstance(evidence_context, dict):
        evidence_context = {}

    context_comparison = compare_clinical_context(
        claim_context,
        evidence_context,
    )

    context_status = _resolve_context(
        claim_context,
        evidence_context,
    )

    if context_status == "mismatch":
        return _result(
            CONTEXT_MISMATCH,
            "Explicit clinical context differs between claim and evidence.",
            context_comparison=context_comparison,
        )

    numerical_comparison = compare_numerical_mentions(
        claim_mention,
        evidence_mention,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    numerical_status = numerical_comparison.get("status")

    if numerical_status == "incomparable":
        return _result(
            INCONCLUSIVE,
            numerical_comparison.get(
                "reason",
                "Numerical comparison could not be completed.",
            ),
            numerical_comparison,
            context_comparison,
        )

    if numerical_status not in (
        CONSISTENT_STATUSES | DISCREPANCY_STATUSES
    ):
        return _result(
            INCONCLUSIVE,
            "Numerical relationship is not sufficiently determined.",
            numerical_comparison,
            context_comparison,
        )

    # Context is incomplete or unknown.
    # Preserve the numerical observation but do not promote it
    # into a context-confirmed result.
    if context_status == "unconfirmed":

        if numerical_status in CONSISTENT_STATUSES:
            return _result(
                INCONCLUSIVE,
                "Numerical values appear consistent, but clinical "
                "context has not been sufficiently established.",
                numerical_comparison,
                context_comparison,
            )

        return _result(
            INCONCLUSIVE,
            "A numerical discrepancy was detected, but clinical "
            "context has not been sufficiently established.",
            numerical_comparison,
            context_comparison,
        )

    if numerical_status in CONSISTENT_STATUSES:
        return _result(
            NUMERICALLY_CONSISTENT,
            "Numerical values are consistent and the explicitly "
            "compared clinical context matches.",
            numerical_comparison,
            context_comparison,
        )

    return _result(
        NUMERICAL_DISCREPANCY,
        "Numerical values differ under the comparison rules. "
        "This is not independently a medical contradiction.",
        numerical_comparison,
        context_comparison,
    )


def assess_numerical_text(
    claim_text: str,
    evidence_text: str,
    claim_mentions: list[dict[str, Any]],
    evidence_mentions: list[dict[str, Any]],
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    """
    Assess all extracted numerical mentions from two text passages.

    This function intentionally does not automatically pair multiple
    values by list position. Automatic semantic alignment requires
    additional context matching and is outside this phase.
    """

    if not isinstance(claim_mentions, list):
        claim_mentions = []

    if not isinstance(evidence_mentions, list):
        evidence_mentions = []

    if not claim_mentions:
        return _result(
            NO_NUMERICAL_EVIDENCE,
            "No numerical claims were extracted.",
        )

    if not evidence_mentions:
        return _result(
            NO_NUMERICAL_EVIDENCE,
            "No numerical evidence was extracted.",
        )

    claim_context = extract_clinical_context(claim_text)
    evidence_context = extract_clinical_context(evidence_text)

    context_comparison = compare_clinical_context(
        claim_context,
        evidence_context,
    )

    context_status = _resolve_context(
        claim_context,
        evidence_context,
    )

    if context_status == "mismatch":
        return _result(
            CONTEXT_MISMATCH,
            "Explicit clinical context differs between the passages.",
            context_comparison=context_comparison,
        )

    # When exactly one claim mention exists, attempt unit-aware alignment across evidence mentions
    if len(claim_mentions) == 1 and len(evidence_mentions) >= 1:
        c_mention = claim_mentions[0]
        c_unit = (c_mention.get("unit") or "").lower()

        candidates = [
            e for e in evidence_mentions
            if (e.get("unit") or "").lower() == c_unit
            or (not c_unit and not e.get("unit"))
        ]
        if not candidates:
            candidates = evidence_mentions

        best_discrepancy = None
        for cand_e in candidates:
            cand_res = assess_numerical_claim(
                c_mention,
                cand_e,
                claim_context=claim_context,
                evidence_context=evidence_context,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            if cand_res["verdict"] == NUMERICALLY_CONSISTENT:
                return cand_res
            if cand_res["verdict"] == NUMERICAL_DISCREPANCY and best_discrepancy is None:
                best_discrepancy = cand_res

        if best_discrepancy is not None and len(candidates) == 1:
            return best_discrepancy

        return _result(
            INCONCLUSIVE,
            "Multiple numerical mentions could not be unambiguously paired.",
            context_comparison=context_comparison,
        )

    # Multiple mentions must not be paired arbitrarily.
    if len(claim_mentions) != 1 or len(evidence_mentions) != 1:
        return _result(
            INCONCLUSIVE,
            "Multiple numerical mentions require explicit semantic "
            "alignment before comparison.",
            context_comparison=context_comparison,
        )

    return assess_numerical_claim(
        claim_mentions[0],
        evidence_mentions[0],
        claim_context=claim_context,
        evidence_context=evidence_context,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
