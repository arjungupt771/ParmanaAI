import re
import logging
import os

from typing import Dict, List
from backend.agents.numerical_aggregator import (
    aggregate_numerical_assessments,
    CONFLICTING_EVIDENCE,
)
from backend.agents.nli_engine import predict_nli
from backend.agents.numerical_extractor import extract_numerical_mentions
from backend.agents.numerical_verdict import (
    assess_numerical_text,
    NO_NUMERICAL_EVIDENCE,
    NUMERICALLY_CONSISTENT,
    NUMERICAL_DISCREPANCY,
    CONTEXT_MISMATCH,
    INCONCLUSIVE,
)

logger = logging.getLogger(__name__)


# Calibrated operating thresholds.
ENTAILMENT_THRESHOLD = float(
    os.getenv("PARMANA_ENTAILMENT_THRESHOLD", "0.55")
)

CONTRADICTION_THRESHOLD = float(
    os.getenv("PARMANA_CONTRADICTION_THRESHOLD", "0.60")
)

CONTRADICTION_MIN_SHARED_TERMS = int(
    os.getenv("PARMANA_CONTRADICTION_MIN_SHARED_TERMS", "2")
)

def empty_result(claim: str) -> Dict:
    numerical_assessment = {
        "verdict": NO_NUMERICAL_EVIDENCE,
        "reason": "No evidence available for numerical verification.",
        "numerical_comparison": None,
        "context_comparison": None,
        "source_assessment_count": 0,
        "unique_document_count": 0,
        "consistent_source_count": 0,
        "discrepant_source_count": 0,
        "context_mismatch_count": 0,
        "inconclusive_source_count": 0,
        "no_numerical_evidence_count": 0,
        "source_assessments": [],
        "clinical_truth_established": False,
        "medical_contradiction_established": False,
        "primary_evidence_id": None,
        "evidence_assessments": [],
        "conflict_details": {
            "has_conflict": False,
            "conflict_type": "none",
            "within_document_conflict_count": 0,
            "cross_document_conflict": False,
            "conflicting_documents": [],
        },
    }

    return {
        "claim": claim,
        "claim_text": claim,
        "verdict": "NO_EVIDENCE_PROVIDED",
        "nli_verdict": "NO_EVIDENCE_PROVIDED",
        "combined_verdict": "NO_EVIDENCE_PROVIDED",
        "numerical_assessment": numerical_assessment,
        "verification_signals": {
            "nli": {
                "verdict": "NO_EVIDENCE_PROVIDED",
                "scores": {
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                },
            },
            "numerical": {
                "verdict": numerical_assessment["verdict"],
                "reason": numerical_assessment["reason"],
                "source_assessment_count": 0,
                "unique_document_count": 0,
            },
        },
        "verification_components": [],
        "confidence": 0.0,
        "method": "nli",
        "supporting_source": None,
        "source_type": "unknown",
        "citation": None,
        "citation_match": "missing",
        "supporting_sources": 0,
        "total_sources": 0,
        "claim_type": "medical",
        "explanation": "No external evidence available.",
        "nli_scores": {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 1.0
        },
        "verification_method_used": "none",
        "conflict_details": {
            "has_conflict": False,
            "conflict_type": "none",
            "within_document_conflict_count": 0,
            "cross_document_conflict": False,
            "conflicting_documents": [],
        }
    }


def classify_nli_scores(scores: Dict[str, float]) -> str:
    """
    Convert NLI scores into calibrated conservative verdicts.
    """

    entailment = scores.get("entailment", 0.0)
    contradiction = scores.get("contradiction", 0.0)
    neutral = scores.get("neutral", max(0.0, 1.0 - entailment - contradiction))

    # Entailment check: must exceed calibrated threshold and lead contradiction
    if entailment >= ENTAILMENT_THRESHOLD and entailment > contradiction:
        return "SUPPORTED"

    # Margin entailment check: dominant over neutral and clear lead over contradiction
    if entailment >= 0.50 and entailment > neutral and entailment > (contradiction + 0.15):
        return "SUPPORTED"

    # Contradiction check: must exceed calibrated threshold and lead entailment
    if contradiction >= CONTRADICTION_THRESHOLD and contradiction > entailment:
        return "CONTRADICTED"

    # Margin contradiction check: dominant over neutral and clear lead over entailment
    if contradiction >= 0.50 and contradiction > neutral and contradiction > (entailment + 0.15):
        return "CONTRADICTED"

    return "INSUFFICIENT_EVIDENCE"


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on",
    "at", "for", "from", "with", "by", "as", "and",
    "or", "but", "if", "then", "that", "this",
    "these", "those", "it", "its", "he", "she",
    "they", "them", "their", "his", "her",
    "has", "have", "had", "do", "does", "did",
    "can", "could", "may", "might", "will",
    "would", "should", "must", "not", "no",
    "patient"
}


def extract_content_terms(text: str) -> set[str]:
    """
    Extract meaningful terms for a conservative
    contradiction relevance check.
    """

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    return {
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    }


def has_contradiction_relevance(
    evidence_text: str,
    claim: str
) -> bool:
    """
    Check whether evidence and claim share enough
    meaningful terms before accepting an NLI
    contradiction prediction.

    This is only a relevance guard.
    It does not establish factual contradiction.
    """

    evidence_terms = extract_content_terms(evidence_text)
    claim_terms = extract_content_terms(claim)

    if not evidence_terms or not claim_terms:
        return False

    shared_terms = evidence_terms.intersection(claim_terms)

    return (
        len(shared_terms)
        >= CONTRADICTION_MIN_SHARED_TERMS
    )


def combine_verification_signals(nli_verdict, numerical_assessment):
    numerical_verdict = numerical_assessment.get(
        "verdict",
        INCONCLUSIVE,
    )

    if numerical_verdict == NO_NUMERICAL_EVIDENCE:
        return nli_verdict

    if numerical_verdict == CONFLICTING_EVIDENCE:
        return "CONFLICTING_SIGNALS"

    if numerical_verdict == CONTEXT_MISMATCH:
        return CONTEXT_MISMATCH

    if (
        nli_verdict == "SUPPORTED"
        and numerical_verdict == NUMERICALLY_CONSISTENT
    ):
        return "SUPPORTED"

    if (
        nli_verdict == "CONTRADICTED"
        and numerical_verdict == NUMERICAL_DISCREPANCY
    ):
        return "CONTRADICTED"

    if (
        nli_verdict == "SUPPORTED"
        and numerical_verdict == NUMERICAL_DISCREPANCY
    ):
        return "CONFLICTING_SIGNALS"

    if (
        nli_verdict == "CONTRADICTED"
        and numerical_verdict == NUMERICALLY_CONSISTENT
    ):
        return "CONFLICTING_SIGNALS"

    if numerical_verdict == INCONCLUSIVE:
        if nli_verdict in {"SUPPORTED", "CONTRADICTED"}:
            return nli_verdict
        return INCONCLUSIVE

    if nli_verdict == "INSUFFICIENT_EVIDENCE":
        return "INSUFFICIENT_EVIDENCE"

    return nli_verdict


def assess_numerical_across_candidates(claim, candidates):
    claim_mentions = extract_numerical_mentions(claim)
    source_assessments = []

    if not claim_mentions:
        return {
            "verdict": NO_NUMERICAL_EVIDENCE,
            "reason": "The claim contains no extractable numerical information.",
            "source_assessment_count": 0,
            "unique_document_count": 0,
            "consistent_source_count": 0,
            "discrepant_source_count": 0,
            "context_mismatch_count": 0,
            "inconclusive_source_count": 0,
            "no_numerical_evidence_count": 0,
            "source_assessments": [],
            "clinical_truth_established": False,
            "medical_contradiction_established": False,
        }

    for index, candidate in enumerate(candidates):
        # Only assess evidence that passed the existing NLI relevance
        # decision. An unrelated chunk must not influence aggregation.
        if candidate.get("verdict") not in {"SUPPORTED", "CONTRADICTED"}:
            continue

        evidence = candidate.get("evidence") or {}
        evidence_text = evidence.get("text", "")

        if not isinstance(evidence_text, str) or not evidence_text.strip():
            continue

        evidence_mentions = extract_numerical_mentions(evidence_text)

        assessment = assess_numerical_text(
            claim_text=claim,
            evidence_text=evidence_text,
            claim_mentions=claim_mentions,
            evidence_mentions=evidence_mentions,
        )

        provenance = evidence.get("provenance") or {}
        if not isinstance(provenance, dict):
            provenance = {}
        document_id = (
            evidence.get("document_id")
            or provenance.get("document_id")
        )

        source_assessments.append({
            "candidate_index": index,
            "evidence_id": evidence.get("evidence_id"),
            "document_id": document_id,
            "chunk_id": (
                evidence.get("chunk_id")
                or provenance.get("chunk_id")
            ),
            "source_type": evidence.get("source_type", "unknown"),
            "nli_verdict": candidate.get("verdict"),
            "nli_scores": candidate.get("scores"),
            "assessment": assessment,
        })

    return aggregate_numerical_assessments(source_assessments)


def select_primary_candidate(candidates: List[Dict]) -> Dict:
    """
    Select primary evidence using the final accepted verification signals.

    Ensures that a candidate whose contradiction verdict was downgraded
    by relevance guards does not override legitimate supported, contradicted,
    or sufficiently relevant evidence.
    """
    def selection_key(item: Dict) -> tuple:
        verdict = item.get("verdict")
        scores = item.get("scores") or {}
        guard_passed = item.get("contradiction_guard_passed", True)

        entailment = scores.get("entailment", 0.0)
        contradiction = scores.get("contradiction", 0.0)
        neutral = scores.get("neutral", 0.0)

        # Tier 2: Valid accepted decisive verdicts
        if verdict == "SUPPORTED":
            return (2, entailment)
        if verdict == "CONTRADICTED" and guard_passed:
            return (2, contradiction)

        # Tier 1: Neutral / Insufficient evidence where guard was not violated
        if guard_passed:
            return (1, neutral)

        # Tier 0: Downgraded candidate (failed contradiction relevance guard)
        return (0, neutral)

    return max(candidates, key=selection_key)


def deterministic_verify_claim(
    claim: str,
    evidence_list: List[Dict]
) -> Dict:

    if not evidence_list:
        return empty_result(claim)

    candidates = []

    clean_hypothesis = re.sub(r"^\s*(?:[-*+]|\d+[.)]|•|◦|▪)\s*", "", claim).strip().strip("\"'")

    for index, evidence in enumerate(evidence_list):

        evidence_text = evidence.get("text", "")

        if not evidence_text.strip():
            continue

        try:
            scores = predict_nli(
                premise=evidence_text,
                hypothesis=clean_hypothesis
            )

        except Exception as exc:
            logger.exception(
                "NLI inference failed for evidence %s",
                index
            )

            raise RuntimeError(
                "NLI verification failed. "
                "Refusing to substitute similarity "
                "for factual verification."
            ) from exc

        verdict = classify_nli_scores(scores)
        contradiction_guard_passed = True
        if verdict == "CONTRADICTED":
            contradiction_guard_passed = has_contradiction_relevance(
            evidence_text=evidence_text,
            claim=claim
            )

            if not contradiction_guard_passed:
                logger.info(
                    "Downgrading contradiction to insufficient evidence "
                    "because evidence and claim lack sufficient lexical overlap."
                )
                verdict = "INSUFFICIENT_EVIDENCE"

        candidates.append({
            "index": index,
            "scores": scores,
            "verdict": verdict,
            "evidence": evidence,
            "contradiction_guard_passed": contradiction_guard_passed
        })

    if not candidates:
        return empty_result(claim)

    # Keep the strongest NLI signal for the legacy verdict while the
    # numerical assessment considers every relevant candidate.
    # Uses the final accepted verification signals.

    best = select_primary_candidate(candidates)

    evidence = best["evidence"]
    scores = best["scores"]
    verdict = best["verdict"]

    numerical_assessment = assess_numerical_across_candidates(
        claim=claim,
        candidates=candidates,
    )
    combined_verdict = combine_verification_signals(
        nli_verdict=verdict,
        numerical_assessment=numerical_assessment,
    )

    evidence_assessments = []
    for candidate in candidates:
        candidate_evidence = candidate["evidence"]
        candidate_provenance = (
            candidate_evidence.get("provenance") or {}
        )
        if not isinstance(candidate_provenance, dict):
            candidate_provenance = {}
        candidate_scores = candidate["scores"]
        evidence_assessments.append({
            "candidate_index": candidate["index"],
            "evidence_id": candidate_evidence.get("evidence_id"),
            "document_id": (
                candidate_evidence.get("document_id")
                or candidate_provenance.get("document_id")
            ),
            "chunk_id": (
                candidate_evidence.get("chunk_id")
                or candidate_provenance.get("chunk_id")
            ),
            "source_type": candidate_evidence.get(
                "source_type",
                "unknown",
            ),
            "citation_match": candidate_evidence.get(
                "match_type",
                "unknown",
            ),
            "score": candidate_evidence.get("score"),
            "citation": candidate_evidence.get("text", "")[:500],
            "nli_verdict": candidate["verdict"],
            "nli_scores": {
                "entailment": candidate_scores["entailment"],
                "contradiction": candidate_scores["contradiction"],
                "neutral": candidate_scores["neutral"],
            },
            "contradiction_guard_passed": candidate.get(
                "contradiction_guard_passed",
            ),
        })

    if verdict == "SUPPORTED":
        confidence = scores["entailment"]

    elif verdict == "CONTRADICTED":
        confidence = scores["contradiction"]

    else:
        confidence = max(
            scores["neutral"],
            1 - max(
                scores["entailment"],
                scores["contradiction"]
            )
        )

    return {
        "claim": claim,
        "claim_text": claim,
        "verdict": verdict,
        "nli_verdict": verdict,
        "combined_verdict": combined_verdict,
        "numerical_assessment": numerical_assessment,
        "verification_signals": {
            "nli": {
                "verdict": verdict,
                "scores": {
                    "entailment": scores["entailment"],
                    "contradiction": scores["contradiction"],
                    "neutral": scores["neutral"],
                },
            },
            "numerical": {
                "verdict": numerical_assessment["verdict"],
                "reason": numerical_assessment["reason"],
                "source_assessment_count": numerical_assessment[
                    "source_assessment_count"
                ],
                "unique_document_count": numerical_assessment[
                    "unique_document_count"
                ],
            },
        },
        "verification_components": ["nli", "numerical"],
        "confidence": round(float(confidence), 4),
        "method": "deberta-nli",
        "supporting_source": evidence.get(
            "source_type",
            "unknown"
        ),
        "source_type": evidence.get(
            "source_type",
            "unknown"
        ),
        "citation": evidence.get("text", "")[:500],
        "citation_match": evidence.get(
            "match_type",
            "unknown"
        ),
        "primary_evidence_id": evidence.get("evidence_id"),
        "evidence_assessments": evidence_assessments,
        "conflict_details": numerical_assessment.get("conflict_details", {
            "has_conflict": False,
            "conflict_type": "none",
            "within_document_conflict_count": 0,
            "cross_document_conflict": False,
            "conflicting_documents": [],
        }),
        "supporting_sources": sum(
            1 for candidate in candidates
            if candidate["verdict"] == "SUPPORTED"
        ),
        "total_sources": len(candidates),
        "claim_type": "medical",
        "explanation": (
    f"NLI classification: {verdict}. "
    f"Entailment={scores['entailment']:.4f}, "
    f"Contradiction={scores['contradiction']:.4f}, "
    f"Neutral={scores['neutral']:.4f}. "
    + (
        "The contradiction prediction was downgraded "
        "because the evidence did not pass the relevance guard."
        if (
            scores["contradiction"] >= CONTRADICTION_THRESHOLD
            and not best["contradiction_guard_passed"]
        )
        else ""
    )
),
        "nli_scores": {
            key: round(float(value), 4)
            for key, value in scores.items()
        },
        "verification_method_used": "nli"
    }


def verify_claim(
    claim: str,
    evidence_list: List[Dict]
) -> Dict:

    return deterministic_verify_claim(
        claim,
        evidence_list
    )


def compute_hallucination_score(
    verified_claims: List[Dict]
) -> float:

    if not verified_claims:
        return 0.0

    penalties = {
        "SUPPORTED": 0.0,
        "PARTIALLY_SUPPORTED": 0.5,
        "INSUFFICIENT_EVIDENCE": 0.75,
        "INCONCLUSIVE": 0.75,
        "CONTEXT_MISMATCH": 0.75,
        "CONFLICTING_SIGNALS": 0.75,
        "NO_EVIDENCE_PROVIDED": 1.0,
        "CONTRADICTED": 1.0
    }

    total_penalty = 0.0

    for claim in verified_claims:
        verdict = claim.get(
            "combined_verdict",
            claim.get("verdict", "INSUFFICIENT_EVIDENCE"),
        )

        total_penalty += penalties.get(verdict, 0.75)

    return total_penalty / len(verified_claims)


if __name__ == "__main__":

    claim = "Smoking increases the risk of lung cancer."

    evidence = [
        {
            "source_type": "peer_reviewed",
            "text": (
                "Smoking is associated with an increased "
                "risk of developing lung cancer."
            ),
            "match_type": "semantic"
        }
    ]

    result = verify_claim(claim, evidence)

    print("\nVerification result:")
    print(result)

    print(
        "\nVerifier score:",
        compute_hallucination_score([result])
    )
