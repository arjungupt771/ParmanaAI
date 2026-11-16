"""
ParmanaAI - Query-to-Evidence Relevance Guard

Prevents cross-topic evidence confusion (e.g., retrieving diabetes evidence
for a hypertension query) and filters passages that share only generic terminology.
"""

import re
from typing import Any, Dict, List, Set, Tuple

from backend.retrieval.query_expander import extract_query_entities, COMMON_CONDITIONS


GENERIC_MEDICAL_WORDS: Set[str] = {
    "patient", "patients", "study", "studies", "trial", "trials",
    "treatment", "treatments", "therapy", "therapies", "clinical",
    "outcome", "outcomes", "effect", "effects", "result", "results",
    "significant", "significantly", "group", "groups", "control",
    "placebo", "dose", "doses", "daily", "daily", "associated",
    "increase", "increased", "decrease", "decreased", "risk", "risks",
    "medical", "health", "care", "evaluation", "assessed", "observed",
    "reported", "data", "analysis", "evidence", "guidelines", "guideline",
    "recommendation", "recommendations", "management", "hospital", "doctor"
}


def _extract_meaningful_stems(text: str) -> Set[str]:
    """Extract informative non-generic content words."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in GENERIC_MEDICAL_WORDS}


def validate_query_evidence_relevance(
    query: str,
    evidence_text: str,
    min_overlap: int = 1,
) -> Dict[str, Any]:
    """
    Validate whether a retrieved evidence passage is medically relevant
    to the specific query, detecting condition/topic mismatch.
    """
    if not query or not evidence_text:
        return {
            "is_relevant": False,
            "relevance_level": "unrelated",
            "score": 0.0,
            "shared_entities": [],
            "mismatch_detected": True,
            "mismatch_reason": "Empty query or evidence text",
        }

    query_entities = extract_query_entities(query)
    evidence_entities = extract_query_entities(evidence_text)

    query_conditions = set(query_entities.get("conditions", []))
    evidence_conditions = set(evidence_entities.get("conditions", []))

    query_medications = set(query_entities.get("medications", []))
    evidence_medications = set(evidence_entities.get("medications", []))

    query_measurements = set(query_entities.get("measurements", []))
    evidence_measurements = set(evidence_entities.get("measurements", []))

    # 1. Condition mismatch check (e.g. Hypertension vs Diabetes)
    # If the query asks explicitly about one condition, but the evidence mentions
    # a completely different condition and never mentions the query condition:
    condition_mismatch = False
    mismatch_reason = ""

    if query_conditions and evidence_conditions:
        shared_conditions = query_conditions.intersection(evidence_conditions)
        if not shared_conditions:
            # Query has Condition A, evidence has Condition B with NO overlap
            condition_mismatch = True
            mismatch_reason = (
                f"Topic mismatch: Query focuses on {sorted(query_conditions)} "
                f"but evidence discusses {sorted(evidence_conditions)}."
            )

    # 2. Content lexical overlap on non-generic clinical terms
    query_stems = _extract_meaningful_stems(query)
    evidence_stems = _extract_meaningful_stems(evidence_text)
    shared_stems = query_stems.intersection(evidence_stems)

    # Calculate stem overlap ratio
    overlap_ratio = len(shared_stems) / max(len(query_stems), 1)

    # Shared high-value entities
    shared_entities = list(
        (query_conditions & evidence_conditions) |
        (query_medications & evidence_medications) |
        (query_measurements & evidence_measurements)
    )

    if condition_mismatch and not shared_entities:
        return {
            "is_relevant": False,
            "relevance_level": "topic_mismatch",
            "score": round(max(0.0, overlap_ratio * 0.2), 3),
            "shared_entities": [],
            "shared_terms": list(shared_stems),
            "mismatch_detected": True,
            "mismatch_reason": mismatch_reason,
        }

    if len(shared_entities) > 0:
        relevance_level = "highly_relevant"
        relevance_score = min(1.0, 0.6 + overlap_ratio * 0.4)
        is_relevant = True
    elif len(shared_stems) >= min_overlap:
        relevance_level = "partially_relevant"
        relevance_score = min(0.6, overlap_ratio)
        is_relevant = True
    else:
        relevance_level = "unrelated"
        relevance_score = 0.0
        is_relevant = False
        mismatch_reason = "No meaningful specific clinical terms shared."

    return {
        "is_relevant": is_relevant,
        "relevance_level": relevance_level,
        "score": round(relevance_score, 3),
        "shared_entities": shared_entities,
        "shared_terms": list(shared_stems),
        "mismatch_detected": condition_mismatch and not is_relevant,
        "mismatch_reason": mismatch_reason if not is_relevant else "",
    }


def filter_relevant_evidence(
    query: str,
    evidence_list: List[Dict[str, Any]],
    min_score: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter candidate evidence list, separating accepted relevant passages
    from rejected/unrelated passages.
    """
    accepted = []
    rejected = []

    for item in evidence_list:
        text = item.get("text", "")
        validation = validate_query_evidence_relevance(query, text)

        enriched_item = {
            **item,
            "relevance_validation": validation,
        }

        if validation["is_relevant"] and validation["score"] >= min_score:
            accepted.append(enriched_item)
        else:
            rejected.append(enriched_item)

    return accepted, rejected
