
from typing import Any, Dict, List

from backend.agents.numerical_verdict import (
    NO_NUMERICAL_EVIDENCE,
    NUMERICALLY_CONSISTENT,
    NUMERICAL_DISCREPANCY,
    CONTEXT_MISMATCH,
    INCONCLUSIVE,
)


CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


def _get_document_id(item: Dict[str, Any]):
    provenance = item.get("provenance") or {}

    if not isinstance(provenance, dict):
        provenance = {}

    return (
        item.get("document_id")
        or provenance.get("document_id")
        or item.get("chunk_id")
        or provenance.get("chunk_id")
        or item.get("id")
    )


def _classify_document_assessments(
    assessments: List[Dict[str, Any]],
) -> str:
    """
    Produce one outcome per document.

    Opposing numerical outcomes within the same document are
    inconclusive rather than independent-source conflict.
    """
    verdicts = {
        (item.get("assessment") or {}).get(
            "verdict",
            INCONCLUSIVE,
        )
        for item in assessments
    }

    has_consistent = NUMERICALLY_CONSISTENT in verdicts
    has_discrepant = NUMERICAL_DISCREPANCY in verdicts

    if has_consistent and has_discrepant:
        return INCONCLUSIVE

    if has_discrepant:
        return NUMERICAL_DISCREPANCY

    if has_consistent:
        return NUMERICALLY_CONSISTENT

    if CONTEXT_MISMATCH in verdicts:
        return CONTEXT_MISMATCH

    if (
        verdicts
        and verdicts == {NO_NUMERICAL_EVIDENCE}
    ):
        return NO_NUMERICAL_EVIDENCE

    return INCONCLUSIVE


def aggregate_numerical_assessments(
    source_assessments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate numerical assessments at document level.

    Unknown-provenance assessments are retained, but cannot establish
    a conflict between independent documents.
    """
    if not source_assessments:
        return {
            "verdict": NO_NUMERICAL_EVIDENCE,
            "reason": (
                "No relevant evidence sources were available "
                "for numerical comparison."
            ),
            "source_assessment_count": 0,
            "unique_document_count": 0,
            "source_group_count": 0,
            "consistent_source_count": 0,
            "discrepant_source_count": 0,
            "context_mismatch_count": 0,
            "inconclusive_source_count": 0,
            "no_numerical_evidence_count": 0,
            "unattributed_assessment_count": 0,
            "within_document_conflict_count": 0,
            "source_assessments": [],
            "clinical_truth_established": False,
            "medical_contradiction_established": False,
        }

    documents = {}
    unattributed = []

    for index, item in enumerate(source_assessments):
        document_id = _get_document_id(item)

        if document_id is None:
            unattributed.append(item)
            continue

        documents.setdefault(str(document_id), []).append(item)

    document_results = []

    for document_id, assessments in documents.items():
        document_results.append({
            "document_id": document_id,
            "verdict": _classify_document_assessments(
                assessments
            ),
            "assessment_count": len(assessments),
        })

    # Without document identity, each assessment is retained as
    # unattributed. It is not treated as proof of an independent source.
    unattributed_results = [
        {
            "document_id": None,
            "verdict": (
                item.get("assessment") or {}
            ).get("verdict", INCONCLUSIVE),
            "assessment_count": 1,
        }
        for item in unattributed
    ]

    known_verdicts = {
        item["verdict"]
        for item in document_results
    }

    known_consistent = (
        NUMERICALLY_CONSISTENT in known_verdicts
    )
    known_discrepant = (
        NUMERICAL_DISCREPANCY in known_verdicts
    )

    unattributed_verdicts = {
        item["verdict"]
        for item in unattributed_results
    }

    unattributed_consistent = (
        NUMERICALLY_CONSISTENT in unattributed_verdicts
    )
    unattributed_discrepant = (
        NUMERICAL_DISCREPANCY in unattributed_verdicts
    )

    within_document_conflicts = sum(
        1
        for document_id, assessments in documents.items()
        if (
            NUMERICALLY_CONSISTENT
            in {
                (item.get("assessment") or {}).get("verdict")
                for item in assessments
            }
            and NUMERICAL_DISCREPANCY
            in {
                (item.get("assessment") or {}).get("verdict")
                for item in assessments
            }
        )
    )

    # A conflict is established only between identifiable documents.
    if known_consistent and known_discrepant:
        verdict = CONFLICTING_EVIDENCE
        reason = (
            "Different identifiable documents contain both "
            "numerically consistent and discrepant assessments."
        )

    # Unknown provenance cannot safely establish source independence.
    elif (
        (known_consistent or unattributed_consistent)
        and (known_discrepant or unattributed_discrepant)
    ):
        verdict = INCONCLUSIVE
        reason = (
            "Numerical assessments disagree, but available provenance "
            "is insufficient to establish an independent-source conflict."
        )

    elif known_discrepant or unattributed_discrepant:
        verdict = NUMERICAL_DISCREPANCY
        reason = (
            "The available comparable numerical assessments indicate "
            "a discrepancy."
        )

    elif known_consistent or unattributed_consistent:
        verdict = NUMERICALLY_CONSISTENT
        reason = (
            "The available comparable numerical assessments are "
            "numerically consistent."
        )

    elif any(
        item["verdict"] == CONTEXT_MISMATCH
        for item in document_results + unattributed_results
    ):
        verdict = CONTEXT_MISMATCH
        reason = (
            "The available numerical comparisons involve mismatched "
            "clinical contexts."
        )

    elif (
        source_assessments
        and all(
            (item.get("assessment") or {}).get("verdict")
            == NO_NUMERICAL_EVIDENCE
            for item in source_assessments
        )
    ):
        verdict = NO_NUMERICAL_EVIDENCE
        reason = (
            "The relevant evidence did not contain comparable "
            "numerical information."
        )

    else:
        verdict = INCONCLUSIVE
        reason = (
            "The available numerical comparisons are insufficient "
            "or incomparable."
        )

    all_groups = document_results + unattributed_results

    has_conflict = verdict == CONFLICTING_EVIDENCE or within_document_conflicts > 0
    conflict_type = (
        "cross_document" if verdict == CONFLICTING_EVIDENCE
        else ("within_document" if within_document_conflicts > 0 else "none")
    )
    conflicting_docs = [
        item["document_id"]
        for item in document_results
        if item["verdict"] in (NUMERICALLY_CONSISTENT, NUMERICAL_DISCREPANCY, INCONCLUSIVE)
    ] if has_conflict else []

    conflict_details = {
        "has_conflict": has_conflict,
        "conflict_type": conflict_type,
        "within_document_conflict_count": within_document_conflicts,
        "cross_document_conflict": verdict == CONFLICTING_EVIDENCE,
        "conflicting_documents": conflicting_docs,
    }

    return {
        "verdict": verdict,
        "reason": reason,
        "source_assessment_count": len(source_assessments),
        "unique_document_count": len(documents),
        "source_group_count": len(all_groups),
        "consistent_source_count": sum(
            (item.get("assessment") or {}).get("verdict") == NUMERICALLY_CONSISTENT
            for item in source_assessments
        ),
        "discrepant_source_count": sum(
            (item.get("assessment") or {}).get("verdict") == NUMERICAL_DISCREPANCY
            for item in source_assessments
        ),
        "context_mismatch_count": sum(
            (item.get("assessment") or {}).get("verdict") == CONTEXT_MISMATCH
            for item in source_assessments
        ),
        "inconclusive_source_count": sum(
            (item.get("assessment") or {}).get("verdict") == INCONCLUSIVE
            for item in source_assessments
        ),
        "no_numerical_evidence_count": sum(
            (item.get("assessment") or {}).get("verdict") == NO_NUMERICAL_EVIDENCE
            for item in source_assessments
        ),
        "unattributed_assessment_count": len(unattributed),
        "within_document_conflict_count": (
            within_document_conflicts
        ),
        "conflict_details": conflict_details,
        "document_results": document_results,
        "document_assessments": document_results,
        "source_assessments": source_assessments,
        "clinical_truth_established": False,
        "medical_contradiction_established": False,
    }
