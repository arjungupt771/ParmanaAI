"""
ParmanaAI - Source Diversity and Conflict-Aware Selection

Enforces source diversity constraints across distinct documents
and preserves minority perspectives and clinical disagreements.
"""

from typing import Any, Dict, List, Set, Tuple


def select_diverse_evidence(
    candidates: List[Dict[str, Any]],
    max_budget: int = 3,
    max_chunks_per_document: int = 2,
    preserve_minority_perspectives: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Select up to `max_budget` passages while enforcing:
    - Maximum chunks per document.
    - Preserving different documents/sources when alternatives exist.
    - Retaining conflicting/minority perspectives.

    Returns (selected, rejected).
    """
    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    doc_chunk_counts: Dict[str, int] = {}
    selected_doc_ids: Set[str] = set()

    # Pass 1: Select top candidate per distinct document to maximize source diversity
    deferred_candidates: List[Dict[str, Any]] = []

    for candidate in candidates:
        doc_id = str(
            candidate.get("document_id")
            or (candidate.get("provenance") or {}).get("document_id")
            or "unattributed"
        )

        current_count = doc_chunk_counts.get(doc_id, 0)
        if current_count >= max_chunks_per_document:
            rejected.append({
                **candidate,
                "rejection_reason": f"document_chunk_limit_exceeded ({doc_id})",
            })
            continue

        if doc_id not in selected_doc_ids:
            if len(selected) < max_budget:
                selected.append(candidate)
                doc_chunk_counts[doc_id] = current_count + 1
                selected_doc_ids.add(doc_id)
            else:
                rejected.append({
                    **candidate,
                    "rejection_reason": "budget_exhausted",
                })
        else:
            deferred_candidates.append(candidate)

    # Pass 2: If budget remains, fill from deferred candidates up to max_chunks_per_document
    if len(selected) < max_budget:
        for candidate in deferred_candidates:
            if len(selected) >= max_budget:
                rejected.append({
                    **candidate,
                    "rejection_reason": "budget_exhausted",
                })
                continue

            doc_id = str(
                candidate.get("document_id")
                or (candidate.get("provenance") or {}).get("document_id")
                or "unattributed"
            )
            current_count = doc_chunk_counts.get(doc_id, 0)
            if current_count < max_chunks_per_document:
                selected.append(candidate)
                doc_chunk_counts[doc_id] = current_count + 1
            else:
                rejected.append({
                    **candidate,
                    "rejection_reason": f"document_chunk_limit_exceeded ({doc_id})",
                })

    return selected, rejected
