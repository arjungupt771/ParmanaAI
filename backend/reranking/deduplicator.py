"""
ParmanaAI - Evidence Deduplication and Document Grouping

Identifies exact duplicates and near-duplicates (Jaccard similarity),
and groups passages by document identity.
"""

import hashlib
import re
from typing import Any, Dict, List, Set, Tuple


def _normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalized_text(text).encode("utf-8")).hexdigest()


def _get_word_set(text: str) -> Set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def deduplicate_evidence(
    candidates: List[Dict[str, Any]],
    near_duplicate_threshold: float = 0.82,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Remove exact duplicates and near-duplicates, preserving highest-ranked passage.
    Returns (unique_candidates, removed_duplicates).
    """
    unique = []
    removed = []

    seen_hashes: Set[str] = set()
    accepted_word_sets: List[Tuple[Set[str], Dict[str, Any]]] = []

    for candidate in candidates:
        text = candidate.get("text", "")
        h = _text_hash(text)

        # 1. Exact duplicate
        if h in seen_hashes:
            removed.append({
                **candidate,
                "rejection_reason": "exact_duplicate",
            })
            continue

        # 2. Near duplicate (high lexical overlap)
        words = _get_word_set(text)
        is_near_dup = False
        duplicate_of = None

        for prev_words, prev_candidate in accepted_word_sets:
            sim = jaccard_similarity(words, prev_words)
            if sim >= near_duplicate_threshold:
                is_near_dup = True
                duplicate_of = prev_candidate.get("chunk_id")
                break

        if is_near_dup:
            removed.append({
                **candidate,
                "rejection_reason": f"near_duplicate_of_{duplicate_of}",
            })
            continue

        seen_hashes.add(h)
        accepted_word_sets.append((words, candidate))
        unique.append(candidate)

    return unique, removed


def group_evidence_by_document(
    evidence_list: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group passages belonging to the same document to prevent
    treating multiple chunks as independent evidence.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for item in evidence_list:
        doc_id = (
            item.get("document_id")
            or (item.get("provenance") or {}).get("document_id")
            or "unattributed"
        )
        grouped.setdefault(str(doc_id), []).append(item)

    return grouped
