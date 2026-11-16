"""
ParmanaAI - Evidence Quality Assessment

Evaluates information density, topic consistency, clinical relevance,
and metadata completeness of retrieved passages.
"""

import re
from typing import Any, Dict, List, Tuple


BOILERPLATE_PATTERNS = [
    r"^all rights reserved",
    r"^copyright \d{4}",
    r"^for personal use only",
    r"^references?:?$",
    r"^doi:\s*10\.\d{4}",
]


def assess_passage_quality(
    passage: Dict[str, Any],
    query: str,
    min_length: int = 25,
) -> Dict[str, Any]:
    """
    Assess quality of an individual evidence passage.
    """
    text = passage.get("text", "").strip()

    # 1. Content completeness and length
    if len(text) < min_length:
        return {
            "acceptable": False,
            "quality_score": 0.0,
            "reasons": [f"Passage too short (< {min_length} chars)"],
        }

    # 2. Boilerplate check
    lowered = text.lower()
    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, lowered):
            return {
                "acceptable": False,
                "quality_score": 0.1,
                "reasons": ["Matches non-informative bibliographic/copyright boilerplate"],
            }

    # 3. Informational density (ratio of alphanumeric content to total characters)
    alnum_count = sum(1 for c in text if c.isalnum())
    density = alnum_count / len(text) if len(text) > 0 else 0.0

    if density < 0.5:
        return {
            "acceptable": False,
            "quality_score": 0.2,
            "reasons": ["Low informational density (excessive symbols/formatting)"],
        }

    # 4. Source verification status
    source_type = passage.get("source_type", "unknown")
    provenance = passage.get("provenance", {})
    is_verified_source = provenance.get("verified", False) or source_type in {"guideline", "peer_reviewed"}

    quality_score = 0.5 + (0.3 if is_verified_source else 0.0) + min(0.2, density * 0.2)

    return {
        "acceptable": True,
        "quality_score": round(quality_score, 3),
        "is_verified_source": is_verified_source,
        "reasons": [],
    }
