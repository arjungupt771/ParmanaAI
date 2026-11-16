"""
ParmanaAI - Source Quality, Trust Management, and Duplicate Detection

Tracks document trust levels, publication vintage, duplicate detection,
and prevents equating source hierarchy with absolute clinical certainty.
"""

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Set

from backend.ingestion.base import Document


# Standard clinical evidence hierarchies (Oxford CEBM adapted)
EVIDENCE_HIERARCHY = {
    "guideline": {"level": "A", "weight": 1.0, "description": "Clinical Practice Guideline"},
    "systematic_review": {"level": "A-", "weight": 0.95, "description": "Systematic Review / Meta-Analysis"},
    "rct": {"level": "B", "weight": 0.85, "description": "Randomized Controlled Trial"},
    "observational": {"level": "C", "weight": 0.70, "description": "Cohort or Case-Control Study"},
    "peer_reviewed": {"level": "B-", "weight": 0.80, "description": "Peer-Reviewed Literature"},
    "external_unverified": {"level": "D", "weight": 0.40, "description": "Unverified External Input"},
    "unknown": {"level": "U", "weight": 0.50, "description": "Unspecified Source"},
}


class TrustManager:
    """Manages document trust classification, duplicate detection, and vintage checks."""

    def __init__(self, guideline_expiry_years: int = 5):
        self.guideline_expiry_years = guideline_expiry_years
        self._registered_hashes: Dict[str, str] = {}  # content_hash -> document_id

    def assess_document_trust(self, document: Document) -> Dict[str, Any]:
        """
        Produce source quality assessment without presenting heuristics as clinical certainty.
        """
        source_type = document.source_type.lower() if document.source_type else "unknown"
        hierarchy = EVIDENCE_HIERARCHY.get(source_type, EVIDENCE_HIERARCHY["unknown"])

        # Check vintage / outdated status
        is_potentially_outdated = False
        years_old = None

        if document.publication_date:
            year_match = re.search(r"\b(19\d\d|20\d\d)\b", str(document.publication_date))
            if year_match:
                pub_year = int(year_match.group(1))
                current_year = datetime.now(timezone.utc).year
                years_old = current_year - pub_year
                if years_old > self.guideline_expiry_years:
                    is_potentially_outdated = True

        return {
            "evidence_level": hierarchy["level"],
            "trust_weight": hierarchy["weight"],
            "source_category": hierarchy["description"],
            "is_potentially_outdated": is_potentially_outdated,
            "years_old": years_old,
            "verified_origin": document.provenance.get("verified", False),
            "disclaimer": "Source trust rating reflects publication methodology only; it does not replace individualized clinical judgment.",
        }

    def check_duplicate(self, document: Document) -> Optional[str]:
        """
        Check if an identical document has already been registered.
        Returns existing document_id if duplicate, else None.
        """
        chash = document.content_hash()
        if chash in self._registered_hashes:
            existing_id = self._registered_hashes[chash]
            if existing_id != document.document_id:
                return existing_id
        return None

    def register_hash(self, document: Document) -> None:
        """Register document content hash."""
        self._registered_hashes[document.content_hash()] = document.document_id

    def unregister_hash(self, document: Document) -> None:
        """Unregister document content hash on deletion."""
        self._registered_hashes.pop(document.content_hash(), None)
