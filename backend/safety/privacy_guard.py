"""
ParmanaAI - Clinical Privacy & PII De-identification Guard

Detects and redacts Protected Health Information (PHI) and Personally Identifiable
Information (PII) under HIPAA Safe Harbor guidelines before storage or external LLM inference.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Tuple


@dataclass
class RedactionEvent:
    entity_type: str
    original_snippet: str
    replacement: str
    start_pos: int
    end_pos: int


class PrivacyGuard:
    """
    HIPAA Safe Harbor-aligned scrubber for clinical notes and user queries.
    """

    # Regex patterns for HIPAA identifiers
    PATTERNS: Dict[str, str] = {
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "MRN": r"(?i)\b(mrn|medical\s+record(?:\s+number|\s+#)?|patient\s+id)[\s:#=]*([A-Za-z0-9\-]{5,15})\b",
        "PHONE": r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "DOB": r"(?i)\b(?:dob|date\s+of\s+birth|born(?:\s+on)?)[\s:#=]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        "PATIENT_NAME": r"\b(?:[Pp]atient(?:\s+[Nn]ame)?|[Pp]t)[\s]*[:#=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
    }

    def __init__(self):
        self._compiled_patterns = {
            k: re.compile(v) for k, v in self.PATTERNS.items()
        }

    def contains_pii(self, text: str) -> bool:
        """
        Check if text contains any detected PII/PHI.
        """
        if not text:
            return False
        return any(regex.search(text) is not None for regex in self._compiled_patterns.values())

    def scrub_pii(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Scrub and replace identified PII/PHI with standardized placeholder tokens.
        Returns the sanitized text and a list of redactions performed.
        """
        if not text:
            return "", []

        scrubbed = text
        redactions = []

        # 1. Scrub SSN
        for match in self._compiled_patterns["SSN"].finditer(text):
            val = match.group(0)
            token = "[REDACTED_SSN]"
            scrubbed = scrubbed.replace(val, token)
            redactions.append({"type": "SSN", "token": token})

        # 2. Scrub MRN
        for match in self._compiled_patterns["MRN"].finditer(text):
            full_match = match.group(0)
            id_val = match.group(2)
            token = f"{match.group(1)}: [REDACTED_MRN]"
            scrubbed = scrubbed.replace(full_match, token)
            redactions.append({"type": "MRN", "token": "[REDACTED_MRN]"})

        # 3. Scrub DOB
        for match in self._compiled_patterns["DOB"].finditer(text):
            full_match = match.group(0)
            token = "[REDACTED_DOB]"
            scrubbed = scrubbed.replace(full_match, token)
            redactions.append({"type": "DOB", "token": token})

        # 4. Scrub Patient Name
        for match in self._compiled_patterns["PATIENT_NAME"].finditer(text):
            full_match = match.group(0)
            token = "Patient: [REDACTED_NAME]"
            scrubbed = scrubbed.replace(full_match, token)
            redactions.append({"type": "NAME", "token": "[REDACTED_NAME]"})

        # 5. Scrub Email
        for match in self._compiled_patterns["EMAIL"].finditer(scrubbed):
            val = match.group(0)
            token = "[REDACTED_EMAIL]"
            scrubbed = scrubbed.replace(val, token)
            redactions.append({"type": "EMAIL", "token": token})

        # 6. Scrub Phone numbers
        for match in self._compiled_patterns["PHONE"].finditer(scrubbed):
            val = match.group(0)
            token = "[REDACTED_PHONE]"
            scrubbed = scrubbed.replace(val, token)
            redactions.append({"type": "PHONE", "token": token})

        return scrubbed, redactions
