"""
ParmanaAI - Input Safety & Prompt Injection Guard

Validates inbound queries for:
1. Malicious prompt injection / jailbreak attempts.
2. Medical emergencies requiring immediate triage.
3. Completely off-topic / non-medical queries.
4. Input formatting anomalies (null bytes, excessive length).
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List, Optional


class SafetyRiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class InputValidationResult:
    is_safe: bool
    risk_level: SafetyRiskLevel
    is_emergency: bool = False
    is_injection: bool = False
    is_out_of_domain: bool = False
    risk_reasons: List[str] = field(default_factory=list)
    sanitized_query: str = ""
    emergency_message: Optional[str] = None


class InputSafetyGuard:
    """
    Validates query safety before execution by the clinical orchestrator.
    """

    # Prompt injection and jailbreak patterns
    INJECTION_PATTERNS = [
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)\b",
        r"(?i)\byou\s+are\s+now\s+(in\s+)?developer\s+mode\b",
        r"(?i)\bDAN\s+mode\b",
        r"(?i)\bdo\s+anything\s+now\b",
        r"(?i)\bprint\s+(your\s+)?(system\s+prompt|initial\s+instructions)\b",
        r"(?i)\breveal\s+(your\s+)?(system\s+prompt|hidden\s+prompt|instructions)\b",
        r"(?i)\bjailbreak\b",
        r"(?i)\bbypass\s+(all\s+)?safety\s+filters\b",
        r"(?i)\bact\s+as\s+an\s+unrestricted\b",
        r"(?i)<\s*system\s*>",
        r"(?i)\[\s*system\s*\]",
    ]

    # Acute life-threatening emergencies requiring immediate referral
    EMERGENCY_PATTERNS = [
        r"(?i)\b(crushing\s+)?chest\s+pain\b",
        r"(?i)\bpain\s+radiating\s+to\s+(left\s+arm|jaw|neck)\b",
        r"(?i)\bsevere\s+(difficulty|trouble)\s+breathing\b",
        r"(?i)\bcannot\s+breathe\b",
        r"(?i)\bchoking\b",
        r"(?i)\banaphylax(is|ic)\b",
        r"(?i)\b(throat|tongue)\s+swelling\b",
        r"(?i)\b(facial\s+droop|face\s+drooping|slurred\s+speech|arm\s+weakness)\b",
        r"(?i)\bsudden\s+numbness\s+on\s+one\s+side\b",
        r"(?i)\bsuicid(e|al)\b",
        r"(?i)\bwant\s+to\s+(kill\s+myself|end\s+my\s+life)\b",
        r"(?i)\bintentional\s+overdose\b",
        r"(?i)\bunconscious(ness)?\b",
        r"(?i)\bseizure\s+lasting\s+more\s+than\s+5\s+minutes\b",
        r"(?i)\bsevere\s+uncontrolled\s+bleeding\b",
    ]

    # Non-medical indicators (strictly non-clinical queries)
    NON_MEDICAL_INDICATORS = [
        r"(?i)\bwrite\s+(a\s+)?(python|java|c\+\+|javascript|sql|code|script)\b",
        r"(?i)\bstock\s+(market|price|trading|investment)\b",
        r"(?i)\bwho\s+won\b.*?\b(game|match|super\s+bowl|world\s+cup|election|fifa)\b",
        r"(?i)\b(world\s+cup|soccer\s+match|super\s+bowl|nfl|fifa)\b",
        r"(?i)\brecipe\s+for\s+(cake|pasta|bread|soup|curry)\b",
        r"(?i)\bbitcoin|crypto|ethereum|blockchain\b",
        r"(?i)\btell\s+me\s+a\s+(funny\s+)?joke\b",
    ]

    # Medical anchor terms to allow queries that mention non-medical words in medical context
    MEDICAL_ANCHORS = [
        r"(?i)\b(patient|dose|dosage|medication|drug|disease|syndrome|therapy|symptom|blood|cell|clinical|treatment|infection|pain|diagnosis|surgery|cancer|diabetes|hypertension|physician|hospital)\b"
    ]

    def __init__(self, max_length: int = 4000, min_length: int = 3):
        self.max_length = max_length
        self.min_length = min_length
        self._compiled_injections = [re.compile(p) for p in self.INJECTION_PATTERNS]
        self._compiled_emergencies = [re.compile(p) for p in self.EMERGENCY_PATTERNS]
        self._compiled_non_medical = [re.compile(p) for p in self.NON_MEDICAL_INDICATORS]
        self._compiled_medical_anchors = [re.compile(p) for p in self.MEDICAL_ANCHORS]

    def validate_input(self, query: str) -> InputValidationResult:
        """
        Validate and sanitize the user query.
        """
        if not query or not query.strip():
            return InputValidationResult(
                is_safe=False,
                risk_level=SafetyRiskLevel.HIGH,
                risk_reasons=["Empty query"],
                sanitized_query="",
            )

        # Sanitize control characters and null bytes
        sanitized = "".join(ch for ch in query if ch == "\n" or ch == "\t" or (32 <= ord(ch) <= 126 or ord(ch) > 127))
        sanitized = re.sub(r"\x00", "", sanitized).strip()

        if len(sanitized) < self.min_length:
            return InputValidationResult(
                is_safe=False,
                risk_level=SafetyRiskLevel.LOW,
                risk_reasons=["Query too short"],
                sanitized_query=sanitized,
            )

        if len(sanitized) > self.max_length:
            return InputValidationResult(
                is_safe=False,
                risk_level=SafetyRiskLevel.HIGH,
                risk_reasons=[f"Query exceeds maximum character limit of {self.max_length}"],
                sanitized_query=sanitized[:self.max_length],
            )

        # 1. Prompt injection / jailbreak check
        for pattern in self._compiled_injections:
            if pattern.search(sanitized):
                return InputValidationResult(
                    is_safe=False,
                    risk_level=SafetyRiskLevel.CRITICAL,
                    is_injection=True,
                    risk_reasons=["Prompt injection or system override pattern detected."],
                    sanitized_query=sanitized,
                )

        # 2. Medical emergency check
        for pattern in self._compiled_emergencies:
            if pattern.search(sanitized):
                return InputValidationResult(
                    is_safe=False,
                    risk_level=SafetyRiskLevel.CRITICAL,
                    is_emergency=True,
                    risk_reasons=["Acute medical emergency detected."],
                    sanitized_query=sanitized,
                    emergency_message=(
                        "EMERGENCY NOTICE: Your query describes potential acute medical symptoms. "
                        "ParmanaAI is an informational clinical research tool and cannot provide immediate emergency medical triage. "
                        "Please call 911 (or your local emergency response number) or go to the nearest emergency room immediately."
                    ),
                )

        # 3. Off-topic check (unless medical anchors are present)
        has_medical_anchor = any(p.search(sanitized) for p in self._compiled_medical_anchors)
        if not has_medical_anchor:
            for pattern in self._compiled_non_medical:
                if pattern.search(sanitized):
                    return InputValidationResult(
                        is_safe=False,
                        risk_level=SafetyRiskLevel.LOW,
                        is_out_of_domain=True,
                        risk_reasons=["Query appears to be outside clinical and medical domains."],
                        sanitized_query=sanitized,
                    )

        return InputValidationResult(
            is_safe=True,
            risk_level=SafetyRiskLevel.SAFE,
            sanitized_query=sanitized,
        )
