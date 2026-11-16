"""
ParmanaAI - Clinical Refusal & Controlled Abstention Engine

Provides conservative refusal and controlled abstention policies when:
1. Retrieved evidence is insufficient or empty.
2. Retrieved evidence fails topic relevance checks (topic mismatch).
3. The query attempts prompt injection or safety bypass.
4. Prescriptive individual dosing is requested without physician oversight.
5. Inconclusive or conflicting clinical studies prevent definitive advice.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RefusalReason(str, Enum):
    PROMPT_INJECTION = "PROMPT_INJECTION"
    EMERGENCY_TRIAGE = "EMERGENCY_TRIAGE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    TOPIC_MISMATCH = "TOPIC_MISMATCH"
    PRESCRIPTIVE_DOSING = "PRESCRIPTIVE_DOSING"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    SEVERE_CONTRADICTION = "SEVERE_CONTRADICTION"
    NONE = "NONE"


@dataclass
class RefusalDecision:
    should_refuse: bool
    reason: RefusalReason
    refusal_message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RefusalEngine:
    """
    Evaluates evidence sufficiency, topic alignment, and safety bounds
    to determine whether the system should safely abstain.
    """

    STANDARD_MESSAGES = {
        RefusalReason.PROMPT_INJECTION: (
            "Request rejected: The provided prompt contains instruction override or jailbreak patterns "
            "that violate safety protocols."
        ),
        RefusalReason.EMERGENCY_TRIAGE: (
            "EMERGENCY NOTICE: Your inquiry indicates potential acute medical symptoms. "
            "ParmanaAI is an informational clinical research tool and cannot provide emergency medical triage. "
            "Please call 911 or proceed to the nearest emergency medical facility immediately."
        ),
        RefusalReason.INSUFFICIENT_EVIDENCE: (
            "I do not have sufficient peer-reviewed clinical evidence in the indexed knowledge base to reliably "
            "answer this question. Please consult a licensed healthcare professional or verified clinical guidelines."
        ),
        RefusalReason.TOPIC_MISMATCH: (
            "The retrieved medical evidence does not adequately match the specific clinical topic of your query. "
            "To prevent medical misinformation, ParmanaAI has abstained from generating a conclusion."
        ),
        RefusalReason.PRESCRIPTIVE_DOSING: (
            "ParmanaAI cannot recommend individual patient prescription dosages or treatment modifications without "
            "direct verified physician oversight. Please consult the treating physician or a licensed clinical pharmacist."
        ),
        RefusalReason.OUT_OF_DOMAIN: (
            "ParmanaAI is a clinical knowledge and biomedical evidence verification assistant. "
            "Your query falls outside biomedical and healthcare domains."
        ),
        RefusalReason.SEVERE_CONTRADICTION: (
            "Identified medical sources present directly conflicting clinical evidence on this topic. "
            "A definitive clinical conclusion cannot be reached without further primary literature review."
        ),
    }

    def evaluate_pre_retrieval(
        self,
        is_safe: bool,
        is_injection: bool = False,
        is_emergency: bool = False,
        is_out_of_domain: bool = False,
        emergency_message: Optional[str] = None,
    ) -> RefusalDecision:
        """
        Evaluate refusal prior to retrieval based on input safety validation.
        """
        if is_injection:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.PROMPT_INJECTION,
                refusal_message=self.STANDARD_MESSAGES[RefusalReason.PROMPT_INJECTION],
            )

        if is_emergency:
            msg = emergency_message or self.STANDARD_MESSAGES[RefusalReason.EMERGENCY_TRIAGE]
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.EMERGENCY_TRIAGE,
                refusal_message=msg,
            )

        if is_out_of_domain:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.OUT_OF_DOMAIN,
                refusal_message=self.STANDARD_MESSAGES[RefusalReason.OUT_OF_DOMAIN],
            )

        return RefusalDecision(
            should_refuse=False,
            reason=RefusalReason.NONE,
            refusal_message="",
        )

    def evaluate_post_retrieval(
        self,
        evidence_items: List[Dict[str, Any]],
        topic_aligned: bool = True,
        relevance_score: float = 1.0,
        min_relevance_threshold: float = 0.25,
    ) -> RefusalDecision:
        """
        Evaluate refusal after evidence retrieval and filtering.
        """
        if not evidence_items or len(evidence_items) == 0:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.INSUFFICIENT_EVIDENCE,
                refusal_message=self.STANDARD_MESSAGES[RefusalReason.INSUFFICIENT_EVIDENCE],
                metadata={"evidence_count": 0},
            )

        if not topic_aligned or relevance_score < min_relevance_threshold:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.TOPIC_MISMATCH,
                refusal_message=self.STANDARD_MESSAGES[RefusalReason.TOPIC_MISMATCH],
                metadata={"relevance_score": relevance_score, "topic_aligned": topic_aligned},
            )

        return RefusalDecision(
            should_refuse=False,
            reason=RefusalReason.NONE,
            refusal_message="",
            metadata={"evidence_count": len(evidence_items), "relevance_score": relevance_score},
        )
