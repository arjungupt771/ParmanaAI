"""
ParmanaAI - Query Understanding Agent Node

Analyzes user queries: performs privacy/PII scrubbing, safety & prompt injection validation,
mode classification, question type identification, clinical entity extraction,
controlled query expansion, and emergency detection.
"""

from typing import Any, Dict, Optional

from backend.agents.multi_agent.agent_base import BaseAgent
from backend.agents.multi_agent.state import SharedAgentState
from backend.orchestrator.mode_classifier import classify_query
from backend.retrieval.query_expander import expand_query
from backend.safety.input_safety import InputSafetyGuard, SafetyRiskLevel
from backend.safety.privacy_guard import PrivacyGuard
from backend.safety.refusal_engine import RefusalEngine, RefusalReason


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent responsible for clinical question understanding, safety verification,
    PII de-identification, and query expansion.
    """

    def __init__(
        self,
        name: str = "QueryUnderstandingAgent",
        enabled: bool = True,
        safety_guard: Optional[InputSafetyGuard] = None,
        privacy_guard: Optional[PrivacyGuard] = None,
        refusal_engine: Optional[RefusalEngine] = None,
    ):
        super().__init__(name=name, enabled=enabled)
        self.safety_guard = safety_guard or InputSafetyGuard()
        self.privacy_guard = privacy_guard or PrivacyGuard()
        self.refusal_engine = refusal_engine or RefusalEngine()

    def execute(self, state: SharedAgentState) -> SharedAgentState:
        raw_query = state.user_query.strip()

        # 1. PII/PHI De-identification (HIPAA Safe Harbor)
        scrubbed_query, redactions = self.privacy_guard.scrub_pii(raw_query)
        if redactions:
            state.log_event(self.name, "PII_REDACTED", {"redaction_count": len(redactions), "types": [r["type"] for r in redactions]})
            state.user_query = scrubbed_query

        # 2. Input Safety & Prompt Injection Guard
        validation = self.safety_guard.validate_input(scrubbed_query)
        if not validation.is_safe:
            state.log_event(self.name, "SAFETY_RISK_DETECTED", {
                "risk_level": validation.risk_level.value,
                "reasons": validation.risk_reasons,
                "is_emergency": validation.is_emergency,
                "is_injection": validation.is_injection,
                "is_out_of_domain": validation.is_out_of_domain,
            })

            pre_refusal = self.refusal_engine.evaluate_pre_retrieval(
                is_safe=validation.is_safe,
                is_injection=validation.is_injection,
                is_emergency=validation.is_emergency,
                is_out_of_domain=validation.is_out_of_domain,
                emergency_message=validation.emergency_message,
            )

            if validation.is_emergency:
                state.is_emergency = True
                state.query_mode = "EMERGENCY"
                state.question_type = "emergency"

            state.is_refusal = True
            state.refusal_reason = pre_refusal.reason.value
            state.generated_answer = pre_refusal.refusal_message
            return state

        # 3. Mode Classification (Factual vs Inferential)
        state.query_mode = classify_query(scrubbed_query)

        # 4. Question Type Identification
        lowered = scrubbed_query.lower()
        if any(w in lowered for w in ("more than", "better than", "compared to", "vs", "versus")):
            state.question_type = "comparative"
        elif any(w in lowered for w in ("what dose", "how much", "target", "mmhg", "mg/dl", "%", "bpm", "threshold")):
            state.question_type = "numerical"
        elif state.query_mode == "INFERENTIAL":
            state.question_type = "inferential"
        else:
            state.question_type = "factual"

        # 5. Entity Extraction & Controlled Expansion
        expansion_info = expand_query(scrubbed_query)
        state.entities = expansion_info.get("entities", {})
        state.expanded_query = expansion_info.get("expanded_query", scrubbed_query)

        # 6. Ambiguity Detection (e.g. fewer than 3 words)
        words = scrubbed_query.split()
        if len(words) < 3 and not state.is_emergency:
            state.is_ambiguous = True

        return state
