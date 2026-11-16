"""
ParmanaAI - Generation Agent Node

Synthesizes evidence-grounded answers using only selected evidence passages,
handles emergency protocols and clinical abstentions.
"""

from typing import Any, Dict, Optional

from backend.agents.multi_agent.agent_base import BaseAgent
from backend.agents.multi_agent.state import SharedAgentState
from backend.agents.generation_agent import GenerationAgent
from backend.agents.refusal_detector import is_refusal


EMERGENCY_ESCALATION_TEXT = (
    "URGENT MEDICAL NOTICE: Your query indicates a potentially life-threatening medical emergency. "
    "Please call 911 (or your local emergency services) or go immediately to the nearest emergency department. "
    "ParmanaAI is an artificial intelligence research system and cannot provide acute emergency clinical triage."
)


class GenerationAgentNode(BaseAgent):
    """
    Coordinates answer generation from validated evidence.
    """

    def __init__(
        self,
        generator: Optional[GenerationAgent] = None,
        name: str = "GenerationAgent",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled)
        self.generator = generator or GenerationAgent()

    def execute(self, state: SharedAgentState) -> SharedAgentState:
        # 1. Emergency & Safety Refusal Protocols
        if state.is_emergency:
            state.generated_answer = EMERGENCY_ESCALATION_TEXT
            state.is_refusal = True
            state.relevance_score = 1.0
            return state

        if state.is_refusal:
            if not state.generated_answer:
                state.generated_answer = "Request declined by clinical safety policy."
            return state

        # 2. Abstention Protocol
        if state.abstained or not state.selected_evidence:
            state.generated_answer = (
                f"No verified medical evidence found to answer this question reliably. "
                f"({state.abstention_reason or 'No qualifying evidence retrieved.'})"
            )
            state.is_refusal = True
            state.relevance_score = 0.0
            return state

        # 3. Evidence-Grounded Prompt Construction
        evidence_text = "\n\n".join([e.get("text", "") for e in state.selected_evidence])

        if state.query_mode == "FACTUAL":
            prompt = f"""
You are a medical assistant.
Use ONLY the guideline information below.
Do NOT add external knowledge.

GUIDELINES:
{evidence_text}

QUESTION:
{state.user_query}

Answer in clear bullet points.
"""
        else:
            prompt = f"""
You are a medical research assistant.
Summarize evidence cautiously from the abstracts below.
Use phrases like "evidence suggests", "is associated with", "may increase risk".
Do NOT define diseases.

ABSTRACTS:
{evidence_text}

QUESTION:
{state.user_query}

Answer in 2-3 sentences.
"""

        gen_output = self.generator.generate(prompt, strict=False)
        answer = gen_output.get("answer", "").strip()

        if is_refusal(answer):
            state.generated_answer = "Unable to answer safely."
            state.is_refusal = True
        else:
            state.generated_answer = answer

        # Calculate relevance score
        state.relevance_score = self.generator.relevance_score(state.user_query, state.generated_answer)
        state.generation_metadata = gen_output

        return state
