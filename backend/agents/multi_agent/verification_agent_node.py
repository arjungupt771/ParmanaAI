"""
ParmanaAI - Verification Agent Node

Extracts atomic claims and executes hybrid NLI, numerical consistency,
and multi-source aggregation verification.
"""

from typing import Any, Dict, List

from backend.agents.multi_agent.agent_base import BaseAgent
from backend.agents.multi_agent.state import SharedAgentState
from backend.agents.extractor import extract_claim_records
from backend.agents.verifier import verify_claim, compute_hallucination_score


class VerificationAgentNode(BaseAgent):
    """
    Coordinates atomic claim verification and hallucination scoring.
    """

    def __init__(self, name: str = "VerificationAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    def execute(self, state: SharedAgentState) -> SharedAgentState:
        # If refusal or emergency or empty answer, skip verification
        if state.is_refusal or not state.generated_answer or not state.selected_evidence:
            state.verified_claims = []
            state.hallucination_score = 0.0 if (state.is_emergency or state.is_refusal) else 1.0
            return state

        # 1. Extract atomic claim records
        claim_records = extract_claim_records(
            state.generated_answer,
            inferential=(state.query_mode == "INFERENTIAL"),
        )
        state.claim_records = claim_records

        # 2. Verify each claim
        verified_results = []
        for record in claim_records:
            claim_text = record["claim_text"]
            res = verify_claim(claim_text, state.selected_evidence)

            # Enrich with extraction metadata
            res.update({
                "claim_id": record["claim_id"],
                "extraction_type": record["claim_type"],
                "is_inferential": record["is_inferential"],
                "is_definitive": record["is_definitive"],
                "contains_causal_language": record["contains_causal_language"],
                "contains_negation": record.get("contains_negation", False),
                "contains_comparative_language": record.get("contains_comparative_language", False),
                "relationship_type": record.get("relationship_type", "factual"),
                "contains_numerical_information": record["contains_numerical_information"],
                "numerical_values": record["numerical_values"],
                "numerical_mentions": record["numerical_mentions"],
                "original_sentence": record["original_sentence"],
                "atomic": record["atomic"],
            })
            verified_results.append(res)

        state.verified_claims = verified_results
        state.hallucination_score = compute_hallucination_score(verified_results)

        return state
