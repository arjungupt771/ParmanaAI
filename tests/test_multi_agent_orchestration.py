"""
ParmanaAI - Phase 8 Multi-Agent Orchestration Tests

Tests:
- Query understanding agent (emergency escalation, entity extraction)
- Centralized orchestrator execution graph
- Shared agent state mutation and event logging
- Inter-agent communication and quality control contradiction detection
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.agents.multi_agent.state import SharedAgentState
from backend.agents.multi_agent.query_understanding_agent import QueryUnderstandingAgent
from backend.agents.multi_agent.quality_control_agent import QualityControlAgent
from backend.agents.multi_agent.orchestrator_graph import OrchestratorGraph


def test_query_understanding_agent_emergency_detection():
    agent = QueryUnderstandingAgent()
    state = SharedAgentState(user_query="Patient having severe chest pain and difficulty breathing")
    result_state = agent.run(state)

    assert result_state.is_emergency is True
    assert result_state.query_mode == "EMERGENCY"
    assert result_state.question_type == "emergency"
    assert "QueryUnderstandingAgent" in result_state.agent_latencies_ms


def test_quality_control_agent_contradiction_detection():
    qc_agent = QualityControlAgent()
    state = SharedAgentState(
        user_query="Does Drug X treat Condition Y?",
        generated_answer="Drug X effectively treats Condition Y.",
        verified_claims=[{
            "claim_text": "Drug X effectively treats Condition Y.",
            "combined_verdict": "CONTRADICTED",
            "verdict": "CONTRADICTED"
        }],
        hallucination_score=1.0,
        relevance_score=0.9
    )

    result_state = qc_agent.run(state)
    assert result_state.quality_checks["no_severe_contradictions"] is False
    assert "contradict" in result_state.generated_answer.lower()


def test_orchestrator_graph_emergency_escalation_flow():
    graph = OrchestratorGraph()
    result = graph.run_to_dict("I have severe crushing chest pain")

    assert result["is_emergency"] is True
    assert "911" in result["answer"]
    assert len(result["execution_trace"]) >= 2
    assert "total_pipeline_ms" in result["agent_latencies_ms"]


def test_orchestrator_graph_factual_flow_with_mocks():
    # Mock generation agent to avoid live Groq API calls in test
    class MockGenNode(MagicMock):
        def run(self, state):
            state.generated_answer = "Metformin lowers blood glucose in type 2 diabetes."
            state.relevance_score = 0.95
            state.log_event("GenerationAgent", "COMPLETED", {})
            return state

    # Mock verification agent
    class MockVerNode(MagicMock):
        def run(self, state):
            state.verified_claims = [{
                "claim_id": "CLM-0001",
                "claim_text": "Metformin lowers blood glucose in type 2 diabetes.",
                "verdict": "SUPPORTED",
                "combined_verdict": "SUPPORTED",
                "primary_evidence_id": "ev-1",
            }]
            state.hallucination_score = 0.0
            return state

    graph = OrchestratorGraph(
        generation_agent=MockGenNode(),
        verification_agent=MockVerNode(),
    )

    result = graph.run_to_dict(
        user_query="How does metformin work in type 2 diabetes?",
        external_context="Metformin reduces hepatic glucose production in type 2 diabetes patients."
    )

    assert result["is_emergency"] is False
    assert "Metformin" in result["answer"]
    assert len(result["claims"]) == 1
    assert result["claims"][0]["verdict"] == "SUPPORTED"
    assert result["hallucination_score"] == 0.0
    assert len(result["evidence"]) >= 1
