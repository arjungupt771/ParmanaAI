"""
ParmanaAI - Question-Answer Alignment Tests

Tests:
- Extraction of query content tokens and entity recall
- Numerical intent detection (cutoffs, dosages, percentages, thresholds)
- Penalty for generic evasive answers when numerical values are expected
- Integration with GenerationAgent and GenerationAgentNode
"""

import pytest
from backend.agents.qa_alignment import (
    compute_qa_alignment,
    extract_content_tokens,
    has_numerical_intent,
    contains_numerical_content,
)
from backend.agents.generation_agent import GenerationAgent
from backend.agents.multi_agent.generation_agent_node import GenerationAgentNode
from backend.agents.multi_agent.state import SharedAgentState


def test_extract_content_tokens_removes_stopwords():
    query = "What are the blood pressure cutoffs for diagnosing hypertension in elderly adults?"
    tokens = extract_content_tokens(query)

    assert "what" not in tokens
    assert "are" not in tokens
    assert "the" not in tokens
    assert "for" not in tokens
    assert "in" not in tokens

    assert "blood" in tokens
    assert "pressure" in tokens
    assert "cutoffs" in tokens
    assert "diagnosing" in tokens
    assert "hypertension" in tokens
    assert "elderly" in tokens
    assert "adults" in tokens


def test_numerical_intent_detection():
    assert has_numerical_intent("What is the fasting blood glucose cutoff for diabetes?") is True
    assert has_numerical_intent("What dosage of metformin is recommended?") is True
    assert has_numerical_intent("What percentage reduction in LDL is achieved by atorvastatin 20mg?") is True
    assert has_numerical_intent("What are the risk factors for hypertension?") is False


def test_qa_alignment_penalizes_missing_numbers_when_expected():
    query = "What is the fasting blood glucose cutoff for diagnosing type 2 diabetes?"

    # Generic evasive answer without numbers
    generic_answer = "Diabetes is a chronic metabolic disease where blood sugar levels are elevated."
    align_generic = compute_qa_alignment(query, generic_answer)

    assert align_generic["numerical_intent"] is True
    assert align_generic["numerical_satisfied"] is False
    assert align_generic["score"] < 0.35
    assert align_generic["is_aligned"] is False

    # Specific answer with numbers
    direct_answer = "According to ADA guidelines, type 2 diabetes is diagnosed with a fasting plasma glucose >= 126 mg/dL."
    align_direct = compute_qa_alignment(query, direct_answer)

    assert align_direct["numerical_intent"] is True
    assert align_direct["numerical_satisfied"] is True
    assert align_direct["score"] >= 0.60
    assert align_direct["is_aligned"] is True


def test_generation_agent_qa_alignment_integration():
    agent = GenerationAgent()

    # Query with cutoffs
    q = "What are the blood pressure cutoffs for diagnosing hypertension?"
    result = agent.generate(q, strict=True)
    answer = result["answer"]

    # Verify deterministic fallback answers the question directly with cutoffs
    assert "130" in answer or "80" in answer
    assert "hypertension" in answer.lower()

    score = agent.relevance_score(q, answer)
    assert score >= 0.50

    alignment = agent.compute_alignment(q, answer)
    assert alignment["is_aligned"] is True
    assert alignment["numerical_satisfied"] is True


def test_generation_agent_node_alignment_metadata():
    node = GenerationAgentNode()
    state = SharedAgentState(
        user_query="What are the diagnostic blood pressure thresholds?",
        selected_evidence=[{
            "id": "ev-1",
            "text": "Hypertension is defined as blood pressure >= 130/80 mmHg."
        }],
        query_mode="FACTUAL"
    )

    out_state = node.execute(state)
    assert out_state.relevance_score > 0.0
    assert "qa_alignment" in out_state.generation_metadata
    assert out_state.generation_metadata["qa_alignment"]["score"] > 0.0
