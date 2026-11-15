import os
import sys

# Ensure project root is added
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.generation_agent import GenerationAgent
from backend.agents.extractor import extract_claims
from backend.agents.verifier import verify_claim, compute_hallucination_score


def run_baseline(query: str):
    generator = GenerationAgent()

    print(f"\nRunning baseline for query: {query}")

    # Generate WITHOUT evidence gating
    output = generator.generate(query, strict=False)
    answer = output.get("answer", "")

    print("\nGenerated Answer:")
    print(answer)

    # Extract claims
    claims = extract_claims(answer)

    print("\nExtracted Claims:")
    print(claims)

    # No external evidence provided (baseline!)
    evidence_list = []

    verified_claims = [
        verify_claim(c, evidence_list)
        for c in claims
    ]

    score = compute_hallucination_score(verified_claims)

    print("\nVerified Claims:")
    print(verified_claims)

    print(f"\nBaseline Hallucination Score: {score}")

    return score


# -----------------------------
# Run directly
# -----------------------------
if __name__ == "__main__":
    test_query = "Does diabetes increase cardiovascular risk?"
    run_baseline(test_query)