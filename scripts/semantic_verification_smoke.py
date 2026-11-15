
from backend.agents.verifier import verify_claim

evidence = [
    {
        "source_type": "guideline",
        "text": (
            "Fever commonly presents with chills, "
            "sweating, headache, and muscle pain."
        ),
        "match_type": "semantic"
    }
]

test_claims = [
    "Fever causes chills and sweating",
    "An elevated body temperature is often accompanied by shivering and perspiration",
    "The patient has a broken leg"
]

for claim in test_claims:

    result = verify_claim(claim, evidence)

    print(f"\nClaim: {claim}")
    print(f"Verdict: {result['verdict']}")
    print(f"Method: {result['verification_method_used']}")
    print(f"NLI scores: {result['nli_scores']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Explanation: {result['explanation']}")
