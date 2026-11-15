"""
Diagnostic: inspect why claims are landing in each verdict bucket.
Run after evaluate_system.py, against the results.json it produces.

    python -m backend.evaluation.diagnose_verdicts
"""
import json

RESULTS_PATH = "backend/evaluation/results.json"

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

no_claim_samples = [r for r in results if len(r.get("verified_claims", [])) == 0]
print(f"Samples with 0 extracted claims: {len(no_claim_samples)} / {len(results)}")

insufficient = []
for r in results:
    for c in r.get("verified_claims", []):
        if c.get("verdict") == "INSUFFICIENT_EVIDENCE":
            insufficient.append((r["question"], c))

print(f"\nTotal INSUFFICIENT_EVIDENCE claims: {len(insufficient)}")
print("\n--- Sample of 5 ---\n")

for question, c in insufficient[:5]:
    print("Q:", question[:100])
    print("Claim:", c.get("claim_text"))
    print("Overlap ratio:", c.get("overlap_ratio"))
    print("Citation used:", (c.get("citation") or "")[:150])
    print("-" * 60)