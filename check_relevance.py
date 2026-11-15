# check_relevance.py
import json
with open("backend/evaluation/results.json") as f:
    results = json.load(f)
print("relevance_score present:", "relevance_score" in results[0])
print("sample value:", results[0].get("relevance_score"))