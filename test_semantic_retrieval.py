# test_semantic_retrieval.py
from backend.retrieval.semantic_retriever import retrieve

test_cases = [
    ("what causes high body temperature", "guideline"),
    ("danger signs to watch for with a temperature", "guideline"),
    ("blood sugar control disorder", "guideline"),
    ("insulin resistance and heart disease", "peer_reviewed"),
    ("does mars have moons", "guideline"),  # should find nothing relevant
]

for query, source_filter in test_cases:
    print(f"Query: '{query}'  |  filter: {source_filter}")
    results = retrieve(query, source_filter=source_filter, top_k=3)
    if not results:
        print("  -> No results above threshold.")
    else:
        for r in results:
            print(f"  [{r['match_type']} | score={r['score']}] {r['text'][:100]}...")
    print()