import json
import os
from difflib import SequenceMatcher

# -------------------------------------------------
# Resolve project root safely
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# -------------------------------------------------
# EXACT PubMedQA path (CONFIRMED)
# -------------------------------------------------
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "evaluation",
    "pubmedqa.json"
)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"PubMedQA file not found at: {DATA_PATH}")

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    PUBMEDQA = json.load(f)

# -------------------------------------------------
# Simple similarity-based retriever (NO OpenAI, NO embeddings)
# -------------------------------------------------
def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def retrieve_pubmedqa_context(query: str, threshold: float = 0.4):
    best_item = None
    best_score = 0.0

    for item in PUBMEDQA:
        question = item.get("question") or item.get("QUESTION")
        if not question:
            continue

        score = _similarity(query, question)
        if score > best_score:
            best_score = score
            best_item = item

    if best_score < threshold:
        return None

    # PubMedQA usually stores contexts like this
    return (
        best_item.get("contexts")
        or best_item.get("CONTEXTS")
        or best_item.get("context")
    )
