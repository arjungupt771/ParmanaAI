# backend/evaluation/claim_severity.py

CLAIM_SEVERITY = {
    "medical": 1.0,
    "legal": 1.0,
    "financial": 0.8,
    "scientific": 0.7,
    "historical": 0.5,
    "general": 0.3,
    "opinion": 0.1
}

def get_claim_severity(claim_type: str) -> float:
    return CLAIM_SEVERITY.get(claim_type, 0.3)
