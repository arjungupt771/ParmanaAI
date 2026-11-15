# backend/evaluation/evidence_scoring.py

SOURCE_CONFIDENCE = {
    "peer_reviewed": 1.0,
    "government": 0.9,
    "wikipedia": 0.7,
    "news": 0.6,
    "blog": 0.3,
    "llm": 0.2
}

def source_confidence(source_type: str) -> float:
    return SOURCE_CONFIDENCE.get(source_type, 0.3)


def cross_source_agreement(supporting: int, total: int) -> float:
    if total == 0:
        return 0.0
    return supporting / total


def citation_reliability(match_type: str) -> float:
    mapping = {
        "exact": 1.0,
        "semantic": 0.7,
        "weak": 0.4,
        "missing": 0.0
    }
    return mapping.get(match_type, 0.0)


def compute_ess(source_type, supporting, total, match_type,
                alpha=0.4, beta=0.3, gamma=0.3):
    sc = source_confidence(source_type)
    csa = cross_source_agreement(supporting, total)
    cr = citation_reliability(match_type)

    return alpha * sc + beta * csa + gamma * cr
