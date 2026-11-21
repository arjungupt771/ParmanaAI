"""
ParmanaAI - Question-Answer Alignment Engine

Evaluates whether a generated clinical answer directly and accurately
addresses the specific medical question asked, distinguishing direct
factual answers from tangential, evasive, or unsupported definitional prose.
"""

import re
from typing import Any, Dict, List, Set

COMMON_STOPWORDS: Set[str] = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be",
    "because", "before", "between", "both", "but", "by", "can", "could",
    "describe", "did", "discuss", "do", "does", "explain", "for", "from",
    "give", "had", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "list", "me", "more", "my", "no", "not", "of", "on",
    "or", "our", "please", "provide", "show", "so", "some", "tell", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "through", "to", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your"
}

NUMERICAL_INTENT_KEYWORDS: Set[str] = {
    "cutoff", "cutoffs", "threshold", "thresholds", "level", "levels",
    "range", "ranges", "target", "targets", "dose", "dosage", "dosages",
    "percentage", "percent", "rate", "rates", "value", "values", "numbers",
    "ratio", "interval", "intervals", "normal", "abnormal", "measurement",
    "units", "frequency", "prevalence", "incidence"
}

NUMERICAL_REGEX = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*%|\s*mg|\s*mcg|\s*g|\s*kg|\s*ml|\s*mmol/l|\s*mg/dl|\s*mmhg)?\b",
    re.IGNORECASE
)


def extract_content_tokens(text: str) -> Set[str]:
    """
    Extract meaningful content words (excluding stopwords and single letters).
    """
    if not isinstance(text, str):
        return set()
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", text.lower())
    return {w for w in words if w not in COMMON_STOPWORDS and len(w) > 1}


def has_numerical_intent(query: str) -> bool:
    """
    Detect whether the user query specifically asks for numerical values,
    ranges, cutoffs, dosages, or thresholds.
    """
    query_tokens = extract_content_tokens(query)
    return bool(query_tokens & NUMERICAL_INTENT_KEYWORDS) or bool(re.search(r"\b\d+\b", query))


def contains_numerical_content(answer: str) -> bool:
    """
    Detect whether the answer contains explicit numbers or quantitative measurements.
    """
    return bool(NUMERICAL_REGEX.search(answer))


def token_matches(q_tok: str, a_tokens: Set[str]) -> bool:
    """
    Check exact match or morphological stem match for medical variations
    (e.g., diagnosing -> diagnosed, cutoffs -> cutoff, diabetes -> diabetic).
    """
    if q_tok in a_tokens:
        return True
    if len(q_tok) >= 5:
        stem = q_tok[:5]
        return any(a.startswith(stem) or (len(a) >= 5 and a[:5] == stem) for a in a_tokens)
    return False


def compute_qa_alignment(query: str, answer: str) -> Dict[str, Any]:
    """
    Compute structured question-answer alignment metrics.

    Returns:
    - score: Overall alignment score [0.0, 1.0]
    - entity_recall: Proportion of query content tokens present in answer [0.0, 1.0]
    - numerical_intent: Whether the query requested numbers
    - numerical_satisfied: Whether numerical intent was satisfied in the answer
    - is_aligned: True if score >= 0.35 and numerical requirements met
    - missing_tokens: Query content tokens absent from the answer
    """
    if not query or not answer:
        return {
            "score": 0.0,
            "entity_recall": 0.0,
            "numerical_intent": False,
            "numerical_satisfied": True,
            "is_aligned": False,
            "missing_tokens": []
        }

    q_tokens = extract_content_tokens(query)
    a_tokens = extract_content_tokens(answer)

    if not q_tokens:
        return {
            "score": 0.5,
            "entity_recall": 1.0,
            "numerical_intent": False,
            "numerical_satisfied": True,
            "is_aligned": True,
            "missing_tokens": []
        }

    matched_tokens = {q for q in q_tokens if token_matches(q, a_tokens)}
    missing = sorted(list(q_tokens - matched_tokens))
    entity_recall = len(matched_tokens) / len(q_tokens)

    # Assess numerical intent
    requires_numbers = has_numerical_intent(query)
    has_numbers = contains_numerical_content(answer)
    numerical_satisfied = True

    if requires_numbers and not has_numbers:
        numerical_satisfied = False
        # Heavy penalty if query specifically asked for cutoffs/thresholds/values
        alignment_score = round(max(0.0, entity_recall * 0.4), 3)
    else:
        # Full alignment score based on entity recall and topical overlap
        alignment_score = round(min(1.0, entity_recall), 3)

    is_aligned = (alignment_score >= 0.35) and numerical_satisfied

    return {
        "score": alignment_score,
        "entity_recall": round(entity_recall, 3),
        "numerical_intent": requires_numbers,
        "numerical_satisfied": numerical_satisfied,
        "is_aligned": is_aligned,
        "missing_tokens": missing
    }
