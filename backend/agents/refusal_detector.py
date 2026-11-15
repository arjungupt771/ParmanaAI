import re

REFUSAL_PATTERNS = [
    r"\bi am unable\b",
    r"\bi'?m unable\b",
    r"\bi cannot provide\b",
    r"\bi can'?t provide\b",
    r"\bi do not have enough information\b",
    r"\bi don'?t have enough information\b",
    r"\bno reliable evidence-based answer\b",
    r"\bunable to answer\b",
    r"\bcannot answer this question\b",
    r"\bi am not able to\b",
    r"\bi'?m not able to\b",
    r"\bi cannot determine\b",
    r"\bi can'?t determine\b",
]

_COMPILED = [re.compile(p) for p in REFUSAL_PATTERNS]


def is_refusal(answer: str) -> bool:
    """
    Detect whether a generated answer is a refusal / non-answer, rather than
    relying on a single startswith() check against one hardcoded string.
    Matches phrases anywhere in the answer (not just the start), since
    models sometimes preface a refusal with a sentence of context first.
    An empty or very short answer is also treated as a refusal, since it
    likely indicates a generation/parsing failure rather than a real answer.
    """
    if not answer or len(answer.strip()) < 15:
        return True

    lowered = answer.lower()
    return any(pattern.search(lowered) for pattern in _COMPILED)