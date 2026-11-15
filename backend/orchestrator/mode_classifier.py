def classify_query(query: str) -> str:
    q = query.lower()

    inferential_triggers = [
        "does",
        "increase",
        "decrease",
        "risk",
        "association",
        "effect",
        "impact",
        "evidence",
        "study",
        "studies",
        "linked",
        "correlated",
    ]

    if any(k in q for k in inferential_triggers):
        return "INFERENTIAL"

    return "FACTUAL"
