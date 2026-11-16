import re
from typing import Any
from backend.agents.numerical_extractor import (
    extract_numerical_mentions
)

# ---------------------------------------------------------
# 1. LANGUAGE INDICATORS
# ---------------------------------------------------------

DEFINITIVE_PATTERNS = [
    r"\bis\b",
    r"\bare\b",
    r"\bwas\b",
    r"\bwere\b",
    r"\bcauses?\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bincreases?\b",
    r"\breduces?\b",
    r"\bprevents?\b",
    r"\bprotects?\b",
    r"\bwill\b",
    r"\bmust\b",
]

SOFT_PATTERNS = [
    r"\bsuggests?\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bpossibly\b",
    r"\bpotentially\b",
    r"\bassociated with\b",
    r"\bassociated\b",
    r"\bevidence indicates\b",
    r"\bstudies show\b",
    r"\bappears to\b",
    r"\bseems to\b",
    r"\bincrease(?:d)? risk\b",
    r"\bdecrease(?:d)? risk\b",
]

CAUSAL_PATTERNS = [
    r"\bcauses?\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bcontributes? to\b",
    r"\bprevents?\b",
]

NEGATION_PATTERNS = [
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bdid not\b",
    r"\bis not\b",
    r"\bare not\b",
    r"\bwas not\b",
    r"\bwere not\b",
    r"\bcannot\b",
    r"\bno evidence\b",
    r"\bfailed to\b",
    r"\bwithout\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bno significant\b",
    r"\bnot associated\b",
]

COMPARATIVE_PATTERNS = [
    r"\bmore effective than\b",
    r"\bless effective than\b",
    r"\bhigher than\b",
    r"\blower than\b",
    r"\bgreater than\b",
    r"\breduced compared to\b",
    r"\bincreased compared to\b",
    r"\bsuperior to\b",
    r"\binferior to\b",
    r"\bcompared with\b",
    r"\bcompared to\b",
    r"\bmore likely\b",
    r"\bless likely\b",
    r"\bversus\b",
    r"\bvs\.?\b",
]

NUMERICAL_PATTERN = re.compile(
    r"""
    \b\d+(?:\.\d+)?\s*%
    |
    \b\d+(?:\.\d+)?\s*
    (?:mg|mcg|μg|g|kg|ml|mL|mmol/L|mg/dL|mmHg)
    \b
    |
    \b\d+(?:\.\d+)?\s*
    (?:years?|months?|weeks?|days?)
    \b
    |
    \b(?:OR|RR|HR|CI)\s*[:=]?\s*
    \d+(?:\.\d+)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------
# 2. TEXT NORMALIZATION
# ---------------------------------------------------------

def clean_markdown(text: str) -> str:
    """
    Remove presentation-only Markdown while preserving content.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n")

    # Remove code fences but retain their contents.
    text = re.sub(r"```[^\n]*\n?", "", text)
    text = text.replace("```", "")

    # Remove bold and italic markers.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*", r"\1", text)
    text = re.sub(r"(?<!_)_(?!_)(.*?)_", r"\1", text)

    # Remove Markdown headings.
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Preserve list content, remove list markers.
    text = re.sub(
        r"^\s*(?:[-*+]|\d+[.)])\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove inline links but preserve link labels.
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# 3. SENTENCE SEGMENTATION
# ---------------------------------------------------------

ABBREVIATIONS = [
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
    "e.g.", "i.e.", "vs.", "etc.",
    "Fig.", "No."
]


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences while protecting common
    abbreviations, decimal numbers and statistical values.
    """

    if not text:
        return []

    protected = text

    # Protect known abbreviations.
    for index, abbreviation in enumerate(ABBREVIATIONS):
        placeholder = f"ABBREVIATIONTOKEN{index}END"
        protected = protected.replace(abbreviation, placeholder)

    # Protect decimal points.
    protected = re.sub(
        r"(?<=\d)\.(?=\d)",
        "DECIMALPOINTTOKEN",
        protected
    )

    # Split on sentence-ending punctuation.
    parts = re.split(
        r'(?<=[.!?])(?:["\']?)(?=\s+|$)',
        protected
    )

    sentences = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # Restore protected abbreviations.
        for index, abbreviation in enumerate(ABBREVIATIONS):
            placeholder = f"ABBREVIATIONTOKEN{index}END"
            part = part.replace(placeholder, abbreviation)

        part = part.replace(
            "DECIMALPOINTTOKEN",
            "."
        )

        sentences.append(part.strip())

    return sentences


# ---------------------------------------------------------
# 4. CLAIM VALIDATION
# ---------------------------------------------------------

def is_meaningful_claim(text: str) -> bool:
    """
    Filter headings, fragments and very short statements.
    """

    if not isinstance(text, str):
        return False

    words = re.findall(r"\b[A-Za-z]{2,}\b", text)

    if len(words) < 4:
        return False

    lowered = text.lower()

    non_claim_patterns = [
        r"^(?:introduction|conclusion|summary|references)$",
        r"^(?:in conclusion|to summarize)\b",
    ]

    if any(
        re.search(pattern, lowered)
        for pattern in non_claim_patterns
    ):
        return False

    return True


# ---------------------------------------------------------
# 5. CLAUSE DECOMPOSITION
# ---------------------------------------------------------

PREDICATE_STARTERS = {
    "increases", "increase", "decreases", "decrease",
    "reduces", "reduce", "causes", "cause",
    "leads", "lead", "results", "result",
    "prevents", "prevent", "protects", "protect",
    "improves", "improve", "worsens", "worsen",
    "raises", "raise", "lowers", "lower",
    "is", "are", "was", "were", "has", "have",
    "may", "might", "could", "can", "will",
    "should", "must", "does", "do",
    "suggests", "suggest", "indicates", "indicate",
    "associated", "related"
}

def has_explicit_subject(text: str) -> bool:
    """
    Conservative heuristic for identifying a clause that
    begins with an explicit subject.

    This is not a full syntactic parser.
    """

    words = re.findall(
        r"\b[A-Za-z][A-Za-z'-]*\b",
        text.strip()
    )

    if len(words) < 2:
        return False

    first = words[0].lower()

    conjunctions = {
        "and", "but", "or", "while",
        "whereas", "although", "because"
    }

    pronouns = {
        "it", "they", "he", "she",
        "we", "this", "these", "those"
    }

    # Explicit noun-like subject or pronoun.
    return (
        first not in conjunctions
        and first not in PREDICATE_STARTERS
        and first not in {"a", "an", "the"}
        and (
            first in pronouns
            or first[0].isupper()
            or len(words) >= 3
        )
    )
        


def split_compound_claim(sentence: str) -> list[str]:
    """
    Split only clearly independent propositions.

    Supported:
    - Semicolon-separated propositions.
    - Coordinated clauses with repeated explicit subjects.

    Not supported:
    - Reconstructing omitted subjects.
    - Splitting symptom lists.
    - Splitting compound noun phrases.

    Ambiguous sentences remain intact.
    """

    sentence = sentence.strip()

    if not sentence:
        return []

    # First, split explicit semicolon boundaries.
    semicolon_parts = [
        part.strip()
        for part in re.split(r";\s*", sentence)
        if part.strip()
    ]

    output = []

    for part in semicolon_parts:

        # Only split conjunctions when both sides appear
        # to contain independently stated subjects.
        conjunction_parts = re.split(
            r",?\s+\b(and|but|while|whereas|although)\b\s+",
            part,
            flags=re.IGNORECASE
        )

        if len(conjunction_parts) == 1:
            output.append(part)
            continue

        clauses = []
        current = conjunction_parts[0].strip()
        clauses.append(current)

        can_split = True

        for index in range(1, len(conjunction_parts), 2):
            conjunction = conjunction_parts[index]
            next_clause = conjunction_parts[index + 1].strip()

            # Both clauses must have explicit subjects.
            if not has_explicit_subject(current):
                can_split = False
                break

            if not has_explicit_subject(next_clause):
                can_split = False
                break

            clauses.append(next_clause)
            current = next_clause

        if can_split and len(clauses) > 1:
            output.extend(clauses)
        else:
            output.append(part)

    return output


# ---------------------------------------------------------
# 6. CLAIM CLASSIFICATION
# ---------------------------------------------------------

def classify_claim(text: str) -> dict[str, Any]:
    """
    Identify linguistic characteristics.

    These are extraction metadata, not truth judgments.
    """

    lowered = text.lower()

    is_inferential = any(
        re.search(pattern, lowered)
        for pattern in SOFT_PATTERNS
    )

    is_definitive = any(
        re.search(pattern, lowered)
        for pattern in DEFINITIVE_PATTERNS
    )

    is_causal = any(
        re.search(pattern, lowered)
        for pattern in CAUSAL_PATTERNS
    )

    is_negation = any(
        re.search(pattern, lowered)
        for pattern in NEGATION_PATTERNS
    )

    is_comparative = any(
        re.search(pattern, lowered)
        for pattern in COMPARATIVE_PATTERNS
    )

    if is_causal:
        relationship_type = "causal"
    elif is_comparative:
        relationship_type = "comparative"
    elif is_inferential:
        relationship_type = "correlational"
    else:
        relationship_type = "factual"

    numerical_matches = NUMERICAL_PATTERN.findall(text)

    numerical_mentions = extract_numerical_mentions(text)

    return {
        "claim_type": (
            "inferential" if is_inferential else "factual"
        ),
        "is_inferential": is_inferential,
        "is_definitive": is_definitive,
        "contains_causal_language": is_causal,
        "contains_negation": is_negation,
        "contains_comparative_language": is_comparative,
        "relationship_type": relationship_type,
        "contains_numerical_information": bool(
            numerical_matches
        ),
        "numerical_values": numerical_matches,
        "numerical_mentions": numerical_mentions
    }


# ---------------------------------------------------------
# 7. STRUCTURED EXTRACTION
# ---------------------------------------------------------

def extract_claim_records(
    text: str,
    inferential: bool = False
) -> list[dict[str, Any]]:
    """
    Extract atomic claim records.

    Every record represents one independently stated
    proposition, subject to conservative decomposition.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    normalized = clean_markdown(text)

    sentences = split_sentences(normalized)

    records = []
    seen = set()

    for sentence in sentences:

        atomic_candidates = split_compound_claim(sentence)

        for candidate in atomic_candidates:

            candidate = candidate.strip(" -*:\u2022")

            # Remove residual surrounding punctuation.
            candidate = candidate.strip()

            if not is_meaningful_claim(candidate):
                continue

            lowered = candidate.lower()

            # Avoid exact duplicate claims.
            normalized_key = re.sub(
                r"\s+",
                " ",
                lowered
            )

            if normalized_key in seen:
                continue

            metadata = classify_claim(candidate)

            # Preserve existing inferential-mode behavior:
            # only cautious claims are eligible.
            if inferential and not metadata["is_inferential"]:
                continue

            seen.add(normalized_key)

            records.append({
                "claim_id": f"CLM-{len(records) + 1:04d}",
                "claim_text": candidate,
                "original_sentence": sentence,
                "atomic": len(atomic_candidates) > 1,
                **metadata
            })

    return records


# ---------------------------------------------------------
# 8. BACKWARD-COMPATIBLE EXTRACTION API
# ---------------------------------------------------------

def extract_claims(
    text: str,
    inferential: bool = False
) -> list[str]:
    """
    Backward-compatible API.

    Existing orchestrator expects list[str].
    Structured extraction is available through
    extract_claim_records().
    """

    records = extract_claim_records(
        text,
        inferential=inferential
    )

    return [
        record["claim_text"]
        for record in records
    ]


if __name__ == "__main__":

    sample = """
    Smoking increases cardiovascular disease risk,
    and obesity contributes to hypertension.
    Physical activity may reduce blood pressure by 5%.
    """

    print("EXTRACTED CLAIMS")

    for record in extract_claim_records(sample):
        print(record)
