
"""
ParmanaAI - Clinical Context Extraction and Matching

Phase 4.3.3

Extracts explicitly stated clinical context from claim/evidence text
and compares the extracted dimensions.

This module uses deterministic patterns and controlled vocabularies.
It does not infer missing clinical information.
It does not establish clinical equivalence.
"""

import re
from typing import Any


# ---------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------

ANALYTE_ALIASES = {
    "fasting blood glucose": "glucose",
    "fasting plasma glucose": "glucose",
    "random blood glucose": "glucose",
    "random blood sugar": "glucose",
    "blood glucose": "glucose",
    "blood sugar": "glucose",
    "fbg": "glucose",
    "fpg": "glucose",
    "rbs": "glucose",
    "glucose": "glucose",

    "glycated hemoglobin": "hba1c",
    "glycosylated hemoglobin": "hba1c",
    "hba1c": "hba1c",
    "a1c": "hba1c",

    "systolic blood pressure": "systolic_blood_pressure",
    "diastolic blood pressure": "diastolic_blood_pressure",
    "sbp": "systolic_blood_pressure",
    "dbp": "diastolic_blood_pressure",
    "blood pressure": "blood_pressure",
    "bp": "blood_pressure",

    "heart rate": "heart_rate",
    "pulse rate": "heart_rate",
    "pulse": "heart_rate",

    "body temperature": "body_temperature",
    "temperature": "body_temperature",

    "oxygen saturation": "oxygen_saturation",
    "spo2": "oxygen_saturation",
    "sao2": "oxygen_saturation",

    "serum creatinine": "creatinine",
    "creatinine": "creatinine",

    "total cholesterol": "total_cholesterol",
    "ldl cholesterol": "ldl",
    "ldl-c": "ldl",
    "ldl": "ldl",
    "hdl cholesterol": "hdl",
    "hdl-c": "hdl",
    "hdl": "hdl",
    "triglycerides": "triglycerides",
    "cholesterol": "cholesterol",

    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",

    "platelet count": "platelet_count",
    "platelets": "platelet_count",

    "body mass index": "bmi",
    "bmi": "bmi",
}


POPULATION_ALIASES = {
    "neonates": "neonate",
    "neonate": "neonate",
    "newborns": "neonate",
    "newborn": "neonate",

    "infants": "infant",
    "infant": "infant",

    "children": "child",
    "child": "child",
    "pediatric patients": "child",
    "paediatric patients": "child",

    "adolescents": "adolescent",
    "adolescent": "adolescent",
    "teenagers": "adolescent",

    "adults": "adult",
    "adult": "adult",

    "elderly patients": "older_adult",
    "older adults": "older_adult",
    "elderly": "older_adult",

    "pregnant women": "pregnant",
    "pregnant patients": "pregnant",
    "pregnancy": "pregnant",
}


CONDITION_ALIASES = {
    "fasting": "fasting",
    "fasting state": "fasting",

    "postprandial": "post_meal",
    "post-prandial": "post_meal",
    "after meals": "post_meal",
    "after a meal": "post_meal",
    "post meal": "post_meal",

    "preprandial": "pre_meal",
    "pre-prandial": "pre_meal",
    "before meals": "pre_meal",
    "before a meal": "pre_meal",

    "resting": "resting",
    "at rest": "resting",

    "during exercise": "during_exercise",
    "while exercising": "during_exercise",
    "exercise": "during_exercise",

    "sleeping": "sleeping",
    "during sleep": "sleeping",

    "standing": "standing",
    "supine": "supine",
    "lying down": "supine",
}


ROUTE_ALIASES = {
    "intravenous": "intravenous",
    "iv": "intravenous",

    "intramuscular": "intramuscular",
    "im": "intramuscular",

    "subcutaneous": "subcutaneous",
    "subcutaneously": "subcutaneous",

    "oral": "oral",
    "orally": "oral",
    "by mouth": "oral",

    "topical": "topical",
    "topically": "topical",

    "inhaled": "inhaled",
    "inhalation": "inhaled",
}


FREQUENCY_ALIASES = {
    "once daily": "once_daily",
    "once a day": "once_daily",
    "daily": "once_daily",

    "twice daily": "twice_daily",
    "twice a day": "twice_daily",
    "bid": "twice_daily",

    "three times daily": "three_times_daily",
    "three times a day": "three_times_daily",
    "tid": "three_times_daily",

    "four times daily": "four_times_daily",
    "four times a day": "four_times_daily",
    "qid": "four_times_daily",

    "once weekly": "once_weekly",
    "once a week": "once_weekly",
    "weekly": "once_weekly",
}


# ---------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------


def _find_aliases(
    text: str,
    aliases: dict[str, str],
) -> list[str]:
    """
    Find explicitly mentioned aliases.

    Longer phrases take precedence over overlapping shorter phrases.
    Returned canonical values are deduplicated.
    """
    if not isinstance(text, str) or not text:
        return []

    matches = []

    for phrase, canonical in sorted(
        aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(phrase)
            + r"(?![a-z0-9])"
        )

        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()

            overlaps = any(
                start < existing_end and end > existing_start
                for existing_start, existing_end, _ in matches
            )

            if not overlaps:
                matches.append((start, end, canonical))

    matches.sort(key=lambda item: item[0])

    found = []
    for _, _, canonical in matches:
        if canonical not in found:
            found.append(canonical)

    return found



def _extract_time_context(text: str) -> list[str]:
    """
    Extract explicitly stated temporal conditions.

    This captures wording, not inferred calendar durations.
    """

    patterns = [
        r"\bwithin\s+\d+(?:\.\d+)?\s+"
        r"(?:minutes?|hours?|days?|weeks?|months?|years?)\b",

        r"\bafter\s+\d+(?:\.\d+)?\s+"
        r"(?:minutes?|hours?|days?|weeks?|months?|years?)\b",

        r"\bbefore\s+\d+(?:\.\d+)?\s+"
        r"(?:minutes?|hours?|days?|weeks?|months?|years?)\b",

        r"\bat\s+\d+(?:\.\d+)?\s+"
        r"(?:minutes?|hours?|days?|weeks?|months?|years?)\b",

        r"\bfor\s+\d+(?:\.\d+)?\s+"
        r"(?:minutes?|hours?|days?|weeks?|months?|years?)\b",
    ]

    found = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            normalized = re.sub(
                r"\s+",
                " ",
                match.group().strip().lower(),
            )

            if normalized not in found:
                found.append(normalized)

    return found


def extract_clinical_context(text: str) -> dict[str, Any]:
    """
    Extract explicitly stated clinical context.

    Missing dimensions remain empty lists.
    No demographic, clinical, or temporal information is inferred.
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "analytes": [],
            "populations": [],
            "conditions": [],
            "routes": [],
            "frequencies": [],
            "time_contexts": [],
            "context_extracted": False,
        }

    normalized_text = text.lower()

    analytes = _find_aliases(
        normalized_text,
        ANALYTE_ALIASES,
    )

    populations = _find_aliases(
        normalized_text,
        POPULATION_ALIASES,
    )

    conditions = _find_aliases(
        normalized_text,
        CONDITION_ALIASES,
    )

    # Ensure fasting analyte mentions consistently register the fasting condition
    if any(k in normalized_text for k in ("fasting blood glucose", "fasting plasma glucose", "fbg", "fpg")):
        if "fasting" not in conditions:
            conditions.append("fasting")

    routes = _find_aliases(
        normalized_text,
        ROUTE_ALIASES,
    )

    frequencies = _find_aliases(
        normalized_text,
        FREQUENCY_ALIASES,
    )

    time_contexts = _extract_time_context(normalized_text)

    return {
        "analytes": analytes,
        "populations": populations,
        "conditions": conditions,
        "routes": routes,
        "frequencies": frequencies,
        "time_contexts": time_contexts,
        "context_extracted": any([
            analytes,
            populations,
            conditions,
            routes,
            frequencies,
            time_contexts,
        ]),
    }


# ---------------------------------------------------------
# Context comparison
# ---------------------------------------------------------

CONTEXT_DIMENSIONS = (
    "analytes",
    "populations",
    "conditions",
    "routes",
    "frequencies",
    "time_contexts",
)


def compare_clinical_context(
    claim_context: dict[str, Any],
    evidence_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare explicitly extracted clinical context.

    Rules:
    - Different explicit values: mismatch.
    - Equal explicit values: match.
    - Missing information on both sides: not applicable.
    - Missing information on one side: unknown.
    - Multiple conflicting values within one dimension: unknown.

    This is a conservative lexical/context comparison, not a
    complete clinical semantic equivalence model.
    """

    if not isinstance(claim_context, dict):
        claim_context = {}

    if not isinstance(evidence_context, dict):
        evidence_context = {}

    dimension_results = {}
    mismatches = []
    unknowns = []
    not_applicable = []

    for dimension in CONTEXT_DIMENSIONS:
        claim_values = set(
            claim_context.get(dimension) or []
        )

        evidence_values = set(
            evidence_context.get(dimension) or []
        )

        # Neither passage mentions this dimension.
        if not claim_values and not evidence_values:
            status = "not_applicable"
            not_applicable.append(dimension)
        # One-sided missing information or ambiguous values.
        elif (
            (not claim_values and evidence_values)
            or (claim_values and not evidence_values)
            or len(claim_values) > 1
            or len(evidence_values) > 1
        ):
            status = "unknown"
            unknowns.append(dimension)

        elif claim_values == evidence_values:
            status = "match"

        else:
            status = "mismatch"
            mismatches.append(dimension)

        dimension_results[dimension] = {
            "status": status,
            "claim_values": sorted(claim_values),
            "evidence_values": sorted(evidence_values),
        }

    if mismatches:
        overall_status = "mismatch"

    elif unknowns:
        overall_status = "insufficient_context"

    elif not_applicable and len(not_applicable) == len(CONTEXT_DIMENSIONS):
        overall_status = "not_applicable"

    else:
        overall_status = "match"

    return {
        "status": overall_status,
        "comparable": overall_status == "match",
        "dimension_results": dimension_results,
        "mismatches": mismatches,
        "unknown_dimensions": unknowns,
        "not_applicable_dimensions": not_applicable,
        "context_checked": True,
        "clinical_equivalence_established": False,
    }


def extract_and_compare_context(
    claim_text: str,
    evidence_text: str,
) -> dict[str, Any]:
    """
    Convenience function for extracting and comparing two texts.
    """

    claim_context = extract_clinical_context(claim_text)
    evidence_context = extract_clinical_context(evidence_text)

    comparison = compare_clinical_context(
        claim_context,
        evidence_context,
    )

    return {
        "claim_context": claim_context,
        "evidence_context": evidence_context,
        "comparison": comparison,
    }
