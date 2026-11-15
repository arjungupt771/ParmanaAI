
"""
ParmanaAI - Structured Numerical Claim Extractor

Extracts numerical mentions from claims.

Supported:
- Scalar values
- Ranges
- Comparators
- Units
- Percentages
- Statistical measures (OR, RR, HR)
- Confidence interval levels
- ISO dates

This module extracts numerical information only.
It does not verify factual correctness.
"""

import re
from typing import Any


NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

UNIT = (
    r"mmol/L|mg/dL|mmHg|"
    r"°C|°F|"
    r"mcg|μg|µg|"
    r"mg|kg|ml|mL|"
    r"liters?|litres?|"
    r"grams?|"
    r"years?|months?|weeks?|days?|hours?|minutes?|"
    r"%"
)

UNIT_ALIASES = {
    "°c": "celsius",
    "°f": "fahrenheit",
    "mg": "mg",
    "mcg": "mcg",
    "μg": "mcg",
    "µg": "mcg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "mmhg": "mmHg",
    "mmol/l": "mmol/L",
    "mg/dl": "mg/dL",
    "%": "percent",
    "year": "years",
    "years": "years",
    "month": "months",
    "months": "months",
    "week": "weeks",
    "weeks": "weeks",
    "day": "days",
    "days": "days",
    "hour": "hours",
    "hours": "hours",
    "minute": "minutes",
    "minutes": "minutes",
}


def parse_number(value: str) -> float:
    """Convert formatted numeric text into a float."""
    return float(value.replace(",", ""))


def classify_unit(unit: str | None) -> str:
    """Assign a broad category to a numerical unit."""

    if not unit:
        return "unitless"

    normalized = UNIT_ALIASES.get(unit.lower(), unit)

    if normalized == "percent":
        return "percentage"

    if normalized in {"celsius", "fahrenheit"}:
        return "temperature"

    if normalized == "mmHg":
        return "pressure"

    if normalized in {"mg", "mcg", "g", "kg"}:
        return "mass"

    if normalized in {"ml", "l"}:
        return "volume"

    if normalized in {"mmol/L", "mg/dL"}:
        return "concentration"

    if normalized in {
        "years", "months", "weeks", "days",
        "hours", "minutes"
    }:
        return "duration"

    return "unitless"


def _base_record(
    raw_text: str,
    start: int,
    end: int,
    category: str,
    unit: str | None
) -> dict[str, Any]:

    normalized_unit = (
        UNIT_ALIASES.get(unit.lower(), unit)
        if unit else None
    )

    return {
        "raw_text": raw_text,
        "start": start,
        "end": end,
        "category": category,
        "unit": unit,
        "normalized_unit": normalized_unit,
    }


def extract_numerical_mentions(
    text: str
) -> list[dict[str, Any]]:
    """
    Extract structured numerical mentions from text.

    Results are returned in their original text order.
    Overlapping matches are discarded.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    matches = []

    # 1. ISO dates: YYYY-MM-DD.
    date_pattern = re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b"
    )
    date_spans = []

    for match in date_pattern.finditer(text):
        date_spans.append((match.start(), match.end()))

        try:
            from datetime import date
            date.fromisoformat(match.group())
        except ValueError:
            continue

        record = _base_record(
            match.group(),
            match.start(),
            match.end(),
            "date",
            None
        )

        record.update({
            "value": match.group(),
            "upper_value": None,
            "comparator": None,
        })

        matches.append(record)

    def overlaps_date(start: int, end: int) -> bool:
        return any(
            start < date_end and end > date_start
            for date_start, date_end in date_spans
        )

    # 2. Confidence interval level, e.g. 95% CI.
    ci_pattern = re.compile(
        rf"\b(?P<value>{NUMBER})\s*%\s*"
        r"(?:confidence\s+interval|CI)\b",
        re.IGNORECASE
    )

    for match in ci_pattern.finditer(text):
        record = _base_record(
            match.group(),
            match.start(),
            match.end(),
            "confidence_level",
            "%"
        )

        record.update({
            "value": parse_number(match.group("value")),
            "upper_value": None,
            "comparator": None,
            "statistic": "confidence_interval",
        })

        matches.append(record)

    # 3. Statistical ratios: OR, RR, HR.
    statistic_pattern = re.compile(
        rf"\b(?P<label>OR|RR|HR)\s*[:=]?\s*"
        rf"(?P<value>{NUMBER})\b",
        re.IGNORECASE
    )

    for match in statistic_pattern.finditer(text):
        record = _base_record(
            match.group(),
            match.start(),
            match.end(),
            "statistical_measure",
            None
        )

        record.update({
            "value": parse_number(match.group("value")),
            "upper_value": None,
            "comparator": None,
            "statistic": match.group("label").upper(),
        })

        matches.append(record)

    # 4. Numerical ranges, e.g. 120-140 mmHg.
    range_pattern = re.compile(
        rf"(?P<lower>{NUMBER})\s*"
        r"(?:-|–|—|\bto\b)\s*"
        rf"(?P<upper>{NUMBER})\s*"
        rf"(?P<unit>{UNIT})?(?![A-Za-z])",
        re.IGNORECASE
        )

    for match in range_pattern.finditer(text):
        if overlaps_date(match.start(), match.end()):
            continue

        unit = match.group("unit")

        record = _base_record(
            match.group(),
            match.start(),
            match.end(),
            "range",
            unit
        )

        record.update({
            "value": parse_number(match.group("lower")),
            "upper_value": parse_number(match.group("upper")),
            "comparator": "range",
        })

        matches.append(record)

    # 5. Scalar values, percentages and comparators.
    scalar_pattern = re.compile(
        rf"(?P<comparator>"
        r">=|<=|≥|≤|>|<|approximately\s+|about\s+|"
        r"at\s+least\s+|at\s+most\s+"
        r")?"
        rf"\s*(?P<value>{NUMBER})"
        rf"\s*(?P<unit>{UNIT})?(?![A-Za-z])",
        re.IGNORECASE
        )

    for match in scalar_pattern.finditer(text):
        if overlaps_date(match.start(), match.end()):
            continue

        unit = match.group("unit")

        # Ignore standalone values already covered by date
        # or statistical patterns.
        if any(
            match.start() < existing["end"]
            and match.end() > existing["start"]
            for existing in matches
        ):
            continue

        comparator = match.group("comparator")

        if comparator:
            comparator = comparator.strip().lower()

            comparator = {
                ">=": ">=",
                "≥": ">=",
                "<=": "<=",
                "≤": "<=",
                ">": ">",
                "<": "<",
                "approximately": "approximately",
                "about": "approximately",
                "at least": ">=",
                "at most": "<=",
            }.get(comparator, comparator)

        record = _base_record(
            match.group(),
            match.start(),
            match.end(),
            classify_unit(unit),
            unit
        )

        record.update({
            "value": parse_number(match.group("value")),
            "upper_value": None,
            "comparator": comparator,
        })

        matches.append(record)

    # Remove overlapping matches and preserve text order.
    matches.sort(
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"])
        )
    )

    output = []
    occupied_until = -1

    for record in matches:
        if record["start"] < occupied_until:
            continue

        output.append(record)
        occupied_until = record["end"]

    return output
