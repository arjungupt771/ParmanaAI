
"""
ParmanaAI - Numerical Consistency Engine

Phase 4.3.1: Unit normalization.

Normalizes compatible numerical measurements to canonical units.

This module does not establish clinical equivalence or factual
correctness. Unsupported conversions return None.
"""

from decimal import Decimal, InvalidOperation
from typing import Any

from backend.agents.numerical_extractor import UNIT_ALIASES


# unit -> (dimension, canonical unit, multiplicative factor)
LINEAR_UNITS = {
    # Mass: canonical mg
    "mcg": ("mass", "mg", Decimal("0.001")),
    "mg": ("mass", "mg", Decimal("1")),
    "g": ("mass", "mg", Decimal("1000")),
    "kg": ("mass", "mg", Decimal("1000000")),

    # Volume: canonical ml
    "ml": ("volume", "ml", Decimal("1")),
    "l": ("volume", "ml", Decimal("1000")),

    # Duration: canonical minutes
    "minutes": ("duration", "minutes", Decimal("1")),
    "hours": ("duration", "minutes", Decimal("60")),
    "days": ("duration", "minutes", Decimal("1440")),
    "weeks": ("duration", "minutes", Decimal("10080")),

    # Percentages remain percentages
    "percent": ("percentage", "percent", Decimal("1")),

    # Pressure
    "mmHg": ("pressure", "mmHg", Decimal("1")),

    # Concentrations cannot be cross-converted without analyte
    "mg/dL": ("concentration:mg/dL", "mg/dL", Decimal("1")),
    "mmol/L": ("concentration:mmol/L", "mmol/L", Decimal("1")),
}


def canonicalize_unit(unit: str | None) -> str | None:
    """Resolve known unit aliases."""

    if not isinstance(unit, str) or not unit.strip():
        return None

    cleaned = unit.strip()

    return UNIT_ALIASES.get(cleaned.lower(), cleaned)


def normalize_measurement(
    value: int | float | Decimal,
    unit: str | None,
) -> dict[str, Any] | None:
    """
    Normalize a measurement to a canonical unit.

    Returns None for unknown units, invalid numbers, or conversions
    that require additional context.
    """

    if isinstance(value, bool):
        return None

    normalized_unit = canonicalize_unit(unit)

    if normalized_unit is None:
        return None

    try:
        numeric_value = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None

    if not numeric_value.is_finite():
        return None

    # Temperature uses an affine transformation.
    if normalized_unit in {"celsius", "fahrenheit"}:

        if normalized_unit == "fahrenheit":
            converted = (
                (numeric_value - Decimal("32"))
                * Decimal("5")
                / Decimal("9")
            )
        else:
            converted = numeric_value

        return {
            "original_value": float(numeric_value),
            "original_unit": unit,
            "normalized_value": float(converted),
            "canonical_unit": "celsius",
            "dimension": "temperature",
        }

    # Months and years are deliberately unsupported.
    if normalized_unit in {"months", "years"}:
        return None

    specification = LINEAR_UNITS.get(normalized_unit)

    if specification is None:
        return None

    dimension, canonical_unit, factor = specification

    converted = numeric_value * factor

    return {
        "original_value": float(numeric_value),
        "original_unit": unit,
        "normalized_value": float(converted),
        "canonical_unit": canonical_unit,
        "dimension": dimension,
    }


def units_compatible(
    first_unit: str | None,
    second_unit: str | None,
) -> bool:
    """
    Check whether two units can be compared through normalization.

    This checks unit compatibility only, not whether the measurements
    describe the same clinical quantity.
    """

    first = normalize_measurement(1, first_unit)
    second = normalize_measurement(1, second_unit)

    if first is None or second is None:
        return False

    return (
        first["dimension"] == second["dimension"]
        and first["canonical_unit"] == second["canonical_unit"]
    )
