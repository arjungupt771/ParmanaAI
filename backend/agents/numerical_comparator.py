
"""
ParmanaAI - Numerical Comparator

Phase 4.3.2: Compare extracted numerical mentions.

This module evaluates numerical relationships only.
It does not establish clinical equivalence, factual correctness,
or whether two mentions refer to the same clinical context.
"""

from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

from backend.agents.numerical_consistency import normalize_measurement


COMPARISON_STATUSES = {
    "equivalent",
    "mismatch",
    "within_range",
    "outside_range",
    "satisfies_constraint",
    "violates_constraint",
    "overlapping_ranges",
    "disjoint_ranges",
    "incomparable",
}


def _decimal(value: Any) -> Decimal | None:
    """Safely convert a numeric value to Decimal."""

    if isinstance(value, bool):
        return None

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None

    if not result.is_finite():
        return None

    return result


def _normalized_value(
    value: Any,
    unit: str | None,
) -> tuple[Decimal, str, str | None] | None:
    """
    Return normalized value, dimension, and canonical unit.

    Unitless values can only be compared with other unitless values.
    """

    numeric = _decimal(value)

    if numeric is None:
        return None

    if not unit:
        return numeric, "unitless", None

    normalized = normalize_measurement(numeric, unit)

    if normalized is None:
        return None

    return (
        Decimal(str(normalized["normalized_value"])),
        normalized["dimension"],
        normalized["canonical_unit"],
    )


def _result(
    status: str,
    reason: str,
    claim_value: Any = None,
    evidence_value: Any = None,
    canonical_unit: str | None = None,
) -> dict[str, Any]:

    return {
        "status": status,
        "comparable": status != "incomparable",
        "reason": reason,
        "claim_value": claim_value,
        "evidence_value": evidence_value,
        "canonical_unit": canonical_unit,
        "context_checked": False,
    }


def _compatible_pair(
    first_value: Any,
    first_unit: str | None,
    second_value: Any,
    second_unit: str | None,
):
    """Normalize two values and ensure their dimensions match."""

    first = _normalized_value(first_value, first_unit)
    second = _normalized_value(second_value, second_unit)

    if first is None or second is None:
        return None

    if first[1] != second[1] or first[2] != second[2]:
        return None

    return first, second


def _within_tolerance(
    first: Decimal,
    second: Decimal,
    absolute_tolerance: Decimal,
    relative_tolerance: Decimal,
) -> bool:
    """Compare values using explicitly supplied tolerances."""

    difference = abs(first - second)

    allowed_difference = max(
        absolute_tolerance,
        relative_tolerance * max(abs(first), abs(second)),
    )

    return difference <= allowed_difference


def _constraint_satisfied(
    comparator: str,
    threshold: Decimal,
    observed: Decimal,
) -> bool | None:
    """Evaluate an inequality constraint against an observed value."""

    if comparator == ">=":
        return observed >= threshold

    if comparator == ">":
        return observed > threshold

    if comparator == "<=":
        return observed <= threshold

    if comparator == "<":
        return observed < threshold

    return None


def _validate_tolerances(
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[Decimal, Decimal] | None:

    absolute = _decimal(absolute_tolerance)
    relative = _decimal(relative_tolerance)

    if absolute is None or relative is None:
        return None

    if absolute < 0 or relative < 0:
        return None

    return absolute, relative


def compare_numerical_mentions(
    claim: dict[str, Any],
    evidence: dict[str, Any],
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    """
    Compare one extracted claim mention with one evidence mention.

    Tolerances are applied in canonical units.

    Default tolerances are zero: no implicit clinical rounding
    assumptions are made.

    Supported:
    - Scalar vs scalar
    - Scalar vs range
    - Range vs scalar
    - Range vs range
    - Claim inequality vs evidence scalar

    Unsupported or context-dependent comparisons return
    'incomparable'.
    """

    if not isinstance(claim, dict) or not isinstance(evidence, dict):
        return _result("incomparable", "Invalid numerical mention.")

    tolerances = _validate_tolerances(
        absolute_tolerance,
        relative_tolerance,
    )

    if tolerances is None:
        return _result("incomparable", "Invalid comparison tolerance.")

    abs_tol, rel_tol = tolerances

    claim_category = claim.get("category")
    evidence_category = evidence.get("category")

    # Dates and confidence interval levels are not ordinary
    # clinical measurement values.
    if claim_category == "date" or evidence_category == "date":
        return _result("incomparable", "Date comparison is not supported.")

    if (
        claim_category == "confidence_level"
        or evidence_category == "confidence_level"
    ):
        return _result(
            "incomparable",
            "Confidence interval levels require statistical context.",
        )

    # Statistical measures must use the same statistic type.
    if (
        claim_category == "statistical_measure"
        or evidence_category == "statistical_measure"
    ):
        if (
            claim_category != "statistical_measure"
            or evidence_category != "statistical_measure"
            or claim.get("statistic") != evidence.get("statistic")
        ):
            return _result(
                "incomparable",
                "Statistical measure types do not match.",
            )

    claim_is_range = claim_category == "range"
    evidence_is_range = evidence_category == "range"

    claim_comparator = claim.get("comparator")

    if evidence.get("comparator") in {
        ">=", ">", "<=", "<", "approximately"
    }:
        return _result(
            "incomparable",
            "Evidence is itself a constraint; constraint-to-constraint "
            "comparison is not yet supported.",
        )

    # Inequality constraint on the claim side.
    if claim_comparator in {">=", ">", "<=", "<"}:

        if evidence_is_range:
            return _result(
                "incomparable",
                "Inequality claims against evidence ranges are not yet supported.",
            )

        pair = _compatible_pair(
            claim.get("value"),
            claim.get("unit"),
            evidence.get("value"),
            evidence.get("unit"),
        )

        if pair is None:
            return _result(
                "incomparable",
                "Units or numerical values are incompatible.",
            )

        (threshold, _, canonical_unit), (observed, _, _) = pair

        satisfied = _constraint_satisfied(
            claim_comparator,
            threshold,
            observed,
        )

        if satisfied is None:
            return _result("incomparable", "Unsupported comparator.")

        return _result(
            "satisfies_constraint" if satisfied else "violates_constraint",
            "Evidence value satisfies the claim inequality."
            if satisfied
            else "Evidence value violates the claim inequality.",
            float(threshold),
            float(observed),
            canonical_unit,
        )

    # Approximate equality: tolerance must be explicitly supplied.
    if claim_comparator == "approximately":
        if claim_is_range or evidence_is_range:
            return _result(
                "incomparable",
                "Approximate range comparison is not supported.",
            )

        pair = _compatible_pair(
            claim.get("value"),
            claim.get("unit"),
            evidence.get("value"),
            evidence.get("unit"),
        )

        if pair is None:
            return _result(
                "incomparable",
                "Units or numerical values are incompatible.",
            )

        (first, _, canonical_unit), (second, _, _) = pair

        equal = _within_tolerance(
            first, second, abs_tol, rel_tol
        )

        return _result(
            "equivalent" if equal else "mismatch",
            "Values are within the configured tolerance."
            if equal
            else "Values exceed the configured tolerance.",
            float(first),
            float(second),
            canonical_unit,
        )

    # Range vs scalar.
    if claim_is_range != evidence_is_range:

        range_mention = claim if claim_is_range else evidence
        scalar_mention = evidence if claim_is_range else claim

        lower = range_mention.get("value")
        upper = range_mention.get("upper_value")

        lower_pair = _compatible_pair(
            lower,
            range_mention.get("unit"),
            scalar_mention.get("value"),
            scalar_mention.get("unit"),
        )

        upper_pair = _compatible_pair(
            upper,
            range_mention.get("unit"),
            scalar_mention.get("value"),
            scalar_mention.get("unit"),
        )

        if lower_pair is None or upper_pair is None:
            return _result(
                "incomparable",
                "Range and scalar units are incompatible.",
            )

        (lower_value, _, canonical_unit), (scalar_value, _, _) = lower_pair
        (upper_value, _, _), _ = upper_pair

        if lower_value > upper_value:
            return _result("incomparable", "Range bounds are reversed.")

        within = (
            lower_value - abs_tol <= scalar_value <= upper_value + abs_tol
        )

        return _result(
            "within_range" if within else "outside_range",
            "Scalar falls within the evidence range."
            if within
            else "Scalar falls outside the evidence range.",
            float(lower_value),
            float(scalar_value),
            canonical_unit,
        )

    # Range vs range.
    if claim_is_range and evidence_is_range:

        first_lower = _compatible_pair(
            claim.get("value"),
            claim.get("unit"),
            evidence.get("value"),
            evidence.get("unit"),
        )

        first_upper = _compatible_pair(
            claim.get("upper_value"),
            claim.get("unit"),
            evidence.get("value"),
            evidence.get("unit"),
        )

        second_lower = _compatible_pair(
            claim.get("value"),
            claim.get("unit"),
            evidence.get("upper_value"),
            evidence.get("unit"),
        )

        if first_lower is None or first_upper is None or second_lower is None:
            return _result(
                "incomparable",
                "Range units are incompatible.",
            )

        claim_low = first_lower[0][0]
        evidence_low = first_lower[1][0]
        claim_high = first_upper[0][0]
        evidence_high = second_lower[1][0]

        if claim_low > claim_high or evidence_low > evidence_high:
            return _result("incomparable", "Range bounds are reversed.")

        if (
            _within_tolerance(claim_low, evidence_low, abs_tol, rel_tol)
            and _within_tolerance(claim_high, evidence_high, abs_tol, rel_tol)
        ):
            status = "equivalent"
            reason = "Ranges have equivalent bounds."

        elif claim_low <= evidence_high and evidence_low <= claim_high:
            status = "overlapping_ranges"
            reason = "Ranges overlap but their bounds differ."

        else:
            status = "disjoint_ranges"
            reason = "Ranges do not overlap."

        return _result(
            status,
            reason,
            [float(claim_low), float(claim_high)],
            [float(evidence_low), float(evidence_high)],
            first_lower[0][2],
        )

    # Scalar vs scalar.
    pair = _compatible_pair(
        claim.get("value"),
        claim.get("unit"),
        evidence.get("value"),
        evidence.get("unit"),
    )

    if pair is None:
        return _result(
            "incomparable",
            "Units or numerical values are incompatible.",
        )

    (first, _, canonical_unit), (second, _, _) = pair

    equal = _within_tolerance(
        first,
        second,
        abs_tol,
        rel_tol,
    )

    return _result(
        "equivalent" if equal else "mismatch",
        "Numerical values match within the configured tolerance."
        if equal
        else "Numerical values differ beyond the configured tolerance.",
        float(first),
        float(second),
        canonical_unit,
    )
