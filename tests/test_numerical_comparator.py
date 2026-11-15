
import pytest

from backend.agents.numerical_comparator import (
    compare_numerical_mentions,
)


def scalar(value, unit=None, comparator=None, category=None):
    return {
        "value": value,
        "upper_value": None,
        "unit": unit,
        "comparator": comparator,
        "category": category or ("unitless" if unit is None else "measurement"),
    }


def range_value(lower, upper, unit):
    return {
        "value": lower,
        "upper_value": upper,
        "unit": unit,
        "comparator": "range",
        "category": "range",
    }


def test_equivalent_mass_with_different_units():
    result = compare_numerical_mentions(
        scalar(500, "mg"),
        scalar(0.5, "g"),
    )
    assert result["status"] == "equivalent"
    assert result["canonical_unit"] == "mg"


def test_scalar_mismatch():
    result = compare_numerical_mentions(
        scalar(500, "mg"),
        scalar(600, "mg"),
    )
    assert result["status"] == "mismatch"


def test_incompatible_units():
    result = compare_numerical_mentions(
        scalar(5, "mg"),
        scalar(5, "ml"),
    )
    assert result["status"] == "incomparable"


def test_unitless_values_can_be_compared():
    result = compare_numerical_mentions(
        scalar(10),
        scalar(10),
    )
    assert result["status"] == "equivalent"


def test_unitless_and_unitful_values_are_incomparable():
    result = compare_numerical_mentions(
        scalar(10),
        scalar(10, "mg"),
    )
    assert result["status"] == "incomparable"


def test_scalar_inside_claim_range():
    result = compare_numerical_mentions(
        range_value(120, 140, "mmHg"),
        scalar(130, "mmHg"),
    )
    assert result["status"] == "within_range"


def test_scalar_outside_claim_range():
    result = compare_numerical_mentions(
        range_value(120, 140, "mmHg"),
        scalar(150, "mmHg"),
    )
    assert result["status"] == "outside_range"


def test_scalar_inside_evidence_range():
    result = compare_numerical_mentions(
        scalar(130, "mmHg"),
        range_value(120, 140, "mmHg"),
    )
    assert result["status"] == "within_range"


def test_scalar_outside_evidence_range():
    result = compare_numerical_mentions(
        scalar(150, "mmHg"),
        range_value(120, 140, "mmHg"),
    )
    assert result["status"] == "outside_range"


def test_equivalent_ranges():
    result = compare_numerical_mentions(
        range_value(120, 140, "mmHg"),
        range_value(120, 140, "mmHg"),
    )
    assert result["status"] == "equivalent"


def test_overlapping_ranges():
    result = compare_numerical_mentions(
        range_value(100, 140, "mmHg"),
        range_value(130, 160, "mmHg"),
    )
    assert result["status"] == "overlapping_ranges"


def test_disjoint_ranges():
    result = compare_numerical_mentions(
        range_value(100, 120, "mmHg"),
        range_value(130, 150, "mmHg"),
    )
    assert result["status"] == "disjoint_ranges"


def test_greater_than_constraint_satisfied():
    result = compare_numerical_mentions(
        scalar(5, "mg", comparator=">="),
        scalar(7, "mg"),
    )
    assert result["status"] == "satisfies_constraint"


def test_greater_than_constraint_violated():
    result = compare_numerical_mentions(
        scalar(5, "mg", comparator=">="),
        scalar(3, "mg"),
    )
    assert result["status"] == "violates_constraint"


def test_strict_greater_than_constraint():
    result = compare_numerical_mentions(
        scalar(5, "mg", comparator=">"),
        scalar(5, "mg"),
    )
    assert result["status"] == "violates_constraint"


def test_less_than_constraint_satisfied():
    result = compare_numerical_mentions(
        scalar(5, "mg", comparator="<="),
        scalar(4, "mg"),
    )
    assert result["status"] == "satisfies_constraint"


def test_explicit_absolute_tolerance():
    result = compare_numerical_mentions(
        scalar(100, "mg"),
        scalar(100.05, "mg"),
        absolute_tolerance=0.1,
    )
    assert result["status"] == "equivalent"


def test_default_tolerance_is_zero():
    result = compare_numerical_mentions(
        scalar(100, "mg"),
        scalar(100.05, "mg"),
    )
    assert result["status"] == "mismatch"


def test_different_statistical_measures_are_incomparable():
    result = compare_numerical_mentions(
        {
            **scalar(1.5),
            "category": "statistical_measure",
            "statistic": "OR",
        },
        {
            **scalar(1.5),
            "category": "statistical_measure",
            "statistic": "RR",
        },
    )
    assert result["status"] == "incomparable"


def test_same_statistical_measure_can_be_compared():
    result = compare_numerical_mentions(
        {
            **scalar(1.5),
            "category": "statistical_measure",
            "statistic": "OR",
        },
        {
            **scalar(1.5),
            "category": "statistical_measure",
            "statistic": "OR",
        },
    )
    assert result["status"] == "equivalent"


def test_dates_are_not_compared_as_measurements():
    result = compare_numerical_mentions(
        {"category": "date", "value": "2026-01-01"},
        {"category": "date", "value": "2026-01-01"},
    )
    assert result["status"] == "incomparable"


def test_invalid_negative_tolerance():
    result = compare_numerical_mentions(
        scalar(5, "mg"),
        scalar(5, "mg"),
        absolute_tolerance=-1,
    )
    assert result["status"] == "incomparable"


def test_result_does_not_claim_context_was_checked():
    result = compare_numerical_mentions(
        scalar(5, "mg"),
        scalar(5, "mg"),
    )
    assert result["context_checked"] is False
