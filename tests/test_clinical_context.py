from backend.agents.clinical_context import (
    compare_clinical_context,
)


def test_dimensions_absent_from_both_passages_are_not_applicable():
    result = compare_clinical_context(
        {
            "analytes": ["glucose"],
            "conditions": ["fasting"],
        },
        {
            "analytes": ["glucose"],
            "conditions": ["fasting"],
        },
    )

    assert result["status"] == "match"
    assert result["dimension_results"]["routes"]["status"] == "not_applicable"
    assert "routes" in result["not_applicable_dimensions"]
    assert "routes" not in result["unknown_dimensions"]


def test_one_sided_missing_dimension_remains_unknown():
    result = compare_clinical_context(
        {"routes": ["oral"]},
        {},
    )

    assert result["dimension_results"]["routes"]["status"] == "unknown"
    assert "routes" in result["unknown_dimensions"]


def test_entirely_absent_context_is_not_applicable():
    result = compare_clinical_context({}, {})

    assert result["status"] == "not_applicable"
    assert result["comparable"] is False
