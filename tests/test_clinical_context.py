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


def test_overlapping_analyte_aliases():
    from backend.agents.clinical_context import extract_clinical_context

    ctx1 = extract_clinical_context("Patient systolic blood pressure was 130 mmHg.")
    ctx2 = extract_clinical_context("Patient SBP was 130 mmHg.")
    assert ctx1["analytes"] == ["systolic_blood_pressure"]
    assert ctx2["analytes"] == ["systolic_blood_pressure"]

    ctx_dbp = extract_clinical_context("Diastolic blood pressure was 85 mmHg.")
    assert ctx_dbp["analytes"] == ["diastolic_blood_pressure"]

    # Incompatible analytes produce mismatch
    comp = compare_clinical_context(ctx1, ctx_dbp)
    assert comp["status"] == "mismatch"
    assert "analytes" in comp["mismatches"]


def test_incompatible_clinical_dimensions():
    # Route mismatch
    oral_ctx = {"routes": ["oral"]}
    iv_ctx = {"routes": ["intravenous"]}
    res = compare_clinical_context(oral_ctx, iv_ctx)
    assert res["status"] == "mismatch"
    assert "routes" in res["mismatches"]

    # Population mismatch
    adult_ctx = {"populations": ["adult"]}
    peds_ctx = {"populations": ["child"]}
    res = compare_clinical_context(adult_ctx, peds_ctx)
    assert res["status"] == "mismatch"
    assert "populations" in res["mismatches"]

    # Condition mismatch
    fasting_ctx = {"conditions": ["fasting"]}
    postprandial_ctx = {"conditions": ["post_meal"]}
    res = compare_clinical_context(fasting_ctx, postprandial_ctx)
    assert res["status"] == "mismatch"
    assert "conditions" in res["mismatches"]

