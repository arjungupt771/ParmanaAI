
import pytest

from backend.agents.numerical_consistency import (
    canonicalize_unit,
    normalize_measurement,
    units_compatible,
)


def test_mcg_to_mg():
    result = normalize_measurement(500, "mcg")
    assert result["normalized_value"] == pytest.approx(0.5)
    assert result["canonical_unit"] == "mg"


def test_grams_to_mg():
    result = normalize_measurement(2, "g")
    assert result["normalized_value"] == pytest.approx(2000)


def test_kg_to_mg():
    result = normalize_measurement(1, "kg")
    assert result["normalized_value"] == pytest.approx(1000000)


def test_liters_to_ml():
    result = normalize_measurement(1.5, "liters")
    assert result["normalized_value"] == pytest.approx(1500)
    assert result["canonical_unit"] == "ml"


def test_fahrenheit_to_celsius():
    result = normalize_measurement(100, "°F")
    assert result["normalized_value"] == pytest.approx(
        37.7777777778, abs=1e-6
    )


def test_celsius_unchanged():
    result = normalize_measurement(38, "°C")
    assert result["normalized_value"] == pytest.approx(38)


def test_hours_to_minutes():
    result = normalize_measurement(2, "hours")
    assert result["normalized_value"] == pytest.approx(120)


def test_percentage_remains_percentage():
    result = normalize_measurement(5, "%")
    assert result["normalized_value"] == pytest.approx(5)
    assert result["dimension"] == "percentage"


def test_pressure_unit():
    result = normalize_measurement(120, "mmHg")
    assert result["normalized_value"] == pytest.approx(120)
    assert result["dimension"] == "pressure"


def test_compatible_mass_units():
    assert units_compatible("mg", "mcg") is True


def test_incompatible_dimensions():
    assert units_compatible("mg", "ml") is False


def test_concentration_requires_analyte():
    assert units_compatible("mg/dL", "mmol/L") is False


def test_months_not_converted_to_minutes():
    assert normalize_measurement(2, "months") is None


def test_years_not_converted_to_minutes():
    assert normalize_measurement(1, "years") is None


def test_unknown_unit_returns_none():
    assert normalize_measurement(5, "unknown_unit") is None


def test_nan_is_rejected():
    assert normalize_measurement(float("nan"), "mg") is None


def test_boolean_is_rejected():
    assert normalize_measurement(True, "mg") is None


def test_unit_aliases():
    assert canonicalize_unit("μg") == "mcg"
    assert canonicalize_unit("MG/DL") == "mg/dL"
    assert canonicalize_unit("°c") == "celsius"
