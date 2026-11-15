
from backend.agents.numerical_extractor import (
    extract_numerical_mentions
)


def test_extracts_temperature_with_comparator():
    result = extract_numerical_mentions(
        "Fever is defined as ≥ 38°C."
    )

    assert len(result) == 1
    assert result[0]["value"] == 38.0
    assert result[0]["comparator"] == ">="
    assert result[0]["normalized_unit"] == "celsius"
    assert result[0]["category"] == "temperature"


def test_extracts_percentage():
    result = extract_numerical_mentions(
        "Risk increased by 5%."
    )

    assert len(result) == 1
    assert result[0]["value"] == 5.0
    assert result[0]["category"] == "percentage"


def test_extracts_numerical_range():
    result = extract_numerical_mentions(
        "Blood pressure was 120-140 mmHg."
    )

    assert len(result) == 1
    assert result[0]["value"] == 120.0
    assert result[0]["upper_value"] == 140.0
    assert result[0]["normalized_unit"] == "mmHg"


def test_extracts_mass_units():
    result = extract_numerical_mentions(
        "The patient received 500 mg."
    )

    assert len(result) == 1
    assert result[0]["value"] == 500.0
    assert result[0]["normalized_unit"] == "mg"
    assert result[0]["category"] == "mass"


def test_distinguishes_mg_and_mcg():
    result = extract_numerical_mentions(
        "The doses were 5 mg and 5 mcg."
    )

    assert len(result) == 2
    assert result[0]["normalized_unit"] == "mg"
    assert result[1]["normalized_unit"] == "mcg"


def test_extracts_statistical_measure():
    result = extract_numerical_mentions(
        "The study reported OR = 1.5."
    )

    assert len(result) == 1
    assert result[0]["statistic"] == "OR"
    assert result[0]["value"] == 1.5


def test_extracts_confidence_interval_level():
    result = extract_numerical_mentions(
        "The study reported a 95% CI."
    )

    assert len(result) == 1
    assert result[0]["category"] == "confidence_level"
    assert result[0]["value"] == 95.0


def test_extracts_iso_date():
    result = extract_numerical_mentions(
        "The study was published on 2024-05-10."
    )

    assert len(result) == 1
    assert result[0]["category"] == "date"
    assert result[0]["value"] == "2024-05-10"


def test_extracts_multiple_values_in_order():
    result = extract_numerical_mentions(
        "The patient was 65 years old and received 500 mg."
    )

    assert len(result) == 2
    assert result[0]["value"] == 65.0
    assert result[0]["normalized_unit"] == "years"
    assert result[1]["value"] == 500.0
    assert result[1]["normalized_unit"] == "mg"


def test_empty_input():
    assert extract_numerical_mentions("") == []
    assert extract_numerical_mentions(None) == []


def test_invalid_iso_date_is_ignored():
    result = extract_numerical_mentions(
        "The date was 2024-15-45."
    )

    assert result == []


def test_preserves_original_text_positions():
    text = "Fever threshold is 38°C."

    result = extract_numerical_mentions(text)

    mention = result[0]

    assert text[mention["start"]:mention["end"]] == mention["raw_text"]
