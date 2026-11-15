from backend.agents.extractor import (
    clean_markdown,
    split_sentences,
    split_compound_claim,
    extract_claims,
    extract_claim_records,
    classify_claim,
)


def test_empty_input():
    assert extract_claims("") == []
    assert extract_claims(None) == []
    assert extract_claim_records("   ") == []


def test_markdown_bullet_normalization():
    text = """
    - **Smoking** increases cardiovascular disease risk.
    - Obesity contributes to hypertension.
    """

    claims = extract_claims(text)

    assert len(claims) == 2
    assert "Smoking" in claims[0]
    assert "**" not in clean_markdown(text)


def test_multiple_sentences():
    text = (
        "Smoking increases cardiovascular risk. "
        "Physical activity may reduce blood pressure."
    )

    claims = extract_claims(text)

    assert len(claims) == 2
    assert claims[0].startswith("Smoking")
    assert claims[1].startswith("Physical activity")


def test_sentence_split_protects_abbreviations_and_decimals():
    sentences = split_sentences(
        "Dr. Lee reported a 2.5% increase. The result was significant."
    )

    assert sentences == [
        "Dr. Lee reported a 2.5% increase.",
        "The result was significant."
    ]


def test_compound_claim_splits_repeated_explicit_subjects():
    claims = split_compound_claim(
        "Smoking increases cardiovascular risk, and obesity contributes "
        "to hypertension."
    )

    assert claims == [
        "Smoking increases cardiovascular risk",
        "obesity contributes to hypertension."
    ]


def test_compound_claim_preserves_shared_subject_predicates():
    sentence = (
        "Smoking increases cardiovascular risk and raises blood pressure."
    )

    assert split_compound_claim(sentence) == [sentence]


def test_compound_claim_preserves_symptom_lists():
    sentence = "The condition causes fever, fatigue, and muscle pain."

    assert split_compound_claim(sentence) == [sentence]


def test_claim_classification_captures_linguistic_metadata():
    metadata = classify_claim(
        "Exercise may reduce blood pressure by 5% and prevents disease."
    )

    assert metadata["claim_type"] == "inferential"
    assert metadata["is_inferential"] is True
    assert metadata["is_definitive"] is True
    assert metadata["contains_causal_language"] is True
    assert metadata["contains_numerical_information"] is True
    assert metadata["numerical_values"] == ["5%"]


def test_records_preserve_metadata_and_original_sentence():
    original = (
        "Exercise may reduce blood pressure by 5%, and smoking causes "
        "cardiovascular disease."
    )

    records = extract_claim_records(original)

    assert len(records) == 2
    assert all(record["original_sentence"] == original for record in records)
    assert all(record["atomic"] is True for record in records)
    assert [record["claim_id"] for record in records] == [
        "CLM-0001",
        "CLM-0002",
    ]
    assert records[0]["is_inferential"] is True
    assert records[0]["contains_numerical_information"] is True
    assert records[0]["numerical_values"] == ["5%"]
    assert records[1]["contains_causal_language"] is True


def test_inferential_mode_excludes_definitive_claims():
    claims = extract_claims(
        "Smoking causes cardiovascular disease. "
        "Exercise may reduce blood pressure.",
        inferential=True,
    )

    assert claims == ["Exercise may reduce blood pressure."]


def test_duplicate_claims_are_removed():
    claims = extract_claims(
        "Smoking increases cardiovascular risk. "
        "Smoking increases cardiovascular risk."
    )

    assert claims == ["Smoking increases cardiovascular risk."]


from unittest.mock import patch

from backend.agents.verifier import verify_claim


def test_unrelated_contradiction_becomes_insufficient():

    evidence = [{
        "source_type": "guideline",
        "text": (
            "Fever commonly presents with chills, "
            "sweating, headache, and muscle pain."
        )
    }]

    scores = {
        "entailment": 0.0,
        "contradiction": 0.9773,
        "neutral": 0.0227
    }

    with patch(
        "backend.agents.verifier.predict_nli",
        return_value=scores
    ):
        result = verify_claim(
            "The patient has a broken leg.",
            evidence
        )

    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_relevant_contradiction_is_preserved():

    evidence = [{
        "source_type": "guideline",
        "text": (
            "Smoking does not increase the risk "
            "of lung cancer."
        )
    }]

    scores = {
        "entailment": 0.001,
        "contradiction": 0.98,
        "neutral": 0.019
    }

    with patch(
        "backend.agents.verifier.predict_nli",
        return_value=scores
    ):
        result = verify_claim(
            "Smoking increases the risk of lung cancer.",
            evidence
        )

    assert result["verdict"] == "CONTRADICTED"