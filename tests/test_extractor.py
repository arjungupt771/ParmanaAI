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


def test_compound_claim_splits_whereas_and_while():
    sentence = "Metformin lowers blood glucose, whereas insulin therapy stimulates cellular uptake."
    claims = split_compound_claim(sentence)
    assert len(claims) == 2
    assert claims[0] == "Metformin lowers blood glucose"
    assert claims[1] == "insulin therapy stimulates cellular uptake."


def test_claim_classification_detects_negation_and_comparative():
    negation_claim = "Aspirin does not reduce mortality in healthy young adults."
    neg_meta = classify_claim(negation_claim)
    assert neg_meta["contains_negation"] is True

    comp_claim = "Lisinopril is more effective than Atenolol in reducing systolic blood pressure."
    comp_meta = classify_claim(comp_claim)
    assert comp_meta["contains_comparative_language"] is True
    assert comp_meta["relationship_type"] == "comparative"


def test_atomicity_evaluation_benchmark():
    """
    Manually annotated dataset evaluating claim extraction atomicity.
    """
    benchmark_dataset = [
        {
            "input": "Hypertension increases stroke risk, and diabetes accelerates cardiovascular damage.",
            "expected_claims": [
                "Hypertension increases stroke risk",
                "diabetes accelerates cardiovascular damage"
            ],
            "expected_atomicity": [True, True]
        },
        {
            "input": "Dr. Smith noted that patient blood pressure was 120/80 mmHg.",
            "expected_claims": [
                "Dr. Smith noted that patient blood pressure was 120/80 mmHg."
            ],
            "expected_atomicity": [False]
        },
        {
            "input": "Exercise lowers resting heart rate; however, excessive strain can cause arrhythmias.",
            "expected_claims": [
                "Exercise lowers resting heart rate",
                "excessive strain can cause arrhythmias"
            ],
            "expected_atomicity": [True, True]
        }
    ]

    for item in benchmark_dataset:
        records = extract_claim_records(item["input"])
        claim_texts = [r["claim_text"] for r in records]
        assert len(claim_texts) == len(item["expected_claims"])
        for expected in item["expected_claims"]:
            assert any(expected.lower() in ct.lower() for ct in claim_texts)


def test_markdown_lists_and_headers_atomicity():
    """
    Verify markdown lists (with/without periods) and headers are split
    into independent claims with proper hierarchy metadata.
    """
    markdown_doc = """
    # Treatment Guidelines for Type 2 Diabetes

    Metformin is recommended as initial pharmacotherapy for all diagnosed patients.

    ## First-Line Pharmacotherapy
    - Metformin decreases hepatic glucose production
    - SGLT2 inhibitors confer renal protection in CKD patients
    - GLP-1 receptor agonists reduce cardiovascular events by 14%

    ## Second-Line Considerations
    1. Sulfonylureas increase the risk of hypoglycemic episodes.
    2. Pioglitazone is contraindicated in patients with symptomatic heart failure.
    """

    records = extract_claim_records(markdown_doc)
    assert len(records) == 6

    # Verify header tracking
    first_claim = records[0]
    assert first_claim["parent_header"] == "Treatment Guidelines for Type 2 Diabetes"
    assert first_claim["is_list_item"] is False

    # Verify unordered list items under subheader
    list_claims = [r for r in records if r["parent_header"] == "First-Line Pharmacotherapy"]
    assert len(list_claims) == 3
    assert all(r["is_list_item"] is True for r in list_claims)
    assert "Metformin decreases hepatic glucose production" in list_claims[0]["claim_text"]
    assert "SGLT2 inhibitors confer renal protection" in list_claims[1]["claim_text"]
    assert list_claims[2]["numerical_values"] == ["14%"]

    # Verify numbered list items
    ordered_claims = [r for r in records if r["parent_header"] == "Second-Line Considerations"]
    assert len(ordered_claims) == 2
    assert ordered_claims[0]["list_index"] == 1
    assert ordered_claims[1]["list_index"] == 2
    assert ordered_claims[0]["is_list_item"] is True


def test_compound_claim_with_article_subject():
    """
    Verify compound clauses where the second subject begins with 'the', 'a', or 'an'.
    """
    sentence = "Metformin lowers blood sugar, but the risk of lactic acidosis remains low."
    claims = split_compound_claim(sentence)
    assert len(claims) == 2
    assert claims[0] == "Metformin lowers blood sugar"
    assert "the risk of lactic acidosis remains low" in claims[1]


def test_conditional_and_subordinate_clauses_stay_intact():
    """
    Ensure subordinate clauses (e.g. starting with 'if', 'when', 'unless')
    are not broken into nonsensical fragments.
    """
    sentence = "If glomerular filtration rate is below 30, metformin must be discontinued immediately."
    claims = split_compound_claim(sentence)
    assert len(claims) == 1
    assert claims[0] == sentence

