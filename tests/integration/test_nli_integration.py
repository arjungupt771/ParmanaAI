
import pytest

from backend.agents.nli_engine import predict_nli


@pytest.mark.integration
def test_real_nli_model_returns_valid_probabilities():

    result = predict_nli(
        premise=(
            "Smoking increases the risk of developing "
            "lung cancer."
        ),
        hypothesis=(
            "Smoking is associated with an increased "
            "risk of lung cancer."
        )
    )

    assert set(result.keys()) == {
        "entailment",
        "contradiction",
        "neutral"
    }

    assert all(
        0.0 <= value <= 1.0
        for value in result.values()
    )

    assert abs(sum(result.values()) - 1.0) < 0.001


@pytest.mark.integration
def test_real_nli_model_handles_unrelated_evidence():

    result = predict_nli(
        premise=(
            "The study examined agricultural soil "
            "and crop productivity."
        ),
        hypothesis=(
            "Smoking increases cardiovascular disease risk."
        )
    )

    assert result["neutral"] > result["entailment"]
