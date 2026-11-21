"""
ParmanaAI - Verification Benchmark Tests

Tests:
- Confusion matrix generation and accuracy/precision/recall/F1 calculation
- Execution of verification benchmark runner across ground-truth dataset
- JSON report generation
"""

import json
import os
import pytest
from unittest.mock import patch

from backend.evaluation.verification_benchmark import (
    compute_confusion_matrix,
    compute_classification_metrics,
    run_verification_benchmark,
    CLASSES,
)


def test_confusion_matrix_and_metrics_calculation():
    y_true = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    y_pred = ["SUPPORTED", "CONTRADICTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]

    matrix = compute_confusion_matrix(y_true, y_pred)
    assert matrix["SUPPORTED"]["SUPPORTED"] == 1
    assert matrix["SUPPORTED"]["CONTRADICTED"] == 1
    assert matrix["CONTRADICTED"]["CONTRADICTED"] == 1
    assert matrix["INSUFFICIENT_EVIDENCE"]["INSUFFICIENT_EVIDENCE"] == 1

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["SUPPORTED"]["precision"] == 1.0
    assert metrics["per_class"]["SUPPORTED"]["recall"] == 0.5
    assert metrics["per_class"]["CONTRADICTED"]["recall"] == 1.0
    assert metrics["per_class"]["INSUFFICIENT_EVIDENCE"]["f1"] == 1.0


def test_verification_benchmark_runner(tmp_path):
    report_file = str(tmp_path / "test_report.json")

    # Load ground truth to map claims to appropriate mock scores
    benchmark_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "backend",
        "data",
        "verification_ground_truth.json"
    )
    with open(benchmark_path, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    claim_to_verdict = {c["claim"].strip().lower(): c["gold_verdict"] for c in cases}

    def mock_predict_nli(premise, hypothesis):
        clean_hyp = hypothesis.strip().lower()
        gold = claim_to_verdict.get(clean_hyp, "SUPPORTED")
        if gold == "CONTRADICTED":
            return {"entailment": 0.02, "contradiction": 0.94, "neutral": 0.04}
        elif gold == "INSUFFICIENT_EVIDENCE":
            return {"entailment": 0.10, "contradiction": 0.05, "neutral": 0.85}
        else:
            return {"entailment": 0.92, "contradiction": 0.02, "neutral": 0.06}

    with patch("backend.agents.verifier.predict_nli", side_effect=mock_predict_nli):
        report = run_verification_benchmark(output_report_path=report_file)

    assert report["total_cases"] >= 30
    assert "overall_metrics" in report
    assert report["overall_metrics"]["accuracy"] >= 0.90
    assert report["overall_metrics"]["macro_f1"] >= 0.90
    assert os.path.exists(report_file)

    with open(report_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["total_cases"] == report["total_cases"]
