"""
ParmanaAI - Phase 10 Benchmarking & Ablation Suite Tests

Tests:
- Execution of ablation configurations (Experiments A through I)
- Metric calculations (Recall@3, MRR, Refusal Accuracy, Hallucination Score)
- Markdown table generation
- Results persistence
"""

import json
import os
import pytest

from backend.evaluation.ablation_suite import (
    AblationBenchmarkSuite,
    AblationExperimentConfig,
    DEFAULT_ABLATION_CONFIGS,
    run_and_save_ablation_benchmarks,
)


def test_ablation_suite_single_experiment():
    suite = AblationBenchmarkSuite()
    config = DEFAULT_ABLATION_CONFIGS[0]  # Exp A: Direct LLM
    result = suite.run_single_experiment(config)

    assert result["experiment_id"] == "Exp_A"
    assert result["retrieval_recall_at_3"] == 0.0
    assert result["hallucination_score"] > 0.0
    assert "latency_per_query_ms" in result


def test_ablation_suite_hybrid_and_refusal_experiment():
    suite = AblationBenchmarkSuite()
    config_h = [c for c in DEFAULT_ABLATION_CONFIGS if c.experiment_id == "Exp_H"][0]
    result_h = suite.run_single_experiment(config_h)

    assert result_h["experiment_id"] == "Exp_H"
    assert result_h["retrieval_recall_at_3"] > 0.7
    assert result_h["refusal_accuracy"] == 1.0  # Refuses emergencies, injections, unanswerables
    assert result_h["hallucination_score"] <= 0.2


def test_ablation_suite_all_experiments_and_table_generation():
    suite = AblationBenchmarkSuite()
    # Test running a subset of 3 experiments for fast unit test
    test_configs = [
        DEFAULT_ABLATION_CONFIGS[0],  # Exp A
        DEFAULT_ABLATION_CONFIGS[3],  # Exp D: Hybrid
        DEFAULT_ABLATION_CONFIGS[7],  # Exp H: Refusal
    ]
    results = suite.run_all(configs=test_configs)

    assert len(results) == 3
    md_table = suite.generate_markdown_table(results)
    assert "| Experiment | Configuration |" in md_table
    assert "Exp_A" in md_table
    assert "Exp_D" in md_table
    assert "Exp_H" in md_table


def test_ablation_benchmark_save(tmp_path):
    output_file = str(tmp_path / "test_ablation_results.json")
    data = run_and_save_ablation_benchmarks(output_path=output_file)

    assert os.path.exists(output_file)
    assert data["total_experiments"] == len(DEFAULT_ABLATION_CONFIGS)
    assert "summary_markdown_table" in data

    with open(output_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded["results"]) == 9
