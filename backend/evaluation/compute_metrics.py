
"""
ParmanaAI evaluation metrics.

Reports raw claim verdict distributions.
Does not treat heuristic verifier scores as
validated hallucination probabilities.
"""

import json
import argparse
from pathlib import Path

VERDICTS = [
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
    "NO_EVIDENCE_PROVIDED"
]


def compute_metrics(report):
    results = report.get("results", [])

    verdict_counts = {
        verdict: 0 for verdict in VERDICTS
    }

    total_claims = 0
    samples_with_claims = 0
    samples_without_claims = 0
    scored_samples = 0
    score_sum = 0.0

    for item in results:
        claims = item.get("verified_claims", [])

        if claims:
            samples_with_claims += 1
        else:
            samples_without_claims += 1

        for claim in claims:
            verdict = claim.get(
                "verdict",
                "NO_EVIDENCE_PROVIDED"
            )

            if verdict not in verdict_counts:
                verdict_counts[verdict] = 0

            verdict_counts[verdict] += 1
            total_claims += 1

        score = item.get("verifier_score")

        if isinstance(score, (int, float)):
            score_sum += float(score)
            scored_samples += 1

    total_samples = len(results)

    verdict_rates = {
        verdict: (
            count / total_claims
            if total_claims else 0.0
        )
        for verdict, count in verdict_counts.items()
    }

    return {
        "evaluation_track": report.get(
            "evaluation_track",
            "unknown"
        ),
        "total_samples": report.get(
            "total_samples",
            total_samples
        ),
        "successful_samples": total_samples,
        "failed_samples": report.get(
            "failed_samples",
            0
        ),
        "sample_coverage": report.get(
            "coverage",
            1.0
        ),
        "total_claims": total_claims,
        "verdict_counts": verdict_counts,
        "verdict_rates": verdict_rates,
        "samples_with_claims": samples_with_claims,
        "samples_without_claims": samples_without_claims,
        "mean_legacy_verifier_score": (
            score_sum / scored_samples
            if scored_samples else None
        ),
        "scored_samples": scored_samples
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="backend/evaluation/oracle_context_results.json"
    )

    parser.add_argument(
        "--output",
        default=None
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    with input_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        report = json.load(f)

    metrics = compute_metrics(report)

    print("\n===== PARMANA AI EVALUATION =====")
    print("Track:", metrics["evaluation_track"])
    print("Samples:", metrics["successful_samples"])
    print("Failed:", metrics["failed_samples"])
    print("Claims:", metrics["total_claims"])

    print("\nVerdict distribution:")

    for verdict, count in metrics["verdict_counts"].items():
        rate = metrics["verdict_rates"][verdict]

        print(
            f"{verdict}: {count} "
            f"({rate:.2%})"
        )

    print(
        "\nMean legacy verifier score:",
        metrics["mean_legacy_verifier_score"]
    )

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(
            input_path.stem + "_metrics.json"
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics saved:", output_path)


if __name__ == "__main__":
    main()
