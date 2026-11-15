
"""
Track A: Oracle-context evaluation.

Reference PubMedQA context is explicitly supplied to ParmanaAI.
This measures generation and verification with reference evidence.

It does NOT measure the quality of the application's retriever.
"""

import json
import requests
from pathlib import Path

from backend.evaluation.load_pubmedqa import load_pubmedqa

API_URL = "http://127.0.0.1:8000/ask"
TIMEOUT = 60


def evaluate(dataset_path, limit=200):
    samples = load_pubmedqa(dataset_path, limit=limit)

    results = []
    failures = []

    total = len(samples)

    for i, sample in enumerate(samples, 1):
        reference_context = " ".join(
            sample.get("context_chunks", [])
        ).strip()

        payload = {
            "query": sample.get("question", ""),
            "external_context": reference_context
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=TIMEOUT
            )

            response.raise_for_status()
            output = response.json()

            if not isinstance(output, dict):
                raise ValueError("API response must be a JSON object")

        except (requests.RequestException, ValueError) as exc:
            failures.append({
                "index": i,
                "question": sample.get("question", ""),
                "error": str(exc)
            })

            print(f"[{i}/{total}] FAILED: {exc}")
            continue

        claims = output.get("claims", [])

        results.append({
            "evaluation_track": "oracle_context",
            "question": sample.get("question", ""),
            "reference_label": sample.get("label"),
            "answer": output.get("answer", ""),
            "num_claims": len(claims),
            "verified_claims": claims,
            "verifier_score": output.get(
                "hallucination_score"
            ),
            "relevance_score": output.get(
                "relevance_score"
            ),
            "reference_context_supplied": True
        })

        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] processed")

    report = {
        "evaluation_track": "oracle_context",
        "total_samples": total,
        "successful_samples": len(results),
        "failed_samples": len(failures),
        "coverage": (
            len(results) / total if total else 0.0
        ),
        "results": results,
        "failures": failures
    }

    return report


if __name__ == "__main__":
    dataset = "backend/evaluation/pubmedqa.json"

    report = evaluate(dataset, limit=200)

    output_path = Path(
        "backend/evaluation/oracle_context_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n===== ORACLE CONTEXT EVALUATION =====")
    print("Total samples:", report["total_samples"])
    print("Successful:", report["successful_samples"])
    print("Failed:", report["failed_samples"])
    print("Coverage:", round(report["coverage"], 4))
    print("Saved:", output_path)
