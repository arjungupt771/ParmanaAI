"""
End-to-end evaluation of the LIVE retrieval pipeline (no external_context).

Complements evaluate_system.py, which tests generation+verification given
ground-truth context. This script tests the part that's actually exercised
by real users: does our own guideline/abstract retriever find useful
evidence for a real question, and how good is the resulting answer.

Since our knowledge base only covers fever / diabetes / cardiovascular
topics, we filter PubMedQA down to questions plausibly about those topics
before running - otherwise most questions would correctly return "no
evidence found", which would measure corpus narrowness, not retrieval
quality.
"""
import json
import time
import requests

from backend.evaluation.load_pubmedqa import load_pubmedqa

API_URL = "http://127.0.0.1:8000/ask"

TOPIC_KEYWORDS = {
    "fever": ["fever", "febrile", "temperature", "pyrexia"],
    "diabetes": ["diabet", "insulin", "glyc", "hyperglycemi", "hyperglycaemi", "blood sugar"],
    "cardiovascular": ["cardiovascular", "cardiac", "heart", "coronary", "myocard"],
}


def matches_topic(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kws in TOPIC_KEYWORDS.values() for kw in kws)


def filter_relevant_questions(dataset_path, scan_limit=1000):
    all_samples = load_pubmedqa(dataset_path, limit=scan_limit)
    return [s for s in all_samples if matches_topic(s.get("question", ""))]



def evaluate_retrieval(dataset_path, scan_limit=1000, delay=0.3):
    samples = filter_relevant_questions(
        dataset_path,
        scan_limit=scan_limit
    )

    print(
        f"Found {len(samples)} relevant questions "
        f"out of {scan_limit} scanned."
    )

    results = []
    failures = []

    total = len(samples)

    for i, sample in enumerate(samples, 1):

        # Intentionally no external_context.
        payload = {
            "query": sample.get("question", "")
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            output = response.json()

            if not isinstance(output, dict):
                raise ValueError("Invalid API response")

        except (requests.RequestException, ValueError) as exc:
            failures.append({
                "index": i,
                "question": sample.get("question", ""),
                "error": str(exc)
            })

            print(f"[{i}/{total}] FAILED: {exc}")
            time.sleep(delay)
            continue

        answer = output.get("answer", "")
        claims = output.get("claims", [])

        no_evidence = (
            answer == "No verified medical evidence found."
        )

        results.append({
            "evaluation_track": "end_to_end_rag",
            "question": sample.get("question", ""),
            "reference_label": sample.get("label"),
            "answer": answer,
            "num_claims": len(claims),
            "verified_claims": claims,
            "verifier_score": output.get(
                "hallucination_score"
            ),
            "relevance_score": output.get(
                "relevance_score"
            ),
            "no_evidence_found": no_evidence,
            "reference_context_supplied": False
        })

        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] processed")

        time.sleep(delay)

    no_evidence_count = sum(
        1 for item in results
        if item["no_evidence_found"]
    )

    report = {
        "evaluation_track": "end_to_end_rag",
        "total_questions": total,
        "successful_questions": len(results),
        "failed_questions": len(failures),
        "coverage": (
            len(results) / total if total else 0.0
        ),
        "no_evidence_count": no_evidence_count,
        "no_evidence_rate": (
            no_evidence_count / len(results)
            if results else 0.0
        ),
        "results": results,
        "failures": failures
    }

    return report



if __name__ == "__main__":
    report = evaluate_retrieval(
        "backend/evaluation/pubmedqa.json",
        scan_limit=1000
    )

    with open(
        "backend/evaluation/retrieval_results_v2.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\n===== END-TO-END RAG EVALUATION =====")
    print("Total:", report["total_questions"])
    print("Successful:", report["successful_questions"])
    print("Failed:", report["failed_questions"])
    print("Coverage:", round(report["coverage"], 4))
    print(
        "No evidence rate:",
        round(report["no_evidence_rate"], 4)
    )

    print("Saved: retrieval_results_v2.json")
