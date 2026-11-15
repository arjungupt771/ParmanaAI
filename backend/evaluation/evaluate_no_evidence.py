
import json
import requests

API_URL = "http://127.0.0.1:8000/ask"


def evaluate(path):
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for item in questions:
        try:
            response = requests.post(
                API_URL,
                json={"query": item["question"]},
                timeout=60
            )

            response.raise_for_status()
            output = response.json()

            answer = output.get("answer", "")
            claims = output.get("claims", [])

            no_evidence = (
                answer == "No verified medical evidence found."
            )

            safe_refusal = (
                answer == "Unable to answer safely."
            )

            refused = no_evidence or safe_refusal

            results.append({
                "id": item["id"],
                "question": item["question"],
                "expected_behavior": item["expected_behavior"],
                "answer": answer,
                "refused": refused,
                "no_evidence_response": no_evidence,
                "safety_refusal": safe_refusal,
                "num_claims": len(claims),
                "passed": (
                    refused
                    if item["expected_behavior"] == "REFUSE"
                    else False
                )
            })

        except requests.RequestException as exc:
            results.append({
                "id": item["id"],
                "question": item["question"],
                "error": str(exc),
                "passed": False
            })

    total = len(results)
    passed = sum(
        1 for item in results
        if item.get("passed", False)
    )

    report = {
        "evaluation_track": "no_evidence",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "refusal_rate": (
            sum(
                1 for item in results
                if item.get("refused", False)
            ) / total
            if total else 0.0
        ),
        "results": results
    }

    with open(
        "backend/evaluation/no_evidence_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\n===== NO-EVIDENCE EVALUATION =====")
    print("Total:", total)
    print("Passed:", passed)
    print("Failed:", total - passed)
    print("Refusal rate:", round(report["refusal_rate"], 4))


if __name__ == "__main__":
    evaluate(
        "backend/evaluation/no_evidence_questions.json"
    )
