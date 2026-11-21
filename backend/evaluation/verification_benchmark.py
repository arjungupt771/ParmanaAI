"""
ParmanaAI - Clinical Claim Verification Benchmark Evaluator

Evaluates verification accuracy against manually annotated clinical ground truth:
- Precision, Recall, F1-score across SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE
- 3x3 Confusion Matrix
- Specialty breakdown and error analysis
"""

import json
import os
from typing import Any, Dict, List, Optional
from backend.agents.verifier import verify_claim

CLASSES = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Constructs a 3x3 confusion matrix: matrix[true_class][pred_class] = count.
    """
    matrix = {tc: {pc: 0 for pc in CLASSES} for tc in CLASSES}
    for t, p in zip(y_true, y_pred):
        t_clean = t if t in CLASSES else "INSUFFICIENT_EVIDENCE"
        p_clean = p if p in CLASSES else "INSUFFICIENT_EVIDENCE"
        matrix[t_clean][p_clean] += 1
    return matrix


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, Any]:
    """
    Calculate accuracy, per-class Precision/Recall/F1, and Macro/Micro averages.
    """
    total = len(y_true)
    if total == 0:
        return {}

    matrix = compute_confusion_matrix(y_true, y_pred)

    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    total_correct = 0

    for c in CLASSES:
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in CLASSES if other != c)
        fn = sum(matrix[c][other] for other in CLASSES if other != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(matrix[c].values())
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        total_correct += tp

    n_classes = len(CLASSES)
    accuracy = total_correct / total

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p / n_classes, 4),
        "macro_recall": round(macro_r / n_classes, 4),
        "macro_f1": round(macro_f1 / n_classes, 4),
        "micro_f1": round(accuracy, 4),  # Multi-class single-label micro-F1 == accuracy
        "per_class": per_class,
        "confusion_matrix": matrix
    }


def run_verification_benchmark(
    dataset_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute verification pipeline across gold-standard benchmark cases.
    """
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "verification_ground_truth.json"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    y_true: List[str] = []
    y_pred: List[str] = []
    case_results: List[Dict[str, Any]] = []

    for case in cases:
        claim_text = case["claim"]
        evidence_list = case["evidence"]
        gold_verdict = case["gold_verdict"]

        result = verify_claim(claim_text, evidence_list)
        # Use combined_verdict if present and valid; else fallback to verdict
        pred_verdict = result.get("combined_verdict") or result.get("verdict", "INSUFFICIENT_EVIDENCE")

        # Map non-standard signals to benchmark classes
        if pred_verdict not in CLASSES:
            if "CONFLICT" in pred_verdict or "MISMATCH" in pred_verdict:
                pred_verdict = "CONTRADICTED"
            else:
                pred_verdict = "INSUFFICIENT_EVIDENCE"

        y_true.append(gold_verdict)
        y_pred.append(pred_verdict)

        case_results.append({
            "id": case["id"],
            "specialty": case.get("specialty", "General"),
            "claim": claim_text,
            "gold_verdict": gold_verdict,
            "predicted_verdict": pred_verdict,
            "is_correct": (gold_verdict == pred_verdict),
            "confidence": result.get("confidence", 0.0),
            "numerical_verdict": result.get("numerical_assessment", {}).get("verdict")
        })

    metrics = compute_classification_metrics(y_true, y_pred)

    report = {
        "benchmark_name": data.get("metadata", {}).get("name", "Verification Benchmark"),
        "total_cases": len(cases),
        "overall_metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "micro_f1": metrics["micro_f1"]
        },
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "case_details": case_results
    }

    if output_report_path is None:
        output_report_path = os.path.join(
            os.path.dirname(__file__),
            "verification_benchmark_report.json"
        )

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_verification_benchmark()
    print("=" * 60)
    print("ParmanaAI Verification Benchmark Results")
    print("=" * 60)
    print(f"Total Cases:     {report['total_cases']}")
    print(f"Accuracy:        {report['overall_metrics']['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:  {report['overall_metrics']['macro_f1']:.4f}")
    print(f"Macro Precision: {report['overall_metrics']['macro_precision']:.4f}")
    print(f"Macro Recall:    {report['overall_metrics']['macro_recall']:.4f}")
    print("\nPer-Class Breakdown:")
    for c, stats in report["per_class_metrics"].items():
        print(f"  {c:<22} P: {stats['precision']:.3f} | R: {stats['recall']:.3f} | F1: {stats['f1']:.3f} (N={stats['support']})")
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    header = f"{'True \\ Pred':<24}" + "".join(f"{c[:10]:>12}" for c in CLASSES)
    print(header)
    for t in CLASSES:
        row_str = f"{t:<24}" + "".join(f"{report['confusion_matrix'][t][p]:>12}" for p in CLASSES)
        print(row_str)
