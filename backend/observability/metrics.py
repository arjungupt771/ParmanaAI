"""
ParmanaAI - Clinical Metrics & Performance Observability

Provides thread-safe metrics collection across all query modes, pipeline stages,
verification outcomes, and refusal categories.
"""

from collections import defaultdict
import math
import threading
import time
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """
    Central in-memory metrics registry for clinical QA and agent pipelines.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with getattr(self, "_lock", threading.Lock()):
            self.start_time = time.time()
            self.total_queries: int = 0
            self.successful_queries: int = 0
            self.refused_queries: int = 0
            self.emergency_queries: int = 0

            self.mode_counts: Dict[str, int] = defaultdict(int)
            self.refusal_reasons: Dict[str, int] = defaultdict(int)
            self.claim_verdicts: Dict[str, int] = defaultdict(int)

            self.stage_latencies: Dict[str, List[float]] = defaultdict(list)
            self.hallucination_scores: List[float] = []
            self.token_usages: Dict[str, int] = defaultdict(int)

    def record_query(
        self,
        mode: str,
        is_refusal: bool = False,
        is_emergency: bool = False,
        refusal_reason: Optional[str] = None,
        latency_ms: Optional[float] = None,
        hallucination_score: Optional[float] = None,
        claims: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Record top-level query outcome and verification signals.
        """
        with self._lock:
            self.total_queries += 1
            self.mode_counts[mode] += 1

            if is_emergency:
                self.emergency_queries += 1

            if is_refusal:
                self.refused_queries += 1
                if refusal_reason:
                    self.refusal_reasons[refusal_reason] += 1
            else:
                self.successful_queries += 1

            if latency_ms is not None:
                self.stage_latencies["total"].append(latency_ms)

            if hallucination_score is not None:
                self.hallucination_scores.append(hallucination_score)

            if claims:
                for c in claims:
                    verdict = c.get("verdict") or c.get("combined_verdict") or "UNKNOWN"
                    self.claim_verdicts[verdict] += 1

    def record_stage_latency(self, stage: str, latency_ms: float) -> None:
        """
        Record execution duration of a specific multi-agent pipeline node.
        """
        with self._lock:
            self.stage_latencies[stage].append(latency_ms)

    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Compute min, avg, p50, p95, p99, max."""
        if not values:
            return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def pct(p: float) -> float:
            idx = int(math.ceil((p / 100.0) * n)) - 1
            return round(sorted_vals[max(0, min(idx, n - 1))], 2)

        return {
            "min": round(sorted_vals[0], 2),
            "avg": round(sum(sorted_vals) / n, 2),
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "max": round(sorted_vals[-1], 2),
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Produce a JSON-serializable diagnostic performance summary.
        """
        with self._lock:
            uptime_seconds = round(time.time() - self.start_time, 2)
            latency_stats = {
                stage: self._calculate_percentiles(lats)
                for stage, lats in self.stage_latencies.items()
            }

            avg_hallucination = (
                round(sum(self.hallucination_scores) / len(self.hallucination_scores), 4)
                if self.hallucination_scores
                else 0.0
            )

            return {
                "uptime_seconds": uptime_seconds,
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "refused_queries": self.refused_queries,
                "emergency_queries": self.emergency_queries,
                "refusal_rate": round(self.refused_queries / max(self.total_queries, 1), 4),
                "mode_distribution": dict(self.mode_counts),
                "refusal_reasons": dict(self.refusal_reasons),
                "claim_verdict_distribution": dict(self.claim_verdicts),
                "average_hallucination_score": avg_hallucination,
                "stage_latencies_ms": latency_stats,
            }

    def get_prometheus_format(self) -> str:
        """
        Format metrics as standard Prometheus exposition text.
        """
        with self._lock:
            lines = [
                "# HELP parmana_queries_total Total clinical queries received",
                "# TYPE parmana_queries_total counter",
                f"parmana_queries_total {self.total_queries}",
                "# HELP parmana_queries_successful_total Total successfully answered queries",
                "# TYPE parmana_queries_successful_total counter",
                f"parmana_queries_successful_total {self.successful_queries}",
                "# HELP parmana_queries_refused_total Total refused/abstained queries",
                "# TYPE parmana_queries_refused_total counter",
                f"parmana_queries_refused_total {self.refused_queries}",
                "# HELP parmana_queries_emergency_total Total emergency queries detected",
                "# TYPE parmana_queries_emergency_total counter",
                f"parmana_queries_emergency_total {self.emergency_queries}",
            ]

            for mode, count in self.mode_counts.items():
                lines.append(f'parmana_query_mode_total{{mode="{mode}"}} {count}')

            for verdict, count in self.claim_verdicts.items():
                lines.append(f'parmana_claim_verdicts_total{{verdict="{verdict}"}} {count}')

            return "\n".join(lines) + "\n"


global_metrics = MetricsCollector()
