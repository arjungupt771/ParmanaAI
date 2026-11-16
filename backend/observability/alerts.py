"""
ParmanaAI - Clinical Safety & Latency Alerting

Monitors system health, latency thresholds, hallucination spikes,
and high-frequency emergency/refusal events.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AlertRule:
    rule_id: str
    name: str
    metric: str
    threshold: float
    severity: str  # "INFO", "WARNING", "CRITICAL"
    comparison: str  # "gt" | "lt"
    message_template: str


@dataclass
class AlertEvent:
    rule_id: str
    name: str
    severity: str
    message: str
    current_value: float
    threshold: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AlertManager:
    """
    Evaluates real-time metrics against clinical service-level objectives (SLOs).
    """

    DEFAULT_RULES = [
        AlertRule(
            rule_id="ALR-001",
            name="High Total Latency (p95)",
            metric="total_latency_p95",
            threshold=8000.0,  # 8.0s
            severity="WARNING",
            comparison="gt",
            message_template="Total pipeline p95 latency is {val:.1f}ms (threshold: {thresh:.1f}ms).",
        ),
        AlertRule(
            rule_id="ALR-002",
            name="High Hallucination Score",
            metric="average_hallucination_score",
            threshold=0.35,
            severity="CRITICAL",
            comparison="gt",
            message_template="Average clinical hallucination score is elevated at {val:.3f} (threshold: {thresh:.3f}).",
        ),
        AlertRule(
            rule_id="ALR-003",
            name="Elevated Refusal Rate",
            metric="refusal_rate",
            threshold=0.60,
            severity="WARNING",
            comparison="gt",
            message_template="Refusal rate exceeded expected baseline at {val:.1%} (threshold: {thresh:.1%}).",
        ),
    ]

    def __init__(self, rules: Optional[List[AlertRule]] = None):
        self.rules = rules or list(self.DEFAULT_RULES)
        self.active_alerts: List[AlertEvent] = []

    def evaluate_metrics(self, metrics_summary: Dict[str, Any]) -> List[AlertEvent]:
        """
        Evaluate current metrics summary against defined alert rules.
        """
        events: List[AlertEvent] = []

        total_lat_stats = metrics_summary.get("stage_latencies_ms", {}).get("total", {})
        metric_values: Dict[str, float] = {
            "total_latency_p95": total_lat_stats.get("p95", 0.0),
            "average_hallucination_score": metrics_summary.get("average_hallucination_score", 0.0),
            "refusal_rate": metrics_summary.get("refusal_rate", 0.0),
            "total_queries": float(metrics_summary.get("total_queries", 0)),
        }

        # Need at least 5 queries before alerting on percentage thresholds
        if metric_values["total_queries"] < 5:
            return []

        for rule in self.rules:
            val = metric_values.get(rule.metric)
            if val is None:
                continue

            triggered = False
            if rule.comparison == "gt" and val > rule.threshold:
                triggered = True
            elif rule.comparison == "lt" and val < rule.threshold:
                triggered = True

            if triggered:
                event = AlertEvent(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    severity=rule.severity,
                    message=rule.message_template.format(val=val, thresh=rule.threshold),
                    current_value=val,
                    threshold=rule.threshold,
                )
                events.append(event)
                self.active_alerts.append(event)

        return events

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recently recorded alert events as dicts."""
        return [
            {
                "rule_id": a.rule_id,
                "name": a.name,
                "severity": a.severity,
                "message": a.message,
                "current_value": a.current_value,
                "threshold": a.threshold,
                "timestamp": a.timestamp,
            }
            for a in self.active_alerts[-limit:]
        ]
