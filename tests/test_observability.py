"""
ParmanaAI - Phase 11 Observability & Monitoring Tests

Tests:
- Structured JSON logging with PII de-identification
- Distributed trace correlation (request_id, trace_id)
- In-memory metrics collection and Prometheus export
- Alert rule evaluation and notification
"""

import json
import logging
import pytest

from backend.observability.structured_logger import (
    StructuredJSONFormatter,
    request_id_ctx,
    trace_id_ctx,
)
from backend.observability.metrics import MetricsCollector
from backend.observability.alerts import AlertManager, AlertRule


def test_structured_json_logger_with_pii_sanitization():
    formatter = StructuredJSONFormatter()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=25,
        msg="Processing inquiry for Patient: John Doe with SSN 123-45-6789.",
        args=(),
        exc_info=None,
    )

    request_id_ctx.set("req-test-999")
    trace_id_ctx.set("trace-test-888")

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["level"] == "INFO"
    assert data["request_id"] == "req-test-999"
    assert data["trace_id"] == "trace-test-888"
    assert "[REDACTED_NAME]" in data["message"]
    assert "[REDACTED_SSN]" in data["message"]
    assert "123-45-6789" not in data["message"]
    assert "John Doe" not in data["message"]


def test_metrics_collector():
    collector = MetricsCollector()

    # Record normal queries
    collector.record_query(
        mode="FACTUAL",
        is_refusal=False,
        latency_ms=120.0,
        hallucination_score=0.0,
        claims=[{"verdict": "SUPPORTED"}, {"verdict": "SUPPORTED"}],
    )
    collector.record_query(
        mode="FACTUAL",
        is_refusal=False,
        latency_ms=180.0,
        hallucination_score=0.1,
        claims=[{"verdict": "SUPPORTED"}],
    )
    # Record emergency query
    collector.record_query(
        mode="EMERGENCY",
        is_refusal=True,
        is_emergency=True,
        refusal_reason="EMERGENCY_TRIAGE",
        latency_ms=15.0,
    )

    summary = collector.get_metrics_summary()

    assert summary["total_queries"] == 3
    assert summary["successful_queries"] == 2
    assert summary["refused_queries"] == 1
    assert summary["emergency_queries"] == 1
    assert summary["mode_distribution"]["FACTUAL"] == 2
    assert summary["mode_distribution"]["EMERGENCY"] == 1
    assert summary["claim_verdict_distribution"]["SUPPORTED"] == 3

    # Check latency percentiles
    total_stats = summary["stage_latencies_ms"]["total"]
    assert total_stats["min"] == 15.0
    assert total_stats["max"] == 180.0
    assert total_stats["p50"] > 0

    # Check Prometheus output
    prom_text = collector.get_prometheus_format()
    assert "parmana_queries_total 3" in prom_text
    assert "parmana_queries_successful_total 2" in prom_text
    assert 'parmana_query_mode_total{mode="FACTUAL"} 2' in prom_text


def test_alert_manager_evaluation():
    manager = AlertManager()

    # Metrics below threshold -> no alerts
    normal_metrics = {
        "total_queries": 10,
        "refusal_rate": 0.1,
        "average_hallucination_score": 0.05,
        "stage_latencies_ms": {"total": {"p95": 200.0}},
    }
    alerts = manager.evaluate_metrics(normal_metrics)
    assert len(alerts) == 0

    # High hallucination metrics -> alert triggered
    critical_metrics = {
        "total_queries": 10,
        "refusal_rate": 0.1,
        "average_hallucination_score": 0.48,  # Threshold is 0.35
        "stage_latencies_ms": {"total": {"p95": 250.0}},
    }
    alerts = manager.evaluate_metrics(critical_metrics)
    assert len(alerts) == 1
    assert alerts[0].rule_id == "ALR-002"
    assert alerts[0].severity == "CRITICAL"
    assert "hallucination" in alerts[0].message.lower()

    recent = manager.get_recent_alerts()
    assert len(recent) == 1
