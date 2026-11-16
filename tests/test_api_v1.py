"""
ParmanaAI - Phase 12 Backend API v1 Endpoint Integration Tests

Directly tests API v1 endpoints with complete schema validation,
database persistence, and metric collection:
- get_health
- get_metrics (JSON & Prometheus)
- get_alerts
- ingest_document
- ask_question with verification and database persistence
- list_history and get_query_detail
- get_benchmarks
"""

from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException

from backend.api.v1.schemas import (
    QueryRequest,
    QueryResponse,
    IngestionRequest,
    HealthResponse,
)
from backend.api.v1.router import (
    ask_question,
    get_health,
    get_metrics,
    get_alerts,
    list_history,
    get_query_detail,
    ingest_document,
    get_benchmarks,
)
from backend.database.repository import QueryRepository
from backend.database.db import init_db
from backend.ingestion.knowledge_base_manager import KnowledgeBaseManager


@pytest.fixture
def repo(tmp_path):
    test_db = str(tmp_path / "api_test.db")
    init_db(test_db)
    return QueryRepository(db_path=test_db)


@pytest.fixture
def mock_orchestrator():
    mock_orch = MagicMock()
    mock_orch.run.return_value = {
        "answer": "Metformin reduces hepatic gluconeogenesis in type 2 diabetes.",
        "mode": "FACTUAL",
        "hallucination_score": 0.0,
        "is_emergency": False,
        "is_refusal": False,
        "claims": [
            {
                "claim_id": "CLM-TEST-001",
                "claim_text": "Metformin reduces hepatic gluconeogenesis in type 2 diabetes.",
                "verdict": "SUPPORTED",
                "combined_verdict": "SUPPORTED",
                "primary_evidence_id": "ev-test-1",
            }
        ],
        "evidence": [
            {
                "evidence_id": "ev-test-1",
                "document_id": "doc-t2dm-01",
                "chunk_id": "chunk-1",
                "source_type": "guideline",
                "score": 0.95,
                "text": "Metformin decreases hepatic glucose production.",
                "provenance": {"source": "WHO"},
            }
        ],
    }
    return mock_orch


def test_api_v1_health():
    health = get_health()
    assert isinstance(health, HealthResponse)
    assert health.status == "healthy"
    assert "database" in health.components
    assert "api" in health.components


def test_api_v1_metrics():
    # JSON format
    metrics_json = get_metrics(format="json")
    assert "total_queries" in metrics_json
    assert "uptime_seconds" in metrics_json

    # Prometheus format
    prom_res = get_metrics(format="prometheus")
    assert "parmana_queries_total" in prom_res.body.decode()


def test_api_v1_ask_lifecycle_and_history(repo, mock_orchestrator):
    req = QueryRequest(
        query="What is the mechanism of metformin in type 2 diabetes?",
        external_context=None,
        session_id="sess-test-88",
    )

    resp: QueryResponse = ask_question(req=req, repo=repo, orch=mock_orchestrator)

    assert resp.request_id.startswith("req-")
    assert "Metformin" in resp.answer
    assert resp.query_mode == "FACTUAL"
    assert resp.hallucination_score == 0.0
    assert len(resp.claims) == 1
    assert resp.claims[0].verdict == "SUPPORTED"
    assert len(resp.evidence) == 1
    assert resp.latency_ms > 0

    # Verify query was saved in history
    history = list_history(limit=10, offset=0, repo=repo)
    assert len(history["queries"]) == 1
    saved_query_id = history["queries"][0]["id"]

    # Verify query detail with joined claims and evidence
    detail = get_query_detail(query_id=saved_query_id, repo=repo)
    assert detail["id"] == saved_query_id
    assert len(detail["claims"]) == 1
    assert detail["claims"][0]["claim_text"] == "Metformin reduces hepatic gluconeogenesis in type 2 diabetes."
    assert len(detail["evidence"]) == 1
    assert detail["evidence"][0]["document_id"] == "doc-t2dm-01"


def test_api_v1_query_detail_not_found(repo):
    with pytest.raises(HTTPException) as exc_info:
        get_query_detail(query_id="non-existent-query-id", repo=repo)
    assert exc_info.value.status_code == 404


def test_api_v1_ingest():
    kb_mgr = KnowledgeBaseManager()
    req = IngestionRequest(
        content="Clinical practice guidelines for hypertension management: target SBP < 130 mmHg.",
        format="txt",
        document_title="2024 Hypertension Guidelines",
        source_type="guideline",
    )

    ingest_resp = ingest_document(req=req, kb_mgr=kb_mgr)
    assert ingest_resp.status == "SUCCESS"
    assert ingest_resp.chunks_indexed >= 1
    assert ingest_resp.document_id.startswith("doc-")


def test_api_v1_alerts():
    alerts = get_alerts()
    assert "active_alerts" in alerts
    assert "evaluated_at" in alerts


def test_api_v1_benchmarks():
    benchmarks = get_benchmarks()
    assert isinstance(benchmarks, dict)
    assert "total_experiments" in benchmarks or "message" in benchmarks
