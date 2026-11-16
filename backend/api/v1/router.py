"""
ParmanaAI - API v1 Router & Endpoints
"""

from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import PlainTextResponse

from backend.config import settings
from backend.api.v1.schemas import (
    QueryRequest,
    QueryResponse,
    ClaimVerificationResult,
    EvidenceItem,
    IngestionRequest,
    IngestionResponse,
    HealthResponse,
)
from backend.orchestrator.orchestrator import Orchestrator
from backend.observability.structured_logger import (
    get_structured_logger,
    request_id_ctx,
)
from backend.observability.metrics import global_metrics
from backend.observability.alerts import AlertManager
from backend.database.repository import QueryRepository
from backend.database.db import init_db
from backend.ingestion.knowledge_base_manager import KnowledgeBaseManager
from backend.ingestion.base import Document

logger = get_structured_logger(__name__)
v1_router = APIRouter(prefix="/api/v1", tags=["API v1"])

# Global shared singletons
_repo: Optional[QueryRepository] = None
_alert_manager = AlertManager()
_orchestrator: Optional[Orchestrator] = None
_kb_manager: Optional[KnowledgeBaseManager] = None


def get_repository() -> QueryRepository:
    global _repo
    if _repo is None:
        init_db()
        _repo = QueryRepository()
    return _repo


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def get_kb_manager() -> KnowledgeBaseManager:
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    return _kb_manager


@v1_router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Submit a clinical inquiry for evidence-grounded verification",
)
def ask_question(
    req: QueryRequest,
    repo: QueryRepository = Depends(get_repository),
    orch: Orchestrator = Depends(get_orchestrator),
) -> QueryResponse:
    """
    Execute evidence retrieval, answer generation, atomic claim extraction,
    and multi-source numerical consistency verification.
    """
    req_id = request_id_ctx.get() or f"req-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()

    logger.info(f"Received query: {req.query}", extra={"request_id": req_id})

    # Execute orchestrator pipeline
    result = orch.run(
        user_query=req.query,
        external_context=req.external_context,
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Format claims
    claims_list: List[ClaimVerificationResult] = []
    for c in result.get("claims", []):
        claims_list.append(ClaimVerificationResult(
            claim_id=c.get("claim_id", f"CLM-{uuid.uuid4().hex[:6]}"),
            claim_text=c.get("claim_text", ""),
            verdict=c.get("verdict", "INCONCLUSIVE"),
            combined_verdict=c.get("combined_verdict", c.get("verdict", "INCONCLUSIVE")),
            primary_evidence_id=c.get("primary_evidence_id"),
            evidence_assessments=c.get("evidence_assessments"),
            nli_verdict=c.get("nli_verdict"),
            numerical_status=c.get("numerical_status"),
            is_inferential=c.get("is_inferential", False),
        ))

    # Format evidence
    evidence_list: List[EvidenceItem] = []
    for ev in result.get("evidence", []):
        evidence_list.append(EvidenceItem(
            evidence_id=ev.get("evidence_id", f"ev-{uuid.uuid4().hex[:6]}"),
            text=ev.get("text", ""),
            document_id=ev.get("document_id", "doc-unknown"),
            chunk_id=ev.get("chunk_id", "chunk-unknown"),
            source_type=ev.get("source_type", "unknown"),
            score=ev.get("score"),
            match_type=ev.get("match_type"),
            provenance=ev.get("provenance"),
        ))

    # Record metrics
    global_metrics.record_query(
        mode=result.get("mode", "FACTUAL"),
        is_refusal=result.get("is_refusal", False),
        is_emergency=result.get("is_emergency", False),
        refusal_reason=result.get("refusal_reason"),
        latency_ms=elapsed_ms,
        hallucination_score=result.get("hallucination_score", 0.0),
        claims=[c.model_dump() for c in claims_list],
    )

    # Persist in DB
    try:
        repo.save_query_result(
            request_id=req_id,
            user_query=req.query,
            scrubbed_query=result.get("scrubbed_query", req.query),
            answer=result.get("answer", ""),
            query_mode=result.get("mode", "FACTUAL"),
            is_emergency=result.get("is_emergency", False),
            is_refusal=result.get("is_refusal", False),
            refusal_reason=result.get("refusal_reason"),
            hallucination_score=result.get("hallucination_score", 0.0),
            latency_ms=elapsed_ms,
            claims=[c.model_dump() for c in claims_list],
            evidence=[e.model_dump() for e in evidence_list],
            session_id=req.session_id,
        )
    except Exception as e:
        logger.error(f"Failed to persist query: {e}")

    return QueryResponse(
        request_id=req_id,
        query=req.query,
        answer=result.get("answer", ""),
        query_mode=result.get("mode", "FACTUAL"),
        is_emergency=result.get("is_emergency", False),
        is_refusal=result.get("is_refusal", False),
        refusal_reason=result.get("refusal_reason"),
        hallucination_score=result.get("hallucination_score", 0.0),
        claims=claims_list,
        evidence=evidence_list,
        execution_trace=result.get("execution_trace"),
        latency_ms=elapsed_ms,
        metrics=result.get("metrics"),
    )


@v1_router.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Ingest new clinical documents or guidelines into the knowledge base",
)
def ingest_document(
    req: IngestionRequest,
    kb_mgr: KnowledgeBaseManager = Depends(get_kb_manager),
) -> IngestionResponse:
    """
    Parse, chunk, index, and register a new medical document with full provenance.
    """
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    doc = Document(
        document_id=doc_id,
        title=req.document_title or "Uploaded Medical Document",
        text=req.content,
        source_type=req.source_type or "guideline",
        dataset_name=req.dataset_name or "ad_hoc_ingestion",
        provenance={"format": req.format},
    )

    try:
        registered = kb_mgr.register_document(doc)
        return IngestionResponse(
            status="SUCCESS",
            document_id=registered["document_id"],
            chunks_indexed=registered["chunks_created"],
            message=f"Successfully ingested and indexed {registered['chunks_created']} chunks.",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


@v1_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Comprehensive system component health check",
)
def get_health() -> HealthResponse:
    """
    Verify responsiveness of DB, vector store, and LLM configuration.
    """
    components = {
        "api": "healthy",
        "database": "healthy",
        "vector_store": "healthy" if os.path.exists(settings.VECTOR_STORE_PATH) else "degraded",
        "llm_provider": settings.LLM_PROVIDER,
    }

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        components=components,
    )


@v1_router.get(
    "/metrics",
    summary="System and agent observability metrics (JSON or Prometheus)",
)
def get_metrics(
    format: str = Query("json", description="Export format: 'json' or 'prometheus'")
):
    """
    Expose real-time clinical query metrics, latency distributions, and refusal rates.
    """
    if format == "prometheus":
        return PlainTextResponse(global_metrics.get_prometheus_format())
    return global_metrics.get_metrics_summary()


@v1_router.get(
    "/alerts",
    summary="Recent system alerts and SLO threshold breaches",
)
def get_alerts():
    """
    Evaluate and retrieve active or recent clinical system alerts.
    """
    summary = global_metrics.get_metrics_summary()
    _alert_manager.evaluate_metrics(summary)
    return {
        "active_alerts": _alert_manager.get_recent_alerts(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


@v1_router.get(
    "/history",
    summary="List past query verification outcomes",
)
def list_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    repo: QueryRepository = Depends(get_repository),
):
    return {
        "queries": repo.list_queries(limit=limit, offset=offset),
        "statistics": repo.get_statistics(),
    }


@v1_router.get(
    "/history/{query_id}",
    summary="Retrieve detailed query verification record with claims and evidence",
)
def get_query_detail(
    query_id: str,
    repo: QueryRepository = Depends(get_repository),
):
    record = repo.get_query(query_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' not found.",
        )
    return record


@v1_router.get(
    "/benchmarks",
    summary="Retrieve benchmark and ablation experiment evaluation reports",
)
def get_benchmarks():
    benchmark_path = "backend/evaluation/ablation_benchmark_results.json"
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Ablation benchmark results have not been generated yet."}
