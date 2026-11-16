"""
ParmanaAI - API v1 Request and Response Pydantic Schemas
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Clinical or biomedical inquiry.")
    external_context: Optional[str] = Field(None, description="Optional raw external medical passage.")
    session_id: Optional[str] = Field(None, description="Client session identifier.")
    mode_override: Optional[str] = Field(None, description="Force FACTUAL or INFERENTIAL mode.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the blood pressure cutoffs for diagnosing hypertension?",
                "external_context": None,
                "session_id": "sess-12345",
            }
        }
    }


class EvidenceItem(BaseModel):
    evidence_id: str
    text: str
    document_id: str
    chunk_id: str
    source_type: str = "unknown"
    score: Optional[float] = None
    match_type: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class ClaimVerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    verdict: str  # "SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"
    combined_verdict: str
    primary_evidence_id: Optional[str] = None
    evidence_assessments: Optional[List[Dict[str, Any]]] = None
    nli_verdict: Optional[str] = None
    numerical_status: Optional[str] = None
    is_inferential: Optional[bool] = False

    model_config = {"extra": "allow"}


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    query_mode: str
    is_emergency: bool = False
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    hallucination_score: float = Field(..., ge=0.0, le=1.0)
    claims: List[ClaimVerificationResult] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    execution_trace: Optional[List[Dict[str, Any]]] = None
    latency_ms: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class IngestionRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Raw text, markdown, or JSON string.")
    format: str = Field("txt", description="One of 'txt', 'md', 'json', 'csv'.")
    document_title: Optional[str] = None
    dataset_name: Optional[str] = None
    source_type: Optional[str] = "guideline"


class IngestionResponse(BaseModel):
    status: str
    document_id: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
