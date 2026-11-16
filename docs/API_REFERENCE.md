# ParmanaAI REST API Reference (v1)

Base URL: `http://localhost:8000/api/v1`

---

## 1. Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ask` | Submit clinical query for verification and evidence grounding |
| `POST` | `/api/v1/ingest` | Ingest and chunk new medical guideline or document |
| `GET` | `/api/v1/health` | Comprehensive component health check |
| `GET` | `/api/v1/metrics` | System and verification metrics (JSON or Prometheus) |
| `GET` | `/api/v1/alerts` | Active and recent clinical service-level alerts |
| `GET` | `/api/v1/history` | List previous queries with pagination and statistics |
| `GET` | `/api/v1/history/{query_id}` | Retrieve single query record with joined claims and evidence |
| `GET` | `/api/v1/benchmarks` | Retrieve latest multi-stage ablation benchmark results |

---

## 2. Endpoint Details

### 2.1 Submit Inquiry: `POST /api/v1/ask`

**Request Body:**
```json
{
  "query": "What is the recommended target blood pressure in patients with type 2 diabetes?",
  "external_context": null,
  "session_id": "sess-usr-102"
}
```

**Response (`200 OK`):**
```json
{
  "request_id": "req-9b88f341",
  "query": "What is the recommended target blood pressure in patients with type 2 diabetes?",
  "answer": "According to clinical guidelines, the target blood pressure for individuals with type 2 diabetes and hypertension is generally < 130/80 mmHg.",
  "query_mode": "FACTUAL",
  "is_emergency": false,
  "is_refusal": false,
  "refusal_reason": null,
  "hallucination_score": 0.0,
  "claims": [
    {
      "claim_id": "CLM-0001",
      "claim_text": "Target blood pressure for individuals with type 2 diabetes is generally < 130/80 mmHg.",
      "verdict": "SUPPORTED",
      "combined_verdict": "SUPPORTED",
      "primary_evidence_id": "ev-ada-htn-01",
      "nli_verdict": "entailment",
      "numerical_status": "NUMERICALLY_CONSISTENT",
      "is_inferential": false
    }
  ],
  "evidence": [
    {
      "evidence_id": "ev-ada-htn-01",
      "document_id": "doc-ada-2024",
      "chunk_id": "chunk-102",
      "source_type": "guideline",
      "score": 0.94,
      "text": "For individuals with diabetes and hypertension, target blood pressure is < 130/80 mmHg.",
      "provenance": {
        "org": "American Diabetes Association",
        "year": 2024
      }
    }
  ],
  "latency_ms": 182.4
}
```

---

### 2.2 Ingest Document: `POST /api/v1/ingest`

**Request Body:**
```json
{
  "content": "Clinical practice guidelines for acute asthma exacerbation management: Inhaled short-acting beta-2 agonists (SABA) 4-10 puffs every 20 minutes for the first hour.",
  "format": "txt",
  "document_title": "GINA 2024 Asthma Guideline Update",
  "source_type": "guideline"
}
```

**Response (`200 OK`):**
```json
{
  "status": "SUCCESS",
  "document_id": "doc-7a2e89bf",
  "chunks_indexed": 1,
  "message": "Successfully ingested and indexed 1 chunks."
}
```

---

### 2.3 Health Check: `GET /api/v1/health`

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "vector_store": "healthy",
    "llm_provider": "groq"
  },
  "timestamp": "2026-11-16T17:50:00Z"
}
```

---

### 2.4 Metrics Exposition: `GET /api/v1/metrics`

Query Parameters:
- `format`: `json` (default) or `prometheus`

**Response (JSON):**
```json
{
  "uptime_seconds": 3600.5,
  "total_queries": 150,
  "successful_queries": 142,
  "refused_queries": 8,
  "emergency_queries": 3,
  "refusal_rate": 0.0533,
  "mode_distribution": {
    "FACTUAL": 120,
    "INFERENTIAL": 27,
    "EMERGENCY": 3
  },
  "claim_verdict_distribution": {
    "SUPPORTED": 280,
    "CONTRADICTED": 4,
    "INCONCLUSIVE": 12
  },
  "average_hallucination_score": 0.0124,
  "stage_latencies_ms": {
    "total": {
      "min": 18.2,
      "avg": 145.6,
      "p50": 130.4,
      "p95": 280.1,
      "p99": 350.8,
      "max": 412.0
    }
  }
}
```
