"""
ParmanaAI - Clinical Query & Verification History Repository

Implements atomic database persistence and query history retrieval.
"""

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

from backend.database.db import get_db_connection


class QueryRepository:
    """
    Data access layer for clinical queries, claims, evidence, and benchmarks.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        return get_db_connection(self.db_path)

    def save_query_result(
        self,
        request_id: str,
        user_query: str,
        scrubbed_query: str,
        answer: str,
        query_mode: str,
        is_emergency: bool = False,
        is_refusal: bool = False,
        refusal_reason: Optional[str] = None,
        hallucination_score: float = 0.0,
        latency_ms: Optional[float] = None,
        claims: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Atomically persist a complete query execution outcome with verified claims and evidence.
        """
        query_id = f"qry-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            with conn:
                # 1. Insert query_record
                conn.execute(
                    """
                    INSERT INTO query_records (
                        id, request_id, session_id, user_query, scrubbed_query,
                        answer, query_mode, is_emergency, is_refusal, refusal_reason,
                        hallucination_score, latency_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        request_id,
                        session_id,
                        user_query,
                        scrubbed_query,
                        answer,
                        query_mode,
                        1 if is_emergency else 0,
                        1 if is_refusal else 0,
                        refusal_reason,
                        hallucination_score,
                        latency_ms,
                        now_iso,
                    ),
                )

                # 2. Insert claims
                if claims:
                    claim_rows = []
                    for c in claims:
                        cid = f"rec-clm-{uuid.uuid4().hex[:10]}"
                        claim_rows.append((
                            cid,
                            query_id,
                            c.get("claim_id", cid),
                            c.get("claim_text", ""),
                            c.get("verdict", "UNKNOWN"),
                            c.get("combined_verdict", c.get("verdict", "UNKNOWN")),
                            c.get("primary_evidence_id"),
                            c.get("nli_verdict"),
                            c.get("numerical_status"),
                            1 if c.get("is_inferential") else 0,
                            now_iso,
                        ))
                    conn.executemany(
                        """
                        INSERT INTO claim_records (
                            id, query_id, claim_id, claim_text, verdict, combined_verdict,
                            primary_evidence_id, nli_verdict, numerical_status, is_inferential, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        claim_rows,
                    )

                # 3. Insert evidence
                if evidence:
                    ev_rows = []
                    for ev in evidence:
                        eid = f"rec-ev-{uuid.uuid4().hex[:10]}"
                        prov_json = json.dumps(ev.get("provenance", {}))
                        ev_rows.append((
                            eid,
                            query_id,
                            ev.get("evidence_id", eid),
                            ev.get("document_id", "doc-unknown"),
                            ev.get("chunk_id", "chunk-unknown"),
                            ev.get("source_type", "unknown"),
                            ev.get("score"),
                            ev.get("text", ""),
                            prov_json,
                            now_iso,
                        ))
                    conn.executemany(
                        """
                        INSERT INTO evidence_records (
                            id, query_id, evidence_id, document_id, chunk_id, source_type,
                            score, text, provenance_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        ev_rows,
                    )

            return query_id
        finally:
            conn.close()

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a query record with joined claims and evidence.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM query_records WHERE id = ? OR request_id = ?",
                (query_id, query_id),
            )
            q_row = cursor.fetchone()
            if not q_row:
                return None

            q_dict = dict(q_row)
            actual_id = q_dict["id"]

            # Claims
            c_cursor = conn.execute(
                "SELECT * FROM claim_records WHERE query_id = ? ORDER BY id",
                (actual_id,),
            )
            q_dict["claims"] = [dict(r) for r in c_cursor.fetchall()]

            # Evidence
            e_cursor = conn.execute(
                "SELECT * FROM evidence_records WHERE query_id = ? ORDER BY id",
                (actual_id,),
            )
            ev_list = []
            for r in e_cursor.fetchall():
                d = dict(r)
                if d.get("provenance_json"):
                    try:
                        d["provenance"] = json.loads(d["provenance_json"])
                    except Exception:
                        d["provenance"] = {}
                ev_list.append(d)
            q_dict["evidence"] = ev_list

            return q_dict
        finally:
            conn.close()

    def list_queries(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List past queries in reverse chronological order.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM query_records ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Aggregate top-level usage and verification statistics.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    SUM(is_emergency) as emergency_count,
                    SUM(is_refusal) as refusal_count,
                    AVG(hallucination_score) as avg_hallucination_score,
                    AVG(latency_ms) as avg_latency_ms
                FROM query_records
            """)
            row = cursor.fetchone()
            return {
                "total_queries": row["total_queries"] or 0,
                "emergency_count": row["emergency_count"] or 0,
                "refusal_count": row["refusal_count"] or 0,
                "avg_hallucination_score": round(row["avg_hallucination_score"] or 0.0, 4),
                "avg_latency_ms": round(row["avg_latency_ms"] or 0.0, 2),
            }
        finally:
            conn.close()
