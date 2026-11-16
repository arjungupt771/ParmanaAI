"""
ParmanaAI - SQLite Database Engine & Schema Initializer
"""

import os
import sqlite3
from typing import Optional
from backend.config import settings


def get_db_path() -> str:
    """Parse sqlite filepath from settings.DATABASE_URL."""
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        path = db_url.replace("sqlite:///", "")
        return path
    elif db_url.startswith("sqlite://"):
        return db_url.replace("sqlite://", "")
    return "parmana_ai.db"


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Establish a thread-safe SQLite connection with WAL journal mode and foreign keys.
    """
    path = db_path or get_db_path()
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize all database tables and indexes.
    """
    conn = get_db_connection(db_path)
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS query_records (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                session_id TEXT,
                user_query TEXT NOT NULL,
                scrubbed_query TEXT NOT NULL,
                answer TEXT NOT NULL,
                query_mode TEXT NOT NULL,
                is_emergency INTEGER DEFAULT 0,
                is_refusal INTEGER DEFAULT 0,
                refusal_reason TEXT,
                hallucination_score REAL NOT NULL,
                latency_ms REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claim_records (
                id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                verdict TEXT NOT NULL,
                combined_verdict TEXT NOT NULL,
                primary_evidence_id TEXT,
                nli_verdict TEXT,
                numerical_status TEXT,
                is_inferential INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (query_id) REFERENCES query_records(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS evidence_records (
                id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                score REAL,
                text TEXT NOT NULL,
                provenance_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (query_id) REFERENCES query_records(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id TEXT PRIMARY KEY,
                benchmark_name TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_query_request_id ON query_records(request_id);
            CREATE INDEX IF NOT EXISTS idx_query_created_at ON query_records(created_at);
            CREATE INDEX IF NOT EXISTS idx_claim_query_id ON claim_records(query_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_query_id ON evidence_records(query_id);
        """)
    conn.close()
