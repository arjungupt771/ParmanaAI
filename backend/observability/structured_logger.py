"""
ParmanaAI - Structured JSON Logger with PII De-identification

Provides standardized JSON logs with trace correlation (request_id, trace_id, agent_name)
and automatic redaction of Protected Health Information (PHI).
"""

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional

from backend.safety.privacy_guard import PrivacyGuard


# Context variables for distributed/request tracing
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
trace_id_ctx: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)

_privacy_guard = PrivacyGuard()


class StructuredJSONFormatter(logging.Formatter):
    """
    Formats log records as structured JSON with PII sanitization.
    """

    def format(self, record: logging.LogRecord) -> str:
        # 1. Scrub PII from log message
        raw_msg = record.getMessage()
        sanitized_msg, _ = _privacy_guard.scrub_pii(raw_msg)

        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_msg,
            "module": record.module,
            "line": record.lineno,
            "request_id": request_id_ctx.get() or getattr(record, "request_id", None),
            "trace_id": trace_id_ctx.get() or getattr(record, "trace_id", None),
        }

        # Include custom attributes if present
        for attr in ("agent_name", "latency_ms", "status", "event_type", "error_code"):
            if hasattr(record, attr):
                log_payload[attr] = getattr(record, attr)

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with StructuredJSONFormatter.
    """
    if "pytest" in sys.modules:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJSONFormatter())
    root_logger.addHandler(handler)


def get_structured_logger(name: str) -> logging.Logger:
    """
    Obtain a logger instance for a given module.
    """
    return logging.getLogger(name)
