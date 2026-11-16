import os
import time
import uuid
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

provider = os.getenv("LLM_PROVIDER", "groq").lower()

if provider == "groq":
    assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY NOT FOUND"
else:
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.config import settings
from backend.api.routes import router as legacy_router
from backend.api.v1.router import v1_router
from backend.database.db import init_db
from backend.observability.structured_logger import (
    configure_logging,
    get_structured_logger,
    request_id_ctx,
    trace_id_ctx,
)

logger = get_structured_logger("backend.main")

app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-Agent Medical Hallucination Reduction System with Hybrid Retrieval & Verification",
    version=settings.APP_VERSION,
)


class CorrelationAndTimingMiddleware:
    """
    Pure ASGI middleware for correlation IDs and latency tracking without BaseHTTPMiddleware deadlocks.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        req_id = headers.get(b"x-request-id", b"").decode("latin1") or f"req-{uuid.uuid4().hex[:12]}"
        trace_id = headers.get(b"x-trace-id", b"").decode("latin1") or f"trace-{uuid.uuid4().hex[:12]}"

        request_id_ctx.set(req_id)
        trace_id_ctx.set(trace_id)

        start_time = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                resp_headers = list(message.get("headers", []))
                resp_headers.append((b"x-request-id", req_id.encode("latin1")))
                resp_headers.append((b"x-trace-id", trace_id.encode("latin1")))
                resp_headers.append((b"x-process-time-ms", str(duration_ms).encode("latin1")))
                message["headers"] = resp_headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


# app.add_middleware(CorrelationAndTimingMiddleware)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include both API v1 and legacy routes for backward compatibility
app.include_router(v1_router)
app.include_router(legacy_router)


@app.on_event("startup")
def startup_event():
    configure_logging()
    init_db()
    logger.info(
        f"ParmanaAI backend initialized (version: {settings.APP_VERSION}, LLM: {provider})"
    )


@app.get("/")
def root():
    return {
        "status": "ParmanaAI backend running",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "api_v1": "/api/v1",
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": settings.APP_VERSION}
