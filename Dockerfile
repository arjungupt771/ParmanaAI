# ==============================================================================
# ParmanaAI - Multi-Stage Production Dockerfile
# ==============================================================================

# Stage 1: Frontend Build
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Backend Runtime
FROM python:3.10-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY backend/ ./backend/
COPY data/ ./data/

# Copy compiled frontend assets into backend static serving directory
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create non-root system user for security
RUN useradd -m -u 1000 parmana && \
    chown -R parmana:parmana /app
USER parmana

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
