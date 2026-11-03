# ==============================================================================
# Production Dockerfile for AI Customer Support Agent
# ------------------------------------------------------------------------------
# Multi-stage build:
#   Stage 1 (builder): installs all python deps into a self-contained venv
#   Stage 2 (runtime): slim image, non-root user, healthcheck, venv copied in
#
# Build:  docker build -t ai-support-agent .
# Run:    docker run -p 8000:8000 --env-file .env ai-support-agent
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder — compile/install dependencies into an isolated venv
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build toolchain for packages that ship no prebuilt wheels (faiss, psycopg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtual environment makes the COPY --from in stage 2 clean and exact
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Runtime — slim, non-root, healthchecked
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# OCI image labels
LABEL org.opencontainers.image.title="ai-support-agent" \
      org.opencontainers.image.description="AI Customer Support Agent (FastAPI + Streamlit + FAISS RAG)" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/your-org/ai-support-agent"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    FRONTEND_PORT=8501 \
    BACKEND_API_URL=http://localhost:8000 \
    FAISS_INDEX_DIR=/app/data/faiss_index

# curl is required by the container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully built venv from the builder stage (exact, self-contained copy)
COPY --from=builder /opt/venv /opt/venv

# Non-root user with stable UID/GID
RUN groupadd -g 10001 appgroup \
    && useradd -u 10001 -g 10001 -m -d /home/appuser -s /usr/sbin/nologin appuser

WORKDIR /app

# App source code
COPY --chown=appuser:appgroup . .

# Persistent volumes for FAISS index, knowledge base documents, uploads, logs
RUN mkdir -p /app/data/faiss_index /app/data/knowledge_base /app/data/uploads /app/logs \
    && chown -R appuser:appgroup /app/data /app/logs /home/appuser

USER appuser

EXPOSE 8000 8501

# OCI-compliant entrypoint (docker/entrypoint.sh handles wait-for-db + migrations)
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default: run the FastAPI API. Override for frontend: docker run ... ai-support-agent streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Healthcheck: probes the FastAPI docs endpoint exposed on PORT
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/docs" || exit 1
