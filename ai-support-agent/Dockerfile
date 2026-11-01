# Multi-stage production Dockerfile for AI Customer Support Agent
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Runtime Stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime utilities (curl for healthcheck, libpq for PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root application user
RUN useradd -m -u 1001 appuser

# Copy application source code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser agent/ ./agent/
COPY --chown=appuser:appuser rag/ ./rag/
COPY --chown=appuser:appuser database/ ./database/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser docker/ ./docker/

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

