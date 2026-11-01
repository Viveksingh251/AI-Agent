# Production Deployment Guide

## 1. Containerized Deployment with Docker Compose

To launch the full production stack:

```bash
docker compose -f docker-compose.yml up --build -d
```

Verify services:
```bash
docker compose ps
```

## 2. Environment Variables Checklist

Ensure the following variables are set securely in production:
- `SECRET_KEY`: High-entropy random key for cryptographic token signing.
- `DATABASE_URL`: Production managed PostgreSQL (e.g. AWS RDS / GCP Cloud SQL).
- `OPENAI_API_KEY`: Production OpenAI or Azure OpenAI endpoint credentials.
- `QDRANT_HOST` / `QDRANT_API_KEY`: Managed Qdrant Cloud or Kubernetes cluster service.
- `REDIS_URL`: Managed Redis (AWS ElastiCache / Redis Enterprise).

## 3. Database Migrations

Run database migrations using Alembic inside the container:
```bash
docker compose exec backend alembic upgrade head
```

## 4. Monitoring & Telemetry

- **LangSmith**: Enable `LANGCHAIN_TRACING_V2=true` for full execution trace trees.
- **Prometheus / Health**: Health probes configured at `/health` for Kubernetes liveness & readiness checks.

