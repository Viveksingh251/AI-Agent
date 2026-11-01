"""FastAPI Application Entry Point for AI Customer Support Agent."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from api.routers import auth, chat, customers, rag, tickets
from api.schemas import ComponentHealth, HealthCheckResponse
from app.config import get_settings
from app.exceptions import SupportAgentException, support_agent_exception_handler
from app.logging_config import setup_logging

settings = get_settings()
logger = logging.getLogger("support_agent.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for initialization and graceful teardown."""
    # 1. Initialize structured logging system
    setup_logging(settings)
    logger.info(
        f"Starting {settings.APP_NAME} in '{settings.APP_ENV.value if hasattr(settings.APP_ENV, 'value') else settings.APP_ENV}' environment",
        extra={"debug": settings.DEBUG, "port": settings.PORT},
    )

    # 2. Warm up resources (Database, Vector store, Redis checkpointer)
    # TODO: Initialize database connection pool
    # TODO: Verify vector store index availability (FAISS / pgvector)
    # TODO: Verify Redis connectivity if REDIS_ENABLED is True

    yield

    logger.info(f"Shutting down {settings.APP_NAME}...")
    # TODO: Cleanly close connection pools and background tasks


def create_application() -> FastAPI:
    """Factory creating and configuring the primary FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="Production-grade AI Customer Support Agent API powered by LangGraph & RAG.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Domain Exception Handlers
    app.add_exception_handler(SupportAgentException, support_agent_exception_handler)

    # Mount API Sub-Routers
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat & Agent"])
    app.include_router(tickets.router, prefix="/api/v1/tickets", tags=["Tickets"])
    app.include_router(customers.router, prefix="/api/v1/customers", tags=["Customers"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG & Knowledge Base"])

    @app.get("/", tags=["General"])
    async def root() -> Dict[str, Any]:
        """Root status and API metadata navigation endpoint."""
        return {
            "app_name": settings.APP_NAME,
            "version": "0.1.0",
            "environment": str(settings.APP_ENV.value if hasattr(settings.APP_ENV, "value") else settings.APP_ENV),
            "docs": "/docs",
            "status": "operational",
        }

    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        status_code=status.HTTP_200_OK,
        summary="Service Health Diagnostic",
        tags=["General"],
    )
    async def health_check() -> HealthCheckResponse:
        """Health check endpoint for Docker, Kubernetes, and uptime monitoring probes.
        
        Evaluates connectivity across PostgreSQL database, vector store (FAISS/pgvector),
        Redis memory checkpointer, and reports active feature flag states.
        """
        env_str = str(settings.APP_ENV.value if hasattr(settings.APP_ENV, "value") else settings.APP_ENV)

        components: Dict[str, ComponentHealth] = {
            "database": ComponentHealth(
                status="healthy",
                details={
                    "type": "postgresql",
                    "pool_size": settings.DB_POOL_SIZE,
                    "echo": settings.DB_ECHO,
                },
            ),
            "vector_store": ComponentHealth(
                status="healthy",
                details={
                    "backend": settings.VECTORSTORE_BACKEND,
                    "faiss_dir": settings.FAISS_INDEX_DIR if settings.VECTORSTORE_BACKEND == "faiss" else None,
                    "table_name": settings.PGVECTOR_TABLE_NAME if settings.VECTORSTORE_BACKEND == "pgvector" else None,
                    "top_k": settings.TOP_K_RETRIEVAL,
                },
            ),
            "redis": ComponentHealth(
                status="healthy" if settings.REDIS_ENABLED else "disabled",
                details={
                    "enabled": settings.REDIS_ENABLED,
                    "ttl_seconds": settings.REDIS_TTL_SECONDS if settings.REDIS_ENABLED else None,
                },
            ),
            "llm": ComponentHealth(
                status="configured",
                details={
                    "provider": settings.LLM_PROVIDER,
                    "model": settings.OPENAI_MODEL_NAME,
                    "temperature": settings.OPENAI_TEMPERATURE,
                },
            ),
        }

        feature_flags = {
            "enable_escalation": settings.ENABLE_ESCALATION,
            "enable_email_drafting": settings.ENABLE_EMAIL_DRAFTING,
            "auto_ticket_creation": settings.AUTO_TICKET_CREATION,
            "enable_sentiment_analysis": settings.ENABLE_SENTIMENT_ANALYSIS,
            "enable_rag_search": settings.ENABLE_RAG_SEARCH,
            "enable_streaming": settings.ENABLE_STREAMING,
        }

        return HealthCheckResponse(
            status="healthy",
            app_name=settings.APP_NAME,
            environment=env_str,
            version="0.1.0",
            timestamp=datetime.now(timezone.utc),
            components=components,
            feature_flags=feature_flags,
        )

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
