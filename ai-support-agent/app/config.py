"""Application Configuration System powered by Pydantic Settings (Pydantic v2).

Supports environment-based configuration (dev, staging, prod), loading from .env,
PostgreSQL database, FAISS/pgvector vector stores, Redis caching, JWT security,
and feature flags.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(str, Enum):
    """Supported application runtime environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


# ==============================================================================
# Nested Configuration Component Models
# ==============================================================================

class LLMConfig(BaseModel):
    """Configuration for LLM and Embedding services."""

    provider: Literal["openai", "azure", "anthropic", "mock"] = "openai"
    api_key: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: float = 30.0
    base_url: Optional[str] = None
    embedding_provider: Literal["openai", "huggingface", "mock"] = "openai"
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dimension: int = 1536


class DatabaseConfig(BaseModel):
    """Configuration for PostgreSQL and async database connection pool."""

    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/support_agent_db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: float = 30.0
    echo: bool = False

    @property
    def sync_url(self) -> str:
        """Converts asyncpg driver URL to standard psycopg2 URL for Alembic/sync operations."""
        return self.url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace(
            "postgresql+asyncpg", "postgresql"
        )


class VectorStoreConfig(BaseModel):
    """Configuration for Vector Store (FAISS local and optional pgvector/Qdrant)."""

    backend: Literal["faiss", "pgvector", "qdrant", "mock"] = "faiss"
    faiss_index_dir: str = "./data/faiss_index"
    faiss_index_name: str = "support_kb"
    pgvector_table_name: str = "support_embeddings"
    similarity_metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    top_k: int = 4
    similarity_threshold: float = 0.75


class AuthConfig(BaseModel):
    """Configuration for JWT authentication and cryptographic signing."""

    secret_key: str = "temporary_dev_secret_key_change_in_production_min_32_chars"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7


class RedisConfig(BaseModel):
    """Configuration for Redis session memory and state checkpointer."""

    enabled: bool = False
    url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 86400  # 24 hours
    connection_timeout: float = 5.0


class FeatureFlags(BaseModel):
    """Application-wide feature toggles for operational control."""

    enable_escalation: bool = True
    enable_email_drafting: bool = True
    auto_ticket_creation: bool = True
    enable_sentiment_analysis: bool = True
    enable_rag_search: bool = True
    enable_streaming: bool = True
    max_agent_turns: int = 10
    confidence_threshold: float = 0.70
    human_escalation_email: str = "support-tier2@company.com"


# ==============================================================================
# Central Settings Class
# ==============================================================================

class Settings(BaseSettings):
    """Central application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # Core Application Settings
    # --------------------------------------------------------------------------
    APP_NAME: str = Field(
        default="AI Customer Support Agent",
        description="Public brand/application name",
    )
    APP_ENV: AppEnvironment = Field(
        default=AppEnvironment.DEVELOPMENT,
        description="Runtime environment: development | staging | production | testing",
    )
    DEBUG: bool = Field(default=True, description="Enable debug mode and verbose errors")
    HOST: str = Field(default="0.0.0.0", description="FastAPI server host")
    PORT: int = Field(default=8000, description="FastAPI server port")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="Allowed CORS origins",
    )
    API_V1_PREFIX: str = Field(default="/api/v1", description="Prefix for API version 1")

    # --------------------------------------------------------------------------
    # Logging Configuration
    # --------------------------------------------------------------------------
    # (LOG_LEVEL/LOG_JSON_FORMAT consumed by app.logging_config)
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging severity threshold",
    )
    LOG_JSON_FORMAT: bool = Field(
        default=True,
        description="Emit structured JSON logs for log aggregators (ELK, Datadog)",
    )

    # --------------------------------------------------------------------------
    # LLM & Embedding Settings
    # --------------------------------------------------------------------------
    LLM_PROVIDER: Literal["openai", "azure", "anthropic", "mock"] = Field(
        default="openai",
        description="Primary LLM backend provider",
    )
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key (or provider API key)",
    )
    OPENAI_MODEL_NAME: str = Field(
        default="gpt-4o-mini",
        description="Default model for reasoning and response generation",
    )
    OPENAI_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    MAX_TOKENS: int = Field(default=2048, description="Maximum tokens per completion")
    OPENAI_TIMEOUT_SECONDS: float = Field(default=30.0, description="LLM client HTTP timeout")
    OPENAI_BASE_URL: Optional[str] = Field(
        default=None,
        description="Optional base URL for Azure OpenAI or local Ollama proxy",
    )
    EMBEDDING_PROVIDER: Literal["openai", "huggingface", "mock"] = Field(
        default="openai",
        description="Provider for text embedding generation",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    EMBEDDING_DIMENSION: int = Field(default=1536, description="Embedding vector dimension")

    # LangSmith Observability
    LANGCHAIN_TRACING_V2: bool = Field(default=False, description="Enable LangSmith tracing")
    LANGCHAIN_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith endpoint",
    )
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None, description="LangSmith API key")
    LANGCHAIN_PROJECT: str = Field(
        default="ai-support-agent",
        description="LangSmith project name",
    )

    # --------------------------------------------------------------------------
    # Observability: OpenTelemetry / Sentry
    # --------------------------------------------------------------------------
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(
        default=None,
        description="OTLP trace collector endpoint (e.g. http://localhost:4317). Tracing is in-process only when unset.",
    )
    SENTRY_DSN: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking. Error tracking is disabled when unset.",
    )
    SENTRY_TRACES_SAMPLE_RATE: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sentry performance tracing sample rate",
    )
    SENTRY_RELEASE: Optional[str] = Field(
        default=None,
        description="Release identifier reported to Sentry (defaults to ai-support-agent@version).",
    )

    # --------------------------------------------------------------------------
    # PostgreSQL Relational Database Settings
    # --------------------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/support_agent_db",
        description="Async SQLAlchemy database connection string",
    )
    DB_ECHO: bool = Field(default=False, description="Echo generated SQL statements")
    DB_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=20, description="Database pool max overflow")
    DB_POOL_TIMEOUT: float = Field(default=30.0, description="Pool connection timeout in seconds")

    # --------------------------------------------------------------------------
    # Vector Store Settings (FAISS local + optional pgvector/qdrant)
    # --------------------------------------------------------------------------
    VECTORSTORE_BACKEND: Literal["faiss", "pgvector", "qdrant", "mock"] = Field(
        default="faiss",
        description="Vector storage backend engine",
    )
    VECTORSTORE_TYPE: str = Field(
        default="faiss",
        description="Backwards-compatible alias for VECTORSTORE_BACKEND",
    )
    FAISS_INDEX_DIR: str = Field(
        default="./data/faiss_index",
        description="Filesystem directory where local FAISS index binaries are saved",
    )
    FAISS_INDEX_NAME: str = Field(
        default="support_kb",
        description="Filename prefix for the saved FAISS index",
    )
    PGVECTOR_TABLE_NAME: str = Field(
        default="support_embeddings",
        description="Table name when using pgvector backend",
    )
    VECTOR_SIMILARITY_METRIC: Literal["cosine", "l2", "inner_product"] = Field(
        default="cosine",
        description="Distance metric for similarity calculation",
    )
    TOP_K_RETRIEVAL: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of knowledge base documents to retrieve",
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score required to keep a retrieved document",
    )

    # --------------------------------------------------------------------------
    # Security & JWT Authentication
    # --------------------------------------------------------------------------
    SECRET_KEY: str = Field(
        default="temporary_dev_secret_key_change_in_production_min_32_chars",
        description="Cryptographic secret key for signing JWT tokens",
    )
    ALGORITHM: str = Field(default="HS256", description="JWT cryptographic signature algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60,
        description="Access token lifespan in minutes",
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        description="Refresh token lifespan in days",
    )

    # --------------------------------------------------------------------------
    # Redis Cache & Memory (Optional)
    # --------------------------------------------------------------------------
    REDIS_ENABLED: bool = Field(
        default=False,
        description="Enable Redis for session memory and LangGraph checkpointer",
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URI",
    )
    REDIS_TTL_SECONDS: int = Field(
        default=86400,
        description="Default TTL for cached session state (seconds)",
    )
    REDIS_CONNECTION_TIMEOUT: float = Field(
        default=5.0,
        description="Redis connection timeout in seconds",
    )

    # --------------------------------------------------------------------------
    # Feature Flags
    # --------------------------------------------------------------------------
    ENABLE_ESCALATION: bool = Field(
        default=True,
        description="Feature flag: Permit routing to human support agents",
    )
    ENABLE_EMAIL_DRAFTING: bool = Field(
        default=True,
        description="Feature flag: Generate draft follow-up emails for tickets",
    )
    AUTO_TICKET_CREATION: bool = Field(
        default=True,
        description="Feature flag: Automatically create tickets for customer issues",
    )
    ENABLE_SENTIMENT_ANALYSIS: bool = Field(
        default=True,
        description="Feature flag: Perform automated customer sentiment evaluation",
    )
    ENABLE_RAG_SEARCH: bool = Field(
        default=True,
        description="Feature flag: Enable RAG search over knowledge base",
    )
    ENABLE_STREAMING: bool = Field(
        default=True,
        description="Feature flag: Allow token streaming on chat endpoints",
    )
    MAX_AGENT_TURNS: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Maximum turns before forced human escalation",
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence threshold below which escalation is triggered",
    )
    HUMAN_ESCALATION_EMAIL: str = Field(
        default="support-tier2@company.com",
        description="Target inbox for human escalation dispatch",
    )

    # --------------------------------------------------------------------------
    # Validators and Model Logic
    # --------------------------------------------------------------------------
    @field_validator("APP_ENV", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> AppEnvironment:
        """Normalizes various environment string formats (e.g., 'dev' -> 'development')."""
        if isinstance(value, AppEnvironment):
            return value
        if isinstance(value, str):
            val_clean = value.strip().lower()
            mapping = {
                "dev": AppEnvironment.DEVELOPMENT,
                "development": AppEnvironment.DEVELOPMENT,
                "stage": AppEnvironment.STAGING,
                "staging": AppEnvironment.STAGING,
                "prod": AppEnvironment.PRODUCTION,
                "production": AppEnvironment.PRODUCTION,
                "test": AppEnvironment.TESTING,
                "testing": AppEnvironment.TESTING,
            }
            if val_clean in mapping:
                return mapping[val_clean]
        return AppEnvironment.DEVELOPMENT

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> List[str]:
        """Parses comma-separated strings or JSON lists of allowed origins."""
        if isinstance(value, str):
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, list):
            return value
        return ["*"]

    @model_validator(mode="after")
    def sync_vectorstore_backend(self) -> "Settings":
        """Ensures VECTORSTORE_TYPE and VECTORSTORE_BACKEND stay in sync."""
        if self.VECTORSTORE_TYPE != self.VECTORSTORE_BACKEND:
            # Prefer VECTORSTORE_BACKEND if explicitly different
            object.__setattr__(self, "VECTORSTORE_TYPE", self.VECTORSTORE_BACKEND)

        # Enforce secure configuration in production
        if self.APP_ENV == AppEnvironment.PRODUCTION:
            if "temporary_dev_secret_key" in self.SECRET_KEY:
                raise ValueError(
                    "Production configuration error: SECRET_KEY must be changed from default value."
                )
            if self.DEBUG:
                # Force debug off in production
                object.__setattr__(self, "DEBUG", False)

        return self

    # --------------------------------------------------------------------------
    # Helper Properties
    # --------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Returns True if running in production."""
        return self.APP_ENV == AppEnvironment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Returns True if running in development."""
        return self.APP_ENV == AppEnvironment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Returns True if running in testing."""
        return self.APP_ENV == AppEnvironment.TESTING

    @property
    def is_staging(self) -> bool:
        """Returns True if running in staging."""
        return self.APP_ENV == AppEnvironment.STAGING

    @property
    def llm(self) -> LLMConfig:
        """Structured sub-view of LLM settings."""
        return LLMConfig(
            provider=self.LLM_PROVIDER,
            api_key=self.OPENAI_API_KEY,
            model_name=self.OPENAI_MODEL_NAME,
            temperature=self.OPENAI_TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            timeout_seconds=self.OPENAI_TIMEOUT_SECONDS,
            base_url=self.OPENAI_BASE_URL,
            embedding_provider=self.EMBEDDING_PROVIDER,
            embedding_model_name=self.EMBEDDING_MODEL_NAME,
            embedding_dimension=self.EMBEDDING_DIMENSION,
        )

    @property
    def database(self) -> DatabaseConfig:
        """Structured sub-view of PostgreSQL settings."""
        return DatabaseConfig(
            url=self.DATABASE_URL,
            pool_size=self.DB_POOL_SIZE,
            max_overflow=self.DB_MAX_OVERFLOW,
            pool_timeout=self.DB_POOL_TIMEOUT,
            echo=self.DB_ECHO,
        )

    @property
    def vector_store(self) -> VectorStoreConfig:
        """Structured sub-view of Vector Store settings."""
        return VectorStoreConfig(
            backend=self.VECTORSTORE_BACKEND,
            faiss_index_dir=self.FAISS_INDEX_DIR,
            faiss_index_name=self.FAISS_INDEX_NAME,
            pgvector_table_name=self.PGVECTOR_TABLE_NAME,
            similarity_metric=self.VECTOR_SIMILARITY_METRIC,
            top_k=self.TOP_K_RETRIEVAL,
            similarity_threshold=self.SIMILARITY_THRESHOLD,
        )

    @property
    def auth(self) -> AuthConfig:
        """Structured sub-view of Authentication settings."""
        return AuthConfig(
            secret_key=self.SECRET_KEY,
            algorithm=self.ALGORITHM,
            access_token_expire_minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES,
            refresh_token_expire_days=self.REFRESH_TOKEN_EXPIRE_DAYS,
        )

    @property
    def redis(self) -> RedisConfig:
        """Structured sub-view of Redis settings."""
        return RedisConfig(
            enabled=self.REDIS_ENABLED,
            url=self.REDIS_URL,
            ttl_seconds=self.REDIS_TTL_SECONDS,
            connection_timeout=self.REDIS_CONNECTION_TIMEOUT,
        )

    @property
    def features(self) -> FeatureFlags:
        """Structured sub-view of application Feature Flags."""
        return FeatureFlags(
            enable_escalation=self.ENABLE_ESCALATION,
            enable_email_drafting=self.ENABLE_EMAIL_DRAFTING,
            auto_ticket_creation=self.AUTO_TICKET_CREATION,
            enable_sentiment_analysis=self.ENABLE_SENTIMENT_ANALYSIS,
            enable_rag_search=self.ENABLE_RAG_SEARCH,
            enable_streaming=self.ENABLE_STREAMING,
            max_agent_turns=self.MAX_AGENT_TURNS,
            confidence_threshold=self.CONFIDENCE_THRESHOLD,
            human_escalation_email=self.HUMAN_ESCALATION_EMAIL,
        )


@lru_cache()
def get_settings() -> Settings:
    """Returns a singleton cached instance of the application Settings."""
    return Settings()
