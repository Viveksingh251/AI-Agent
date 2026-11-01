"""Application Configuration Management via Pydantic Settings."""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core Application Settings
    APP_NAME: str = Field(default="Company AI Agent", description="Name of the service")
    APP_ENV: str = Field(default="development", description="Runtime environment: development | production | testing")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    HOST: str = Field(default="0.0.0.0", description="Server bind host")
    PORT: int = Field(default=8000, description="Server bind port")
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # LLM Settings
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API Key or compatible provider key",
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Default LLM model name (e.g., gpt-4o, gpt-4o-mini, gpt-3.5-turbo)",
    )
    OPENAI_BASE_URL: Optional[str] = Field(
        default=None,
        description="Optional base URL for OpenAI-compatible providers or Azure/Ollama",
    )
    LLM_PROVIDER: str = Field(
        default="openai",
        description="LLM provider: openai | mock",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM responses",
    )
    MAX_ITERATIONS: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum reasoning/tool-call iterations per agent turn",
    )

    # Database & Storage
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./company_agent.db",
        description="SQLAlchemy database connection string",
    )

    # Memory Management
    MAX_HISTORY_MESSAGES: int = Field(
        default=20,
        ge=2,
        description="Maximum conversation turns to retain in active prompt context",
    )

    # External & Internal Tools
    COMPANY_API_BASE_URL: str = Field(
        default="https://api.internal.company.com",
        description="Base URL for company internal microservices",
    )
    HTTP_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        ge=1.0,
        description="HTTP client request timeout in seconds",
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns cached application settings instance."""
    return Settings()

