"""Unit Tests for Application Configuration System and Structured Logging."""

import json
import logging
from app.config import AppEnvironment, Settings
from app.logging_config import JSONLogFormatter


def test_settings_defaults():
    """Verifies that default settings load with expected values."""
    settings = Settings()
    assert settings.APP_NAME == "AI Customer Support Agent"
    assert settings.APP_ENV == AppEnvironment.DEVELOPMENT
    assert settings.is_development is True
    assert settings.is_production is False
    assert settings.ENABLE_ESCALATION is True
    assert settings.ENABLE_EMAIL_DRAFTING is True
    assert settings.VECTORSTORE_BACKEND in ["faiss", "pgvector", "qdrant", "mock"]


def test_environment_normalization():
    """Verifies that shorthand strings ('dev', 'prod', 'staging') normalize properly."""
    dev_settings = Settings(APP_ENV="dev")
    assert dev_settings.APP_ENV == AppEnvironment.DEVELOPMENT

    prod_settings = Settings(
        APP_ENV="prod",
        SECRET_KEY="a-very-secure-random-production-secret-key-32chars",
        DEBUG=False,
    )
    assert prod_settings.APP_ENV == AppEnvironment.PRODUCTION
    assert prod_settings.is_production is True


def test_settings_sub_components():
    """Verifies structured sub-component properties."""
    settings = Settings(
        OPENAI_MODEL_NAME="gpt-4o",
        VECTORSTORE_BACKEND="faiss",
        REDIS_ENABLED=True,
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/support_agent_db",
    )
    # LLM config
    assert settings.llm.model_name == "gpt-4o"
    assert settings.llm.provider == "openai"

    # Vector store config
    assert settings.vector_store.backend == "faiss"
    assert settings.vector_store.faiss_index_dir == "./data/faiss_index"

    # Database config
    assert "postgresql" in settings.database.url
    assert "postgresql" in settings.database.sync_url

    # Redis config
    assert settings.redis.enabled is True

    # Feature flags
    assert settings.features.enable_escalation is True
    assert settings.features.enable_email_drafting is True


def test_json_log_formatter():
    """Verifies that JSONLogFormatter emits valid, parseable JSON with required metadata."""
    formatter = JSONLogFormatter(environment="test-env", service_name="test-service")
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test customer message: %s",
        args=("order refund",),
        exc_info=None,
    )
    # Add custom extra attribute
    record.customer_id = "CUST-9921"

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert "Test customer message: order refund" in parsed["message"]
    assert parsed["environment"] == "test-env"
    assert parsed["service"] == "test-service"
    assert parsed["location"]["line"] == 42
    assert parsed["extra"]["customer_id"] == "CUST-9921"
    assert "timestamp" in parsed

