"""Pytest Configuration, Shared Fixtures, and Test Environments."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import app
from agent.graph import create_support_graph


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provides test-specific configuration settings."""
    return Settings(
        APP_ENV="testing",
        DEBUG=True,
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        VECTORSTORE_TYPE="mock",
        REDIS_ENABLED=False,
    )


@pytest.fixture
def agent_graph():
    """Provides a fresh compiled LangGraph instance with MemorySaver for testing."""
    return create_support_graph()


@pytest_asyncio.fixture
async def api_client():
    """Provides an asynchronous HTTP test client bound to the FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

