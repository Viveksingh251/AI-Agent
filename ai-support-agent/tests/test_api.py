"""Integration Tests for FastAPI Endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(api_client: AsyncClient):
    """Verifies that GET / returns 200 OK and valid service info."""
    response = await api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "AI Customer Support Agent" in data["app_name"]


@pytest.mark.asyncio
async def test_health_endpoint(api_client: AsyncClient):
    """Verifies that GET /health returns 200 OK and detailed diagnostic health status."""
    response = await api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert "database" in data["components"]
    assert "vector_store" in data["components"]
    assert "feature_flags" in data
    assert "enable_escalation" in data["feature_flags"]
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_chat_turn_endpoint(api_client: AsyncClient):
    """Verifies that POST /api/v1/chat processes a customer message turn."""
    payload = {
        "message": "Where is my order #1234?",
        "session_id": "test_session_api_01",
    }
    response = await api_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_session_api_01"
    assert "response" in data
    assert data["turn_count"] >= 1
