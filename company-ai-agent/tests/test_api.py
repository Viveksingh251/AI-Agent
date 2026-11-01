"""API Integration Tests for Company AI Agent FastAPI Application."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import init_db_and_seed


@pytest_asyncio.fixture(autouse=True)
async def setup_test_app():
    """Seed database before API integration test runs."""
    await init_db_and_seed()


@pytest.mark.asyncio
async def test_root_endpoint():
    """Verify root status and documentation links."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "Company AI Agent" in data["service"]


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify health check returns service and dependency statuses."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert data["active_tools"] >= 2


@pytest.mark.asyncio
async def test_list_tools_endpoint():
    """Verify endpoint lists available tools and schemas."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        names = [t["name"] for t in tools]
        assert "database_tool" in names
        assert "api_tool" in names


@pytest.mark.asyncio
async def test_chat_turn_endpoint():
    """Verify executing an agent chat turn via POST /api/v1/chat."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "message": "List all engineering employees and their emails.",
            "session_id": "test_api_session_42",
        }
        response = await client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == "test_api_session_42"
        assert "answer" in data
        assert isinstance(data["tools_used"], list)
        assert isinstance(data["steps"], list)
        assert data["iterations"] >= 1


@pytest.mark.asyncio
async def test_session_history_and_clear_endpoints():
    """Verify fetching session history and subsequently clearing it."""
    transport = ASGITransport(app=app)
    session_id = "test_api_session_clear_01"

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a message turn
        await client.post(
            "/api/v1/chat",
            json={"message": "What is the corporate travel policy?", "session_id": session_id},
        )

        # Inspect history
        history_resp = await client.get(f"/api/v1/sessions/{session_id}/history")
        assert history_resp.status_code == 200
        history_data = history_resp.json()
        assert history_data["session_id"] == session_id
        assert history_data["total_messages"] >= 2

        # Clear session
        delete_resp = await client.delete(f"/api/v1/sessions/{session_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["status"] == "cleared"

        # Verify history is now empty
        after_clear = await client.get(f"/api/v1/sessions/{session_id}/history")
        assert after_clear.status_code == 200
        assert after_clear.json()["total_messages"] == 0

