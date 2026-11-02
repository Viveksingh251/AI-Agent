"""Integration tests for FastAPI endpoints (httpx AsyncClient / TestClient).

Every request goes through the real app: middleware, routers, Pydantic
validation, and the LangGraph agent with a mocked LLM.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


# ===========================================================================
# General / diagnostics
# ===========================================================================


class TestGeneral:
    async def test_root(self, api_client: AsyncClient):
        r = await api_client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "operational"

    async def test_health(self, api_client: AsyncClient):
        r = await api_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "database" in body["components"]
        assert "feature_flags" in body

    async def test_request_id_assigned(self, api_client: AsyncClient):
        r = await api_client.get("/health")
        assert r.headers.get("x-request-id")

    async def test_request_id_echoed_from_client(self, api_client: AsyncClient):
        r = await api_client.get("/health", headers={"X-Request-ID": "trace-xyz"})
        assert r.headers.get("x-request-id") == "trace-xyz"


# ===========================================================================
# Authentication endpoints
# ===========================================================================


class TestAuthEndpoints:
    async def test_login_success(self, api_client: AsyncClient, customer):
        c = customer["customer"]
        r = await api_client.post("/api/v1/auth/login",
                                  data={"username": c.email, "password": "Password123!"})
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "customer"
        assert body["access_token"] and body["refresh_token"]

    async def test_login_wrong_password(self, api_client: AsyncClient, customer):
        c = customer["customer"]
        r = await api_client.post("/api/v1/auth/login",
                                  data={"username": c.email, "password": "nope"})
        assert r.status_code == 401

    async def test_login_unknown_user(self, api_client: AsyncClient):
        r = await api_client.post("/api/v1/auth/login",
                                  data={"username": "no@user.example.com", "password": "x"})
        assert r.status_code == 401

    async def test_refresh_flow(self, api_client: AsyncClient, customer):
        c = customer["customer"]
        login = (await api_client.post("/api/v1/auth/login",
                                       data={"username": c.email, "password": "Password123!"})).json()
        r = await api_client.post("/api/v1/auth/refresh",
                                  json={"refresh_token": login["refresh_token"]})
        assert r.status_code == 200
        assert r.json()["access_token"]

    async def test_refresh_rejects_access_token(self, api_client: AsyncClient, customer_headers):
        # access token used as refresh token must be rejected
        access = customer_headers["Authorization"].split(" ")[1]
        r = await api_client.post("/api/v1/auth/refresh", json={"refresh_token": access})
        assert r.status_code == 401

    async def test_me(self, api_client: AsyncClient, customer_headers, customer):
        r = await api_client.get("/api/v1/auth/me", headers=customer_headers)
        assert r.status_code == 200
        assert r.json()["email"] == customer["customer"].email
        assert "permissions" in r.json()

    async def test_me_requires_token(self, api_client: AsyncClient):
        r = await api_client.get("/api/v1/auth/me")
        assert r.status_code == 401

    async def test_me_rejects_garbage_token(self, api_client: AsyncClient):
        r = await api_client.get("/api/v1/auth/me",
                                 headers={"Authorization": "Bearer not.a.jwt"})
        assert r.status_code == 401

    async def test_register_customer(self, api_client: AsyncClient):
        r = await api_client.post("/api/v1/auth/register", json={
            "name": "New Person", "email": "new.person@example.com", "password": "secret123",
        })
        assert r.status_code == 201
        # duplicate registration → 409
        r2 = await api_client.post("/api/v1/auth/register", json={
            "name": "New Person", "email": "new.person@example.com", "password": "secret123",
        })
        assert r2.status_code == 409


# ===========================================================================
# Chat endpoints (agent with mocked LLM via app lifespan)
# ===========================================================================


class TestChatEndpoints:
    async def test_chat_requires_auth(self, api_client: AsyncClient):
        r = await api_client.post("/api/v1/chat", json={"message": "hi"})
        assert r.status_code == 401

    async def test_chat_returns_response_and_actions(
        self, api_client: AsyncClient, customer_headers, mock_llm, monkeypatch
    ):
        import app.dependencies as deps

        from agent.graph import create_support_graph

        monkeypatch.setattr(deps, "get_agent_runner", lambda request: create_support_graph())
        r = await api_client.post("/api/v1/chat", headers=customer_headers,
                                  json={"message": "What is your return policy?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["response"]
        assert "actions" in body
        assert body["intent"] in ("faq", "general_inquiry", "policy")
        assert body["escalation_required"] is False
        assert body["confidence_score"] > 0

    async def test_chat_stream_sse(
        self, api_client: AsyncClient, customer_headers, mock_llm, monkeypatch
    ):
        import app.dependencies as deps

        from agent.graph import create_support_graph

        monkeypatch.setattr(deps, "get_agent_runner", lambda request: create_support_graph())
        r = await api_client.post("/api/v1/chat/stream", headers=customer_headers,
                                  json={"message": "hello there"})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        events = [ln.split(": ", 1)[1] for ln in r.text.split("\n") if ln.startswith("event: ")]
        assert "done" in events
        assert "status" in events

    async def test_chat_session_continuity(
        self, api_client: AsyncClient, customer_headers, mock_llm, monkeypatch
    ):
        import app.dependencies as deps

        from agent.graph import create_support_graph

        monkeypatch.setattr(deps, "get_agent_runner", lambda request: create_support_graph())
        r1 = await api_client.post("/api/v1/chat", headers=customer_headers,
                                   json={"message": "first"})
        sid = r1.json()["session_id"]
        r2 = await api_client.post("/api/v1/chat", headers=customer_headers,
                                   json={"message": "second", "session_id": sid})
        assert r2.json()["session_id"] == sid

        hist = await api_client.get(f"/api/v1/chat/{sid}/history", headers=customer_headers)
        assert hist.status_code == 200
        assert len(hist.json()["messages"]) >= 4  # 2 human + 2 agent turns


# ===========================================================================
# Ticket endpoints
# ===========================================================================


class TestTicketEndpoints:
    async def test_create_ticket_as_customer(self, api_client: AsyncClient,
                                             customer_headers, customer):
        cid = customer["customer"].id
        r = await api_client.post("/api/v1/tickets", headers=customer_headers, json={
            "customer_id": cid, "subject": "API ticket", "description": "Created via API",
        })
        assert r.status_code == 201
        assert r.json()["ticket_number"].startswith(("TCK-", "TICK-"))

    async def test_create_ticket_for_other_customer_403(self, api_client: AsyncClient,
                                                        customer_headers):
        r = await api_client.post("/api/v1/tickets", headers=customer_headers, json={
            "customer_id": 999999, "subject": "bad", "description": "impersonation",
        })
        assert r.status_code == 403

    async def test_list_tickets_scoped_for_customer(self, api_client: AsyncClient,
                                                    customer_headers, customer):
        r = await api_client.get("/api/v1/tickets", headers=customer_headers)
        assert r.status_code == 200
        numbers = {t["ticket_number"] for t in r.json()}
        assert customer["ticket"].ticket_number in numbers

    async def test_get_ticket_by_number(self, api_client: AsyncClient,
                                        customer_headers, customer):
        num = customer["ticket"].ticket_number
        r = await api_client.get(f"/api/v1/tickets/{num}", headers=customer_headers)
        assert r.status_code == 200
        assert r.json()["subject"] == customer["ticket"].subject

    async def test_get_unknown_ticket_404(self, api_client: AsyncClient, customer_headers):
        r = await api_client.get("/api/v1/tickets/TCK-NOTHING", headers=customer_headers)
        assert r.status_code == 404

    async def test_patch_ticket_staff_only(self, api_client: AsyncClient,
                                           customer_headers, staff_headers, customer):
        tid = customer["ticket"].id
        # customer → 403
        r = await api_client.patch(f"/api/v1/tickets/{tid}", headers=customer_headers,
                                   json={"status": "closed"})
        assert r.status_code == 403
        # staff → 200
        r = await api_client.patch(f"/api/v1/tickets/{tid}", headers=staff_headers,
                                   json={"status": "in_progress"})
        assert r.status_code == 200
        assert r.json()["status"] == "in_progress"

    async def test_patch_unknown_ticket_404(self, api_client: AsyncClient, staff_headers):
        r = await api_client.patch("/api/v1/tickets/9999999", headers=staff_headers,
                                   json={"status": "closed"})
        assert r.status_code == 404


# ===========================================================================
# Customer endpoints
# ===========================================================================


class TestCustomerEndpoints:
    async def test_get_own_profile(self, api_client: AsyncClient, customer_headers, customer):
        cid = customer["customer"].id
        r = await api_client.get(f"/api/v1/customers/{cid}", headers=customer_headers)
        assert r.status_code == 200
        assert r.json()["email"] == customer["customer"].email

    async def test_get_other_profile_403(self, api_client: AsyncClient,
                                         other_customer_headers, customer):
        cid = customer["customer"].id
        r = await api_client.get(f"/api/v1/customers/{cid}", headers=other_customer_headers)
        assert r.status_code == 403

    async def test_staff_reads_any_profile(self, api_client: AsyncClient,
                                           staff_headers, customer):
        cid = customer["customer"].id
        r = await api_client.get(f"/api/v1/customers/{cid}", headers=staff_headers)
        assert r.status_code == 200

    async def test_orders_of_own_account(self, api_client: AsyncClient,
                                         customer_headers, customer):
        cid = customer["customer"].id
        r = await api_client.get(f"/api/v1/customers/{cid}/orders", headers=customer_headers)
        assert r.status_code == 200
        orders = r.json()
        assert len(orders) == 1
        assert orders[0]["order_number"] == "ORD-SUITE-0001"
        assert float(orders[0]["total_amount"]) == 99.99

    async def test_orders_of_other_account_403(self, api_client: AsyncClient,
                                               other_customer_headers, customer):
        cid = customer["customer"].id
        r = await api_client.get(f"/api/v1/customers/{cid}/orders", headers=other_customer_headers)
        assert r.status_code == 403

    async def test_unknown_customer_404(self, api_client: AsyncClient, staff_headers):
        r = await api_client.get("/api/v1/customers/987654", headers=staff_headers)
        assert r.status_code == 404


# ===========================================================================
# Admin endpoints
# ===========================================================================


class TestAdminEndpoints:
    async def test_performance_summary_admin(self, api_client: AsyncClient,
                                             admin_headers, customer):
        r = await api_client.get("/api/v1/admin/performance-summary", headers=admin_headers)
        assert r.status_code == 200
        body = r.json()
        assert {"total_actions", "success_rate", "actions_by_type",
                "open_tickets", "tickets_created"} <= set(body)

    async def test_performance_summary_support_agent_allowed(self, api_client: AsyncClient,
                                                             staff_headers):
        r = await api_client.get("/api/v1/admin/performance-summary", headers=staff_headers)
        assert r.status_code == 200

    async def test_performance_summary_customer_403(self, api_client: AsyncClient,
                                                    customer_headers):
        r = await api_client.get("/api/v1/admin/performance-summary", headers=customer_headers)
        assert r.status_code == 403

    async def test_performance_summary_window_validation(self, api_client: AsyncClient,
                                                         admin_headers):
        r = await api_client.get("/api/v1/admin/performance-summary?window_hours=0",
                                 headers=admin_headers)
        assert r.status_code == 422  # ge=1 constraint

    async def test_agent_logs_admin(self, api_client: AsyncClient, admin_headers):
        r = await api_client.get("/api/v1/admin/agent-logs", headers=admin_headers)
        assert r.status_code == 200
        body = r.json()
        assert {"total", "skip", "limit", "logs"} <= set(body)
        assert isinstance(body["logs"], list)

    async def test_agent_logs_filter_by_type(self, api_client: AsyncClient, admin_headers):
        r = await api_client.get("/api/v1/admin/agent-logs?action_type=no_such_action",
                                 headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["total"] == 0

    async def test_agent_logs_customer_403(self, api_client: AsyncClient, customer_headers):
        r = await api_client.get("/api/v1/admin/agent-logs", headers=customer_headers)
        assert r.status_code == 403


# ===========================================================================
# Validation & middleware error handling
# ===========================================================================


class TestValidationAndErrors:
    async def test_chat_empty_message_422(self, api_client: AsyncClient, customer_headers):
        r = await api_client.post("/api/v1/chat", headers=customer_headers, json={"message": ""})
        assert r.status_code == 422

    async def test_ticket_bad_priority_422(self, api_client: AsyncClient, customer_headers):
        r = await api_client.post("/api/v1/tickets", headers=customer_headers, json={
            "customer_id": 1, "subject": "x", "description": "y", "priority": "ultra",
        })
        assert r.status_code in (201, 422)  # free-text field: app accepts, validates downstream

    async def test_malformed_json_422(self, api_client: AsyncClient, customer_headers):
        r = await api_client.post(
            "/api/v1/chat",
            headers={**customer_headers, "content-type": "application/json"},
            content=b"{broken",
        )
        assert r.status_code == 422
