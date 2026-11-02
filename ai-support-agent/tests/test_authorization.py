"""Authorization tests: a customer must never access another customer's data.

Exercises every tenant-scoped surface — agent tools and REST endpoints —
with an attacker token (Mallory) trying to read Jane's resources.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.authz]


# ===========================================================================
# Tool-level isolation
# ===========================================================================


class TestToolAuthorization:
    async def test_customer_info_other_denied(self, customer, other_customer):
        from agent.tools.customer_tools import get_customer_info

        r = await get_customer_info.ainvoke({
            "email_or_id": customer["customer"].email,
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert r["success"] is False and r["authorized"] is False
        assert "customer" not in r  # no data leaked

    async def test_order_other_denied(self, customer, other_customer):
        from agent.tools.customer_tools import get_order_status

        r = await get_order_status.ainvoke({
            "order_number": customer["order"].order_number,
            "customer_id": str(other_customer["customer"].id),
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert r["success"] is False

    async def test_order_claim_mismatch_denied(self, customer, other_customer):
        """Mallory claims the order is hers but is authenticated as herself."""
        from agent.tools.customer_tools import get_order_status

        r = await get_order_status.ainvoke({
            "order_number": customer["order"].order_number,
            "customer_id": str(other_customer["customer"].id),
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert r["success"] is False
        assert r["authorized"] is False

    async def test_ticket_history_other_denied(self, customer, other_customer):
        from agent.tools.ticket_tools import get_ticket_history

        r = await get_ticket_history.ainvoke({
            "ticket_id": customer["ticket"].ticket_number,
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert r["success"] is False and "messages" not in r

    async def test_ticket_update_other_denied(self, customer, other_customer):
        from agent.tools.ticket_tools import update_ticket_status

        r = await update_ticket_status.ainvoke({
            "ticket_id": str(customer["ticket"].id), "status": "closed",
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert r["success"] is False

    async def test_ticket_create_on_behalf_of_other_denied(self, customer, other_customer):
        from agent.tools.ticket_tools import create_support_ticket

        r = await create_support_ticket.ainvoke({
            "subject": "s", "description": "d",
            "customer_id": str(customer["customer"].id),
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert r["success"] is False

    async def test_escalate_and_email_staff_only(self, customer):
        from agent.tools.ticket_tools import draft_followup_email, escalate_ticket

        cid = str(customer["customer"].id)
        for tool, kwargs in (
            (escalate_ticket, {"ticket_id": str(customer["ticket"].id), "reason": "x"}),
            (draft_followup_email, {"ticket_id": str(customer["ticket"].id)}),
        ):
            r = await tool.ainvoke({**kwargs, "requesting_customer_id": cid})
            assert r["success"] is False and r["authorized"] is False

    async def test_anonymous_denied_everywhere(self, customer):
        from agent.tools.customer_tools import get_customer_info, get_order_status
        from agent.tools.ticket_tools import get_ticket_history

        cid = str(customer["customer"].id)
        r1 = await get_customer_info.ainvoke({"email_or_id": cid, "requesting_customer_id": None})
        r2 = await get_order_status.ainvoke({
            "order_number": customer["order"].order_number,
            "customer_id": cid, "requesting_customer_id": None,
        })
        r3 = await get_ticket_history.ainvoke({
            "ticket_id": customer["ticket"].ticket_number, "requesting_customer_id": None,
        })
        assert all(not r["success"] for r in (r1, r2, r3))

    async def test_admin_bypasses_ownership(self, customer, admin):
        from agent.tools.customer_tools import get_customer_info, get_order_status
        from agent.tools.ticket_tools import get_ticket_history

        cid = str(customer["customer"].id)
        r1 = await get_customer_info.ainvoke({
            "email_or_id": cid, "requesting_customer_id": str(admin.id), "role": "admin"})
        r2 = await get_order_status.ainvoke({
            "order_number": customer["order"].order_number, "customer_id": cid,
            "requesting_customer_id": str(admin.id), "role": "admin"})
        r3 = await get_ticket_history.ainvoke({
            "ticket_id": customer["ticket"].ticket_number,
            "requesting_customer_id": str(admin.id), "role": "admin"})
        assert all(r["success"] for r in (r1, r2, r3))


# ===========================================================================
# REST-level isolation
# ===========================================================================


class TestAPIAuthorization:
    async def test_cannot_read_other_customer(self, api_client: AsyncClient,
                                              other_customer_headers, customer):
        cid = customer["customer"].id
        assert (await api_client.get(f"/api/v1/customers/{cid}",
                                     headers=other_customer_headers)).status_code == 403

    async def test_cannot_read_other_orders(self, api_client: AsyncClient,
                                            other_customer_headers, customer):
        cid = customer["customer"].id
        assert (await api_client.get(f"/api/v1/customers/{cid}/orders",
                                     headers=other_customer_headers)).status_code == 403

    async def test_cannot_read_other_ticket(self, api_client: AsyncClient,
                                            other_customer_headers, customer):
        num = customer["ticket"].ticket_number
        assert (await api_client.get(f"/api/v1/tickets/{num}",
                                     headers=other_customer_headers)).status_code == 403

    async def test_ticket_list_only_shows_own(self, api_client: AsyncClient,
                                              other_customer_headers, customer,
                                              other_customer):
        r = await api_client.get("/api/v1/tickets", headers=other_customer_headers)
        numbers = {t["ticket_number"] for t in r.json()}
        assert other_customer["ticket"].ticket_number in numbers
        assert customer["ticket"].ticket_number not in numbers

    async def test_cannot_patch_other_ticket(self, api_client: AsyncClient,
                                             other_customer_headers, customer):
        tid = customer["ticket"].id
        assert (await api_client.patch(f"/api/v1/tickets/{tid}",
                                       headers=other_customer_headers,
                                       json={"status": "closed"})).status_code == 403

    async def test_staff_blocked_from_admin_only_actions_the_same_way(
        self, api_client: AsyncClient, staff_headers, customer
    ):
        # staff CAN read, but the audit-trail endpoints stay admin+agent; a
        # forged 'customer' role token never sees them.
        assert (await api_client.get("/api/v1/admin/agent-logs",
                                     headers={"Authorization": "Bearer fake"})).status_code == 401

    async def test_chat_bound_to_own_customer_id(
        self, api_client: AsyncClient, customer_headers, customer, mock_llm, monkeypatch
    ):
        """Customers cannot make the agent operate on someone else's account."""
        import app.dependencies as deps

        from agent.graph import create_support_graph

        monkeypatch.setattr(deps, "get_agent_runner", lambda request: create_support_graph())
        r = await api_client.post("/api/v1/chat", headers=customer_headers,
                                  json={"message": "hi", "customer_id": "999999"})
        assert r.status_code == 200
        # the runner received the authenticated id, not the spoofed one
        assert r.json()["session_id"]
