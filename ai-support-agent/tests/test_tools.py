"""Unit tests for the eight LangChain agent tools.

Happy paths, validation errors, and authorization boundaries are covered
per tool. Tools run against the isolated per-test SQLite database via the
``factory``/``customer`` fixtures from conftest.
"""

from __future__ import annotations

import pytest

# The tool fixtures rely on tools.base.db_session which uses database.session's
# global engine (pointed at the shared test DB file by conftest env vars).
# The per-test db_engine fixture re-creates the schema on the SAME file, so
# tools and the fixtures see consistent data.

pytestmark = pytest.mark.asyncio


# ===========================================================================
# search_knowledge_base
# ===========================================================================


class TestSearchKnowledgeBase:
    async def test_returns_results_for_query(self):
        from agent.tools.kb_tools import search_knowledge_base

        result = await search_knowledge_base.ainvoke({"query": "how do I return a product", "top_k": 3})
        assert result["success"] is True
        assert isinstance(result["results"], list)
        assert result["count"] == len(result["results"])

    async def test_empty_query_rejected(self):
        from agent.tools.kb_tools import search_knowledge_base

        result = await search_knowledge_base.ainvoke({"query": "   "})
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_top_k_clamped(self):
        from agent.tools.kb_tools import search_knowledge_base

        result = await search_knowledge_base.ainvoke({"query": "shipping times", "top_k": 999})
        assert result["success"] is True
        assert result["count"] <= 10  # clamp upper bound


# ===========================================================================
# get_customer_info
# ===========================================================================


class TestGetCustomerInfo:
    async def test_lookup_by_email_own_account(self, customer):
        from agent.tools.customer_tools import get_customer_info

        c = customer["customer"]
        result = await get_customer_info.ainvoke({
            "email_or_id": c.email,
            "requesting_customer_id": str(c.id),
        })
        assert result["success"] is True
        assert result["customer"]["email"] == c.email
        assert result["customer"]["name"] == c.name

    async def test_lookup_by_numeric_id(self, customer):
        from agent.tools.customer_tools import get_customer_info

        c = customer["customer"]
        result = await get_customer_info.ainvoke({
            "email_or_id": str(c.id),
            "requesting_customer_id": str(c.id),
        })
        assert result["success"] is True

    async def test_other_customer_denied(self, customer, other_customer):
        from agent.tools.customer_tools import get_customer_info

        result = await get_customer_info.ainvoke({
            "email_or_id": customer["customer"].email,
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert result["success"] is False
        assert result.get("authorized") is False

    async def test_staff_can_view_any(self, customer, staff):
        from agent.tools.customer_tools import get_customer_info

        result = await get_customer_info.ainvoke({
            "email_or_id": customer["customer"].email,
            "requesting_customer_id": str(staff.id),
            "role": "support_agent",
        })
        assert result["success"] is True

    async def test_not_found(self):
        from agent.tools.customer_tools import get_customer_info

        result = await get_customer_info.ainvoke({
            "email_or_id": "ghost.nowhere@example.com",
            "requesting_customer_id": "1",
        })
        assert result["success"] is False

    async def test_invalid_identifier(self):
        from agent.tools.customer_tools import get_customer_info

        result = await get_customer_info.ainvoke({
            "email_or_id": "not-an-id-or-email",
            "requesting_customer_id": "1",
        })
        assert result["success"] is False


# ===========================================================================
# get_order_status
# ===========================================================================


class TestGetOrderStatus:
    async def test_owner_can_view(self, customer):
        from agent.tools.customer_tools import get_order_status

        c, o = customer["customer"], customer["order"]
        result = await get_order_status.ainvoke({
            "order_number": o.order_number,
            "customer_id": str(c.id),
            "requesting_customer_id": str(c.id),
        })
        assert result["success"] is True
        assert result["order"]["order_number"] == o.order_number
        assert result["order"]["total_amount"] == 99.99

    async def test_wrong_owner_id_denied(self, customer):
        from agent.tools.customer_tools import get_order_status

        o = customer["order"]
        result = await get_order_status.ainvoke({
            "order_number": o.order_number,
            "customer_id": "999999",
            "requesting_customer_id": "999999",
        })
        assert result["success"] is False
        assert result.get("authorized") is False

    async def test_unknown_order(self, customer):
        from agent.tools.customer_tools import get_order_status

        result = await get_order_status.ainvoke({
            "order_number": "ORD-DOES-NOT-EXIST",
            "customer_id": str(customer["customer"].id),
            "requesting_customer_id": str(customer["customer"].id),
        })
        assert result["success"] is False

    async def test_staff_can_view_any_order(self, customer, admin):
        from agent.tools.customer_tools import get_order_status

        o = customer["order"]
        result = await get_order_status.ainvoke({
            "order_number": o.order_number,
            "customer_id": str(customer["customer"].id),
            "requesting_customer_id": str(admin.id),
            "role": "admin",
        })
        assert result["success"] is True


# ===========================================================================
# create_support_ticket
# ===========================================================================


class TestCreateSupportTicket:
    async def test_customer_creates_own_ticket(self, customer):
        from agent.tools.ticket_tools import create_support_ticket

        c = customer["customer"]
        result = await create_support_ticket.ainvoke({
            "subject": "Mouse double-clicking",
            "description": "My mouse registers double clicks on single presses.",
            "priority": "medium",
            "requesting_customer_id": str(c.id),
        })
        assert result["success"] is True
        assert result["ticket"]["subject"] == "Mouse double-clicking"
        assert result["ticket"]["priority"] == "medium"
        assert result["ticket"]["ticket_number"].startswith("TCK-")

    async def test_invalid_priority_rejected(self, customer):
        from agent.tools.ticket_tools import create_support_ticket

        result = await create_support_ticket.ainvoke({
            "subject": "X", "description": "Y", "priority": "urgent",
            "requesting_customer_id": str(customer["customer"].id),
        })
        assert result["success"] is False

    async def test_cannot_create_for_other_customer(self, customer, other_customer):
        from agent.tools.ticket_tools import create_support_ticket

        result = await create_support_ticket.ainvoke({
            "subject": "Fraud", "description": "Trying to open ticket on someone else",
            "requesting_customer_id": str(customer["customer"].id),
            "customer_id": str(other_customer["customer"].id),
        })
        assert result["success"] is False
        assert result.get("authorized") is False

    async def test_staff_can_create_on_behalf(self, customer, staff):
        from agent.tools.ticket_tools import create_support_ticket

        result = await create_support_ticket.ainvoke({
            "subject": "Phone intake", "description": "Created during a call",
            "requesting_customer_id": str(staff.id),
            "role": "support_agent",
            "customer_id": str(customer["customer"].id),
        })
        assert result["success"] is True

    async def test_missing_subject(self, customer):
        from agent.tools.ticket_tools import create_support_ticket

        result = await create_support_ticket.ainvoke({
            "subject": "  ", "description": "desc",
            "requesting_customer_id": str(customer["customer"].id),
        })
        assert result["success"] is False


# ===========================================================================
# update_ticket_status
# ===========================================================================


class TestUpdateTicketStatus:
    async def test_owner_updates_status(self, customer):
        from agent.tools.ticket_tools import update_ticket_status

        t = customer["ticket"]
        result = await update_ticket_status.ainvoke({
            "ticket_id": str(t.id), "status": "resolved", "note": "Fixed itself",
            "requesting_customer_id": str(t.customer_id),
        })
        assert result["success"] is True
        assert result["ticket"]["status"] == "resolved"

    async def test_invalid_status_rejected(self, customer):
        from agent.tools.ticket_tools import update_ticket_status

        result = await update_ticket_status.ainvoke({
            "ticket_id": str(customer["ticket"].id), "status": "yeeted",
            "requesting_customer_id": str(customer["ticket"].customer_id),
        })
        assert result["success"] is False

    async def test_other_customer_denied(self, customer, other_customer):
        from agent.tools.ticket_tools import update_ticket_status

        result = await update_ticket_status.ainvoke({
            "ticket_id": str(customer["ticket"].id), "status": "closed",
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert result["success"] is False
        assert result.get("authorized") is False

    async def test_unknown_ticket(self, customer):
        from agent.tools.ticket_tools import update_ticket_status

        result = await update_ticket_status.ainvoke({
            "ticket_id": "999999", "status": "open",
            "requesting_customer_id": str(customer["customer"].id),
        })
        assert result["success"] is False


# ===========================================================================
# escalate_ticket
# ===========================================================================


class TestEscalateTicket:
    async def test_staff_escalates(self, customer, staff):
        from agent.tools.ticket_tools import escalate_ticket

        t = customer["ticket"]
        result = await escalate_ticket.ainvoke({
            "ticket_id": str(t.id), "reason": "SLA breach", "priority": "high",
            "requesting_customer_id": str(staff.id), "role": "support_agent",
        })
        assert result["success"] is True
        assert result["ticket"]["status"] == "escalated"
        assert result["ticket"]["priority"] == "high"

    async def test_customer_denied(self, customer):
        from agent.tools.ticket_tools import escalate_ticket

        result = await escalate_ticket.ainvoke({
            "ticket_id": str(customer["ticket"].id), "reason": "me want manager",
            "requesting_customer_id": str(customer["customer"].id),
        })
        assert result["success"] is False
        assert result.get("authorized") is False

    async def test_bad_priority_rejected(self, customer, staff):
        from agent.tools.ticket_tools import escalate_ticket

        result = await escalate_ticket.ainvoke({
            "ticket_id": str(customer["ticket"].id), "reason": "x", "priority": "low",
            "requesting_customer_id": str(staff.id), "role": "support_agent",
        })
        assert result["success"] is False

    async def test_reason_required(self, customer, staff):
        from agent.tools.ticket_tools import escalate_ticket

        result = await escalate_ticket.ainvoke({
            "ticket_id": str(customer["ticket"].id), "reason": "  ",
            "requesting_customer_id": str(staff.id), "role": "support_agent",
        })
        assert result["success"] is False


# ===========================================================================
# get_ticket_history
# ===========================================================================


class TestGetTicketHistory:
    async def test_owner_views_history_with_notes(self, customer):
        from agent.tools.ticket_tools import get_ticket_history, update_ticket_status

        t = customer["ticket"]
        await update_ticket_status.ainvoke({
            "ticket_id": str(t.id), "status": "in_progress", "note": "Investigating now",
            "requesting_customer_id": str(t.customer_id),
        })
        result = await get_ticket_history.ainvoke({
            "ticket_id": str(t.id), "requesting_customer_id": str(t.customer_id),
        })
        assert result["success"] is True
        assert result["ticket"]["description"] == t.description
        assert any("Investigating now" in m["content"] for m in result["messages"])

    async def test_other_customer_denied(self, customer, other_customer):
        from agent.tools.ticket_tools import get_ticket_history

        result = await get_ticket_history.ainvoke({
            "ticket_id": str(customer["ticket"].id),
            "requesting_customer_id": str(other_customer["customer"].id),
        })
        assert result["success"] is False
        assert result.get("authorized") is False

    async def test_by_ticket_number(self, customer):
        from agent.tools.ticket_tools import get_ticket_history

        t = customer["ticket"]
        result = await get_ticket_history.ainvoke({
            "ticket_id": t.ticket_number,
            "requesting_customer_id": str(t.customer_id),
        })
        assert result["success"] is True
        assert result["ticket"]["ticket_number"] == t.ticket_number


# ===========================================================================
# draft_followup_email
# ===========================================================================


class TestDraftFollowupEmail:
    async def test_staff_gets_template_fallback(self, customer, staff):
        from agent.tools.ticket_tools import draft_followup_email

        t = customer["ticket"]
        result = await draft_followup_email.ainvoke({
            "ticket_id": str(t.id), "tone": "professional",
            "requesting_customer_id": str(staff.id), "role": "support_agent",
        })
        assert result["success"] is True
        assert result["used_llm"] is False          # no OpenAI in test env
        assert t.ticket_number in result["email"]
        assert "Customer Support Team" in result["email"]

    async def test_customer_denied(self, customer):
        from agent.tools.ticket_tools import draft_followup_email

        result = await draft_followup_email.ainvoke({
            "ticket_id": str(customer["ticket"].id),
            "requesting_customer_id": str(customer["customer"].id),
        })
        assert result["success"] is False
        assert result.get("authorized") is False

    async def test_invalid_tone(self, customer, staff):
        from agent.tools.ticket_tools import draft_followup_email

        result = await draft_followup_email.ainvoke({
            "ticket_id": str(customer["ticket"].id), "tone": "passive-aggressive",
            "requesting_customer_id": str(staff.id), "role": "support_agent",
        })
        assert result["success"] is False
