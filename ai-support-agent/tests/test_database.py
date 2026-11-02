"""Database CRUD tests against async SQLite + model unit tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from database import crud
from database.models import (
    AgentActionLog,
    Customer,
    CustomerTier,
    Employee,
    EmployeeRole,
    MessageSenderType,
    Order,
    OrderStatus,
    SupportTicket,
    TicketMessage,
    TicketPriority,
    TicketStatus,
)

# Model instantiation tests below are sync; CRUD classes set asyncio individually.
pytestmark = []


def test_customer_model():
    """Verifies instantiation and defaults for Customer entity."""
    customer = Customer(
        name="Alice Cooper",
        email="alice@example.com",
        hashed_password="pbkdf2:test_hashed_password",
        phone="+1-555-0199",
        company="Cooper Tech LLC",
        tier=CustomerTier.ENTERPRISE,
    )
    assert customer.name == "Alice Cooper"
    assert customer.email == "alice@example.com"
    assert customer.tier == CustomerTier.ENTERPRISE
    assert customer.company == "Cooper Tech LLC"
    assert customer.hashed_password == "pbkdf2:test_hashed_password"
    assert "Alice Cooper" in repr(customer) or "alice@example.com" in repr(customer)


def test_order_model():
    """Verifies instantiation and items JSON structure for Order entity."""
    order = Order(
        customer_id=1,
        order_number="ORD-9921",
        status=OrderStatus.SHIPPED,
        total_amount=Decimal("149.99"),
        items=[
            {"sku": "SKU-KB-01", "name": "Mechanical Keyboard", "quantity": 1, "price": 149.99}
        ],
    )
    assert order.customer_id == 1
    assert order.order_number == "ORD-9921"
    assert order.status == OrderStatus.SHIPPED
    assert order.total_amount == Decimal("149.99")
    assert len(order.items) == 1
    assert order.items[0]["sku"] == "SKU-KB-01"


def test_employee_model():
    """Verifies Employee model attributes and roles."""
    employee = Employee(
        name="Sarah Jenkins",
        email="sarah.jenkins@support.company.com",
        hashed_password="pbkdf2:test_employee_hash",
        role=EmployeeRole.SUPPORT,
        is_active=True,
    )
    assert employee.name == "Sarah Jenkins"
    assert employee.role == EmployeeRole.SUPPORT
    assert employee.is_active is True
    assert employee.hashed_password == "pbkdf2:test_employee_hash"


def test_support_ticket_model():
    """Verifies SupportTicket instantiation, priorities, and statuses."""
    ticket = SupportTicket(
        ticket_number="TICK-8842",
        customer_id=1,
        subject="Defective power supply cable",
        description="The power adapter stopped charging after 2 days of use.",
        status=TicketStatus.OPEN,
        priority=TicketPriority.HIGH,
        assigned_to=2,
        created_by_agent=True,
    )
    assert ticket.ticket_number == "TICK-8842"
    assert ticket.status == TicketStatus.OPEN
    assert ticket.priority == TicketPriority.HIGH
    assert ticket.assigned_to == 2
    assert ticket.created_by_agent is True


def test_ticket_message_model():
    """Verifies TicketMessage model and sender types."""
    message = TicketMessage(
        ticket_id=10,
        sender_type=MessageSenderType.AGENT,
        content="I have verified your warranty and dispatched a replacement power supply.",
    )
    assert message.ticket_id == 10
    assert message.sender_type == MessageSenderType.AGENT
    assert "replacement" in message.content


def test_agent_action_log_model():
    """Verifies AgentActionLog audit logging model."""
    action = AgentActionLog(
        customer_id=1,
        action_type="knowledge_base_retrieval",
        details={"query": "how to return power cable", "top_k": 3, "score": 0.92},
        success=True,
    )
    assert action.customer_id == 1
    assert action.action_type == "knowledge_base_retrieval"
    assert action.details["query"] == "how to return power cable"
    assert action.success is True


# ===========================================================================
# CRUD — customers
# ===========================================================================


class TestCustomerCRUD:
    async def test_create_and_get_by_id(self, db_session, factory):
        c = await factory.customer(email="crud.a@example.com", tier=CustomerTier.PRO)
        await factory.flush()
        fetched = await crud.get_customer_by_id(db_session, c.id)
        assert fetched is not None
        assert fetched.email == "crud.a@example.com"
        assert fetched.tier == CustomerTier.PRO

    async def test_get_by_email_case_insensitive(self, db_session, factory):
        c = await factory.customer(email="Case.Sensitive@Example.COM")
        await factory.flush()
        fetched = await crud.get_customer_by_email(db_session, "case.sensitive@example.com")
        assert fetched is not None and fetched.id == c.id

    async def test_get_missing_returns_none(self, db_session):
        assert await crud.get_customer_by_id(db_session, 987654) is None
        assert await crud.get_customer_by_email(db_session, "none@x.com") is None

    async def test_email_normalised_to_lowercase(self, db_session, factory):
        c = await factory.customer(email="MiXeD@ExAmPlE.CoM")
        assert c.email == "mixed@example.com"

    async def test_password_is_hashed_not_plaintext(self, factory):
        c = await factory.customer(password="VisibleSecret123")
        assert "VisibleSecret123" not in c.hashed_password

    async def test_update_customer(self, db_session, factory):
        c = await factory.customer()
        await factory.flush()
        updated = await crud.update_customer(db_session, c.id, name="Renamed")
        await db_session.commit()
        assert updated.name == "Renamed"

    async def test_delete_customer(self, db_session, factory):
        c = await factory.customer()
        await factory.flush()
        assert await crud.delete_customer(db_session, c.id) is True
        await db_session.commit()
        assert await crud.get_customer_by_id(db_session, c.id) is None


# ===========================================================================
# CRUD — orders
# ===========================================================================


class TestOrderCRUD:
    async def test_create_and_fetch_by_number(self, db_session, factory):
        c = await factory.customer()
        o = await factory.order(c.id, order_number="ORD-CRUD-1")
        await factory.flush()
        fetched = await crud.get_order_by_number(db_session, "ord-crud-1")  # case-insensitive
        assert fetched is not None and fetched.id == o.id
        assert fetched.total_amount == Decimal("99.99")

    async def test_list_customer_orders_sorted_recent_first(self, db_session, factory):
        c = await factory.customer()
        o1 = await factory.order(c.id)
        o2 = await factory.order(c.id)
        await factory.flush()
        orders = await crud.list_customer_orders(db_session, c.id)
        assert {o.id for o in orders} == {o1.id, o2.id}

    async def test_list_orders_status_filter(self, db_session, factory):
        c = await factory.customer()
        await factory.order(c.id, status=OrderStatus.PROCESSING)
        delivered = await factory.order(c.id, status=OrderStatus.DELIVERED)
        await factory.flush()
        filtered = await crud.list_customer_orders(db_session, c.id, status=OrderStatus.DELIVERED)
        assert [o.id for o in filtered] == [delivered.id]


# ===========================================================================
# CRUD — tickets & messages
# ===========================================================================


class TestTicketCRUD:
    async def test_create_and_fetch_by_number(self, db_session, factory):
        c = await factory.customer()
        t = await factory.ticket(c.id, ticket_number="TCK-CRUD-9")
        await factory.flush()
        fetched = await crud.get_ticket_by_number(db_session, "tck-crud-9")
        assert fetched is not None
        assert fetched.subject == t.subject
        assert fetched.status == TicketStatus.OPEN

    async def test_customer_ticket_isolation(self, db_session, factory):
        c1 = await factory.customer()
        c2 = await factory.customer()
        t1 = await factory.ticket(c1.id)
        await factory.ticket(c2.id)
        await factory.flush()
        mine = await crud.list_customer_tickets(db_session, c1.id)
        assert [t.id for t in mine] == [t1.id]

    async def test_update_ticket_status_and_priority(self, db_session, factory):
        c = await factory.customer()
        t = await factory.ticket(c.id)
        await factory.flush()
        updated = await crud.update_ticket(db_session, t.id, status="resolved", priority="high")
        await db_session.commit()
        assert updated.status == TicketStatus.RESOLVED
        assert updated.priority == TicketPriority.HIGH

    async def test_update_missing_ticket_returns_none(self, db_session):
        assert await crud.update_ticket(db_session, 424242, status="open") is None

    async def test_assign_ticket_sets_in_progress(self, db_session, factory):
        c = await factory.customer()
        e = await factory.employee()
        t = await factory.ticket(c.id)
        await factory.flush()
        updated = await crud.assign_ticket_to_employee(db_session, t.id, e.id)
        await db_session.commit()
        assert updated.assigned_to == e.id
        assert updated.status == TicketStatus.IN_PROGRESS

    async def test_ticket_message_thread(self, db_session, factory):
        c = await factory.customer()
        t = await factory.ticket(c.id)
        await factory.flush()
        await crud.add_ticket_message(db_session, t.id, "customer", "It is broken")
        await crud.add_ticket_message(db_session, t.id, "agent", "We are on it")
        await db_session.commit()
        msgs = await crud.get_ticket_messages(db_session, t.id)
        assert [m.sender_type for m in msgs] == [MessageSenderType.CUSTOMER, MessageSenderType.AGENT]
        assert "broken" in msgs[0].content


# ===========================================================================
# CRUD — employees & audit logs
# ===========================================================================


class TestEmployeeAndLogs:
    async def test_create_employee_and_auth_roles(self, db_session, factory):
        admin = await factory.employee(role=EmployeeRole.ADMIN, email="boss@x.com")
        support = await factory.employee(role=EmployeeRole.SUPPORT)
        await factory.flush()
        assert admin.role == EmployeeRole.ADMIN
        assert support.role == EmployeeRole.SUPPORT
        agents = await crud.list_active_support_agents(db_session)
        assert support.id in [a.id for a in agents]
        assert admin.id not in [a.id for a in agents]

    async def test_authenticate_user_valid(self, db_session, factory):
        c = await factory.customer(password="RightPassword1")
        await factory.flush()
        result = await crud.authenticate_user(db_session, c.email, "RightPassword1")
        assert result is not None
        user_obj, user_type, role = result
        assert role == "customer"

    async def test_authenticate_user_wrong_password(self, db_session, factory):
        c = await factory.customer(password="RightPassword1")
        await factory.flush()
        assert await crud.authenticate_user(db_session, c.email, "WrongPassword1") is None

    async def test_log_and_list_agent_actions(self, db_session, factory):
        c = await factory.customer()
        await factory.flush()
        await crud.log_agent_action(db_session, "test_action", {"k": "v"}, customer_id=c.id)
        await crud.log_agent_action(db_session, "test_action", {"ok": False}, success=False)
        await db_session.commit()
        logs = await crud.list_customer_action_logs(db_session, c.id)
        assert len(logs) == 1
        assert logs[0].details == {"k": "v"}
        recent = await crud.list_recent_action_logs(db_session)
        assert len(recent) == 2


# ===========================================================================
# Auth helper shortcut
# ===========================================================================


class TestHelpers:
    async def test_backward_compat_aliases_exist(self):
        assert crud.create_ticket is crud.create_support_ticket
        assert crud.get_ticket_by_code is crud.get_ticket_by_number
