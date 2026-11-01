"""Unit Tests for Database Models, Enums, and Relationships."""

from decimal import Decimal
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
