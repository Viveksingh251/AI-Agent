"""SQLAlchemy 2.0 Asynchronous Database Models for AI Support Agent.

Defines schemas and relationships for:
1. Customer (with password authentication)
2. Order
3. SupportTicket
4. TicketMessage
5. Employee (with password authentication & roles)
6. AgentActionLog
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ==============================================================================
# Domain Enumerations
# ==============================================================================

class CustomerTier(str, Enum):
    """Customer subscription and SLA tiers."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class OrderStatus(str, Enum):
    """E-commerce customer order lifecycle states."""

    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class TicketStatus(str, Enum):
    """Support ticket resolution states."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    """Support ticket urgency priority ratings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MessageSenderType(str, Enum):
    """Identity role of who dispatched a ticket message."""

    CUSTOMER = "customer"
    AGENT = "agent"
    HUMAN = "human"


class EmployeeRole(str, Enum):
    """Support team staff authorization roles."""

    SUPPORT = "support"
    ADMIN = "admin"


# ==============================================================================
# Base Model Definition
# ==============================================================================

class Base(DeclarativeBase):
    """Declarative Base class for all SQLAlchemy ORM models."""
    pass


# ==============================================================================
# Entity Models
# ==============================================================================

class Customer(Base):
    """Customer account profile and credentials."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    company: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    tier: Mapped[CustomerTier] = mapped_column(
        SQLEnum(CustomerTier, native_enum=False, length=32),
        default=CustomerTier.FREE,
        index=True,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    orders: Mapped[List["Order"]] = relationship(
        "Order",
        back_populates="customer",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    support_tickets: Mapped[List["SupportTicket"]] = relationship(
        "SupportTicket",
        back_populates="customer",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    action_logs: Mapped[List["AgentActionLog"]] = relationship(
        "AgentActionLog",
        back_populates="customer",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Backwards-compatibility aliases
    @property
    def tickets(self) -> List["SupportTicket"]:
        return self.support_tickets

    def __repr__(self) -> str:
        return f"<Customer id={self.id} email='{self.email}' tier='{self.tier}'>"


class Order(Base):
    """Customer purchase order and item shipment entity."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("customers.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    order_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    status: Mapped[OrderStatus] = mapped_column(
        SQLEnum(OrderStatus, native_enum=False, length=32),
        default=OrderStatus.PENDING,
        index=True,
        nullable=False,
    )
    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=2),
        nullable=False,
        default=Decimal("0.00"),
    )
    items: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )

    # Relationships
    customer: Mapped["Customer"] = relationship("Customer", back_populates="orders")

    def __repr__(self) -> str:
        return f"<Order id={self.id} number='{self.order_number}' status='{self.status}'>"


class Employee(Base):
    """Human support specialist or administrator account with credentials."""

    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    role: Mapped[EmployeeRole] = mapped_column(
        SQLEnum(EmployeeRole, native_enum=False, length=32),
        default=EmployeeRole.SUPPORT,
        index=True,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    assigned_tickets: Mapped[List["SupportTicket"]] = relationship(
        "SupportTicket",
        back_populates="assigned_employee",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Employee id={self.id} email='{self.email}' role='{self.role}'>"


class SupportTicket(Base):
    """Customer support case incident record."""

    __tablename__ = "support_tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    customer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("customers.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TicketStatus] = mapped_column(
        SQLEnum(TicketStatus, native_enum=False, length=32),
        default=TicketStatus.OPEN,
        index=True,
        nullable=False,
    )
    priority: Mapped[TicketPriority] = mapped_column(
        SQLEnum(TicketPriority, native_enum=False, length=32),
        default=TicketPriority.MEDIUM,
        index=True,
        nullable=False,
    )
    assigned_to: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("employees.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    created_by_agent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    customer: Mapped["Customer"] = relationship("Customer", back_populates="support_tickets")
    assigned_employee: Mapped[Optional["Employee"]] = relationship(
        "Employee",
        back_populates="assigned_tickets",
    )
    messages: Mapped[List["TicketMessage"]] = relationship(
        "TicketMessage",
        back_populates="ticket",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="TicketMessage.created_at",
    )

    # Compatibility aliases
    @property
    def ticket_code(self) -> str:
        return self.ticket_number

    def __repr__(self) -> str:
        return f"<SupportTicket id={self.id} number='{self.ticket_number}' status='{self.status}'>"


class TicketMessage(Base):
    """Thread conversation message belonging to a support ticket."""

    __tablename__ = "ticket_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("support_tickets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    sender_type: Mapped[MessageSenderType] = mapped_column(
        SQLEnum(MessageSenderType, native_enum=False, length=32),
        index=True,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )

    # Relationships
    ticket: Mapped["SupportTicket"] = relationship("SupportTicket", back_populates="messages")

    def __repr__(self) -> str:
        return f"<TicketMessage id={self.id} ticket_id={self.ticket_id} sender='{self.sender_type}'>"


class AgentActionLog(Base):
    """Audit log capturing actions and tool invocations made by the AI agent."""

    __tablename__ = "agent_action_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("customers.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    action_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, default=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )

    # Relationships
    customer: Mapped[Optional["Customer"]] = relationship(
        "Customer",
        back_populates="action_logs",
    )

    def __repr__(self) -> str:
        return f"<AgentActionLog id={self.id} action='{self.action_type}' success={self.success}>"


# ==============================================================================
# Composite Indexes
# ==============================================================================

Index("idx_tickets_customer_status", SupportTicket.customer_id, SupportTicket.status)
Index("idx_tickets_assigned_status", SupportTicket.assigned_to, SupportTicket.status)
Index("idx_orders_customer_status", Order.customer_id, Order.status)
Index("idx_action_logs_customer_created", AgentActionLog.customer_id, AgentActionLog.created_at)


# ==============================================================================
# Backwards-Compatibility Aliases
# ==============================================================================
Ticket = SupportTicket
Message = TicketMessage
AuditLog = AgentActionLog
