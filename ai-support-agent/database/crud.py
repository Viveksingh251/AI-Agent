"""Database CRUD (Create, Read, Update, Delete) Operations (SQLAlchemy 2.0).

Provides asynchronous query operations for:
- Customers (with password authentication)
- Orders
- Support Tickets & Messages
- Employees (with password authentication)
- Agent Action Logs
- Unified user credential authentication
"""

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy import delete, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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

logger = logging.getLogger("support_agent.database.crud")


# ==============================================================================
# 1. Customer CRUD
# ==============================================================================

async def create_customer(
    session: AsyncSession,
    email: str,
    name: str,
    password: Optional[str] = None,
    hashed_password: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    tier: Union[CustomerTier, str] = CustomerTier.FREE,
    **kwargs: Any,
) -> Customer:
    """Inserts a new customer record with hashed password."""
    from app.security import get_password_hash

    tier_enum = CustomerTier(tier) if isinstance(tier, str) else tier
    final_hashed_pw = hashed_password or (get_password_hash(password) if password else get_password_hash("default_password"))

    customer = Customer(
        email=email.strip().lower(),
        hashed_password=final_hashed_pw,
        name=name.strip(),
        phone=phone,
        company=company,
        tier=tier_enum,
        is_active=True,
    )
    session.add(customer)
    await session.flush()
    await session.refresh(customer)
    logger.info(f"Created customer id={customer.id} email='{customer.email}'")
    return customer


async def get_customer_by_id(
    session: AsyncSession, customer_id: int
) -> Optional[Customer]:
    """Retrieves customer by primary key ID."""
    stmt = select(Customer).where(Customer.id == customer_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_customer_by_email(
    session: AsyncSession, email: str
) -> Optional[Customer]:
    """Retrieves customer by email address."""
    clean_email = email.strip().lower()
    stmt = select(Customer).where(Customer.email == clean_email)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_customers(
    session: AsyncSession,
    tier: Optional[CustomerTier] = None,
    skip: int = 0,
    limit: int = 50,
) -> List[Customer]:
    """Returns paginated list of customers, optionally filtered by tier."""
    stmt = select(Customer).order_by(desc(Customer.created_at))
    if tier:
        stmt = stmt.where(Customer.tier == tier)
    stmt = stmt.offset(skip).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_customer(
    session: AsyncSession,
    customer_id: int,
    **updates: Any,
) -> Optional[Customer]:
    """Updates attributes on a customer record."""
    if "password" in updates:
        from app.security import get_password_hash
        updates["hashed_password"] = get_password_hash(updates.pop("password"))

    stmt = (
        update(Customer)
        .where(Customer.id == customer_id)
        .values(**updates)
        .returning(Customer)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def delete_customer(session: AsyncSession, customer_id: int) -> bool:
    """Deletes a customer and cascades to related orders and tickets."""
    stmt = delete(Customer).where(Customer.id == customer_id)
    result = await session.execute(stmt)
    return result.rowcount > 0


# ==============================================================================
# 2. Order CRUD
# ==============================================================================

async def create_order(
    session: AsyncSession,
    customer_id: int,
    order_number: str,
    total_amount: Union[Decimal, float],
    items: List[Dict[str, Any]],
    status: Union[OrderStatus, str] = OrderStatus.PENDING,
) -> Order:
    """Creates a new customer order entry."""
    status_enum = OrderStatus(status) if isinstance(status, str) else status
    decimal_amount = Decimal(str(total_amount)) if not isinstance(total_amount, Decimal) else total_amount

    order = Order(
        customer_id=customer_id,
        order_number=order_number.strip().upper(),
        status=status_enum,
        total_amount=decimal_amount,
        items=items,
    )
    session.add(order)
    await session.flush()
    await session.refresh(order)
    logger.info(f"Created order id={order.id} order_number='{order.order_number}'")
    return order


async def get_order_by_id(session: AsyncSession, order_id: int) -> Optional[Order]:
    """Retrieves order by primary key ID."""
    stmt = select(Order).where(Order.id == order_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_order_by_number(
    session: AsyncSession, order_number: str
) -> Optional[Order]:
    """Retrieves order by order number string (case-insensitive)."""
    clean_number = order_number.strip().upper()
    stmt = select(Order).where(Order.order_number == clean_number)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_customer_orders(
    session: AsyncSession,
    customer_id: int,
    status: Optional[OrderStatus] = None,
    limit: int = 10,
) -> List[Order]:
    """Returns recent orders for a customer."""
    stmt = (
        select(Order)
        .where(Order.customer_id == customer_id)
        .order_by(desc(Order.created_at))
        .limit(limit)
    )
    if status:
        stmt = stmt.where(Order.status == status)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_order_status(
    session: AsyncSession,
    order_id: int,
    status: Union[OrderStatus, str],
) -> Optional[Order]:
    """Updates order fulfillment status."""
    status_enum = OrderStatus(status) if isinstance(status, str) else status
    stmt = (
        update(Order)
        .where(Order.id == order_id)
        .values(status=status_enum)
        .returning(Order)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ==============================================================================
# 3. Support Ticket CRUD
# ==============================================================================

async def create_support_ticket(
    session: AsyncSession,
    ticket_number: str,
    customer_id: int,
    subject: str,
    description: str,
    priority: Union[TicketPriority, str] = TicketPriority.MEDIUM,
    status: Union[TicketStatus, str] = TicketStatus.OPEN,
    assigned_to: Optional[int] = None,
    created_by_agent: bool = False,
    **kwargs: Any,
) -> SupportTicket:
    """Creates a new customer support ticket incident."""
    status_enum = TicketStatus(status) if isinstance(status, str) else status
    priority_enum = TicketPriority(priority) if isinstance(priority, str) else priority

    ticket = SupportTicket(
        ticket_number=ticket_number.strip().upper(),
        customer_id=customer_id,
        subject=subject.strip(),
        description=description.strip(),
        status=status_enum,
        priority=priority_enum,
        assigned_to=assigned_to,
        created_by_agent=created_by_agent,
    )
    session.add(ticket)
    await session.flush()
    await session.refresh(ticket)
    logger.info(f"Created ticket id={ticket.id} number='{ticket.ticket_number}'")
    return ticket


async def get_ticket_by_id(
    session: AsyncSession, ticket_id: int
) -> Optional[SupportTicket]:
    """Retrieves ticket by primary key ID with messages preloaded."""
    stmt = (
        select(SupportTicket)
        .options(selectinload(SupportTicket.messages), selectinload(SupportTicket.customer))
        .where(SupportTicket.id == ticket_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_ticket_by_number(
    session: AsyncSession, ticket_number: str
) -> Optional[SupportTicket]:
    """Retrieves ticket by ticket number."""
    clean_number = ticket_number.strip().upper()
    stmt = (
        select(SupportTicket)
        .options(selectinload(SupportTicket.messages), selectinload(SupportTicket.customer))
        .where(SupportTicket.ticket_number == clean_number)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_customer_tickets(
    session: AsyncSession,
    customer_id: int,
    status: Optional[Union[TicketStatus, str]] = None,
    limit: int = 20,
) -> List[SupportTicket]:
    """Retrieves support ticket history for a customer."""
    stmt = (
        select(SupportTicket)
        .where(SupportTicket.customer_id == customer_id)
        .order_by(desc(SupportTicket.created_at))
        .limit(limit)
    )
    if status:
        status_enum = TicketStatus(status) if isinstance(status, str) else status
        stmt = stmt.where(SupportTicket.status == status_enum)

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_tickets(
    session: AsyncSession,
    status: Optional[Union[TicketStatus, str]] = None,
    priority: Optional[Union[TicketPriority, str]] = None,
    assigned_to: Optional[int] = None,
    skip: int = 0,
    limit: int = 50,
) -> List[SupportTicket]:
    """Queries tickets with optional status, priority, and assignment filters."""
    stmt = select(SupportTicket).order_by(desc(SupportTicket.created_at))

    if status:
        stmt = stmt.where(SupportTicket.status == (TicketStatus(status) if isinstance(status, str) else status))
    if priority:
        stmt = stmt.where(SupportTicket.priority == (TicketPriority(priority) if isinstance(priority, str) else priority))
    if assigned_to is not None:
        stmt = stmt.where(SupportTicket.assigned_to == assigned_to)

    stmt = stmt.offset(skip).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_ticket(
    session: AsyncSession,
    ticket_id: int,
    status: Optional[Union[TicketStatus, str]] = None,
    priority: Optional[Union[TicketPriority, str]] = None,
    assigned_to: Optional[int] = None,
    **kwargs: Any,
) -> Optional[SupportTicket]:
    """Updates status, priority, or assigned employee on a ticket."""
    update_values: Dict[str, Any] = {}
    if status is not None:
        update_values["status"] = TicketStatus(status) if isinstance(status, str) else status
    if priority is not None:
        update_values["priority"] = TicketPriority(priority) if isinstance(priority, str) else priority
    if assigned_to is not None:
        update_values["assigned_to"] = assigned_to

    update_values.update(kwargs)

    stmt = (
        update(SupportTicket)
        .where(SupportTicket.id == ticket_id)
        .values(**update_values)
        .returning(SupportTicket)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def assign_ticket_to_employee(
    session: AsyncSession,
    ticket_id: int,
    employee_id: Optional[int],
) -> Optional[SupportTicket]:
    """Assigns or unassigns a ticket to a support employee."""
    return await update_ticket(
        session=session,
        ticket_id=ticket_id,
        assigned_to=employee_id,
        status=TicketStatus.IN_PROGRESS if employee_id else TicketStatus.OPEN,
    )


# ==============================================================================
# 4. Ticket Message CRUD
# ==============================================================================

async def add_ticket_message(
    session: AsyncSession,
    ticket_id: int,
    sender_type: Union[MessageSenderType, str],
    content: str,
) -> TicketMessage:
    """Appends a customer, agent, or human message to a ticket conversation thread."""
    sender_enum = MessageSenderType(sender_type) if isinstance(sender_type, str) else sender_type

    msg = TicketMessage(
        ticket_id=ticket_id,
        sender_type=sender_enum,
        content=content.strip(),
    )
    session.add(msg)
    await session.flush()
    await session.refresh(msg)
    return msg


async def get_ticket_messages(
    session: AsyncSession,
    ticket_id: int,
    limit: int = 100,
) -> List[TicketMessage]:
    """Retrieves all messages for a ticket ordered chronologically."""
    stmt = (
        select(TicketMessage)
        .where(TicketMessage.ticket_id == ticket_id)
        .order_by(TicketMessage.created_at.asc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ==============================================================================
# 5. Employee CRUD
# ==============================================================================

async def create_employee(
    session: AsyncSession,
    email: str,
    name: str,
    password: Optional[str] = None,
    hashed_password: Optional[str] = None,
    role: Union[EmployeeRole, str] = EmployeeRole.SUPPORT,
    is_active: bool = True,
) -> Employee:
    """Creates a new human support employee or administrator with hashed password."""
    from app.security import get_password_hash

    role_enum = EmployeeRole(role) if isinstance(role, str) else role
    final_hashed_pw = hashed_password or (get_password_hash(password) if password else get_password_hash("default_employee_password"))

    employee = Employee(
        email=email.strip().lower(),
        hashed_password=final_hashed_pw,
        name=name.strip(),
        role=role_enum,
        is_active=is_active,
    )
    session.add(employee)
    await session.flush()
    await session.refresh(employee)
    logger.info(f"Created employee id={employee.id} email='{employee.email}' role='{employee.role}'")
    return employee


async def get_employee_by_id(
    session: AsyncSession, employee_id: int
) -> Optional[Employee]:
    """Retrieves employee by primary key ID."""
    stmt = select(Employee).where(Employee.id == employee_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_employee_by_email(
    session: AsyncSession, email: str
) -> Optional[Employee]:
    """Retrieves employee by email address."""
    clean_email = email.strip().lower()
    stmt = select(Employee).where(Employee.email == clean_email)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_active_support_agents(
    session: AsyncSession,
) -> List[Employee]:
    """Returns list of active support specialists available for ticket assignment."""
    stmt = (
        select(Employee)
        .where(Employee.is_active.is_(True), Employee.role == EmployeeRole.SUPPORT)
        .order_by(Employee.name.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ==============================================================================
# 6. Agent Action Log CRUD
# ==============================================================================

async def log_agent_action(
    session: AsyncSession,
    action_type: str,
    details: Dict[str, Any],
    customer_id: Optional[int] = None,
    success: bool = True,
) -> AgentActionLog:
    """Writes an AI agent action or tool invocation into the audit log."""
    log_entry = AgentActionLog(
        customer_id=customer_id,
        action_type=action_type.strip(),
        details=details,
        success=success,
    )
    session.add(log_entry)
    await session.flush()
    return log_entry


async def list_customer_action_logs(
    session: AsyncSession,
    customer_id: int,
    limit: int = 50,
) -> List[AgentActionLog]:
    """Retrieves chronological audit actions performed for a specific customer."""
    stmt = (
        select(AgentActionLog)
        .where(AgentActionLog.customer_id == customer_id)
        .order_by(desc(AgentActionLog.created_at))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_recent_action_logs(
    session: AsyncSession,
    action_type: Optional[str] = None,
    limit: int = 50,
) -> List[AgentActionLog]:
    """Retrieves recent agent action logs across all customers."""
    stmt = select(AgentActionLog).order_by(desc(AgentActionLog.created_at)).limit(limit)
    if action_type:
        stmt = stmt.where(AgentActionLog.action_type == action_type)
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ==============================================================================
# 7. Unified Authentication Query
# ==============================================================================

async def authenticate_user(
    session: AsyncSession,
    email: str,
    password: str,
) -> Optional[Tuple[Union[Customer, Employee], str, str]]:
    """Authenticates against Employee first (admin/support), then Customer.
    
    Returns:
        Tuple of (user_record, user_type: 'employee'|'customer', role: 'admin'|'support_agent'|'customer')
        or None if credentials are invalid.
    """
    from app.security import verify_password

    clean_email = email.strip().lower()

    # 1. Check Employee (support & admin staff)
    employee = await get_employee_by_email(session, clean_email)
    if employee and employee.is_active and verify_password(password, employee.hashed_password):
        role_str = "admin" if employee.role.value == "admin" else "support_agent"
        return employee, "employee", role_str

    # 2. Check Customer
    customer = await get_customer_by_email(session, clean_email)
    if customer and customer.is_active and verify_password(password, customer.hashed_password):
        return customer, "customer", "customer"

    return None


# ==============================================================================
# Backwards-Compatibility Function Aliases
# ==============================================================================
create_ticket = create_support_ticket
get_ticket_by_code = get_ticket_by_number
update_ticket_status = update_ticket
