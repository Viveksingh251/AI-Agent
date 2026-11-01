"""SQLAlchemy ORM Data Models and Database Initialization."""

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, relationship

from app.config import get_settings

logger = logging.getLogger("company_agent.models")

Base = declarative_base()


class SessionRecord(Base):
    """Database model storing conversation sessions."""

    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    messages = relationship(
        "MessageRecord",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="MessageRecord.id",
    )


class MessageRecord(Base):
    """Database model storing turn-by-turn messages in a session."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    role = Column(String(32), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=True)
    tool_calls_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionRecord", back_populates="messages")


class EmployeeRecord(Base):
    """Enterprise employee directory entity."""

    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(32), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False, index=True)
    department = Column(String(64), nullable=False, index=True)
    role = Column(String(128), nullable=False)
    email = Column(String(128), unique=True, nullable=False)
    status = Column(String(32), default="Active")


class TicketRecord(Base):
    """Enterprise IT and operations support ticket."""

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_code = Column(String(32), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(32), default="Open", index=True)  # Open, In Progress, Resolved, Closed
    priority = Column(String(32), default="Medium", index=True)  # Low, Medium, High, Critical
    assigned_to = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PolicyRecord(Base):
    """Official enterprise policies and handbook documents."""

    __tablename__ = "policies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_code = Column(String(32), unique=True, index=True, nullable=False)
    category = Column(String(64), nullable=False, index=True)  # HR, Security, Travel, IT
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    effective_date = Column(String(32), default="2026-01-01")


# Database Engine and Session Factory Setup
settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for obtaining an asynchronous database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db_and_seed() -> None:
    """Creates database schema if not present and seeds standard enterprise records."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as session:
        # Check if already seeded
        result = await session.execute(select(EmployeeRecord).limit(1))
        existing_emp = result.scalar_one_or_none()

        if existing_emp is None:
            logger.info("Seeding enterprise database with baseline records...")

            # 1. Employees
            seed_employees = [
                EmployeeRecord(
                    employee_id="EMP-101",
                    name="Alice Johnson",
                    department="Engineering",
                    role="Lead Cloud Architect",
                    email="alice.johnson@acme-corp.com",
                    status="Active",
                ),
                EmployeeRecord(
                    employee_id="EMP-102",
                    name="Bob Smith",
                    department="Engineering",
                    role="Senior Backend Engineer",
                    email="bob.smith@acme-corp.com",
                    status="Active",
                ),
                EmployeeRecord(
                    employee_id="EMP-103",
                    name="Carol Davis",
                    department="Human Resources",
                    role="VP of People Operations",
                    email="carol.davis@acme-corp.com",
                    status="Active",
                ),
                EmployeeRecord(
                    employee_id="EMP-104",
                    name="David Lee",
                    department="IT Support",
                    role="Lead IT Systems Specialist",
                    email="david.lee@acme-corp.com",
                    status="Active",
                ),
                EmployeeRecord(
                    employee_id="EMP-105",
                    name="Elena Rostova",
                    department="Product",
                    role="Principal Product Manager",
                    email="elena.rostova@acme-corp.com",
                    status="On Leave",
                ),
            ]
            session.add_all(seed_employees)

            # 2. Support Tickets
            seed_tickets = [
                TicketRecord(
                    ticket_code="TIK-101",
                    title="Corporate VPN authentication timeout in APAC region",
                    description="Employees connecting to VPN gateway in Tokyo report radius authentication timeouts.",
                    status="In Progress",
                    priority="High",
                    assigned_to="David Lee",
                ),
                TicketRecord(
                    ticket_code="TIK-102",
                    title="Kubernetes cluster production node auto-scaling latency",
                    description="Investigate 3-minute delay when spinning up additional node groups during traffic spikes.",
                    status="Open",
                    priority="Critical",
                    assigned_to="Alice Johnson",
                ),
                TicketRecord(
                    ticket_code="TIK-103",
                    title="New employee onboarding hardware kit for Q4 cohort",
                    description="Order 15 MacBook Pro M3 machines and YubiKeys for new engineering hires.",
                    status="Resolved",
                    priority="Medium",
                    assigned_to="David Lee",
                ),
            ]
            session.add_all(seed_tickets)

            # 3. Policies
            seed_policies = [
                PolicyRecord(
                    policy_code="POL-HR-01",
                    category="HR",
                    title="Flexible Remote Work and Home Office Stipend Policy",
                    content="Employees are eligible for hybrid or full-time remote work with manager approval. "
                    "Each full-time remote employee receives an annual $1,000 ergonomic equipment stipend. "
                    "Core collaboration hours are 10:00 AM to 3:00 PM in the employee's designated timezone.",
                ),
                PolicyRecord(
                    policy_code="POL-SEC-02",
                    category="Security",
                    title="Mandatory Multi-Factor Authentication and Password Hygiene",
                    content="All corporate accounts must enforce FIDO2 WebAuthn / hardware security keys. "
                    "Passwords must be at least 16 characters in length and generated via company password vault. "
                    "Lock workstation screens whenever unattended for more than 3 minutes.",
                ),
                PolicyRecord(
                    policy_code="POL-TRV-03",
                    category="Travel",
                    title="Corporate Travel and Expense Reimbursement Guidelines",
                    content="Domestic travel requires Director sign-off; international travel requires VP sign-off. "
                    "Per diem for meals is capped at $85/day for Tier 1 cities and $65/day elsewhere. "
                    "All receipts must be submitted via Concur within 14 calendar days of trip conclusion.",
                ),
            ]
            session.add_all(seed_policies)

            await session.commit()
            logger.info("Enterprise database seeding completed successfully.")

