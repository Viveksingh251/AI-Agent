"""Realistic Demo Seed Script for the AI Customer Support Agent.

Populates the database with a believable mid-size e-commerce dataset
("TechStore") for demos, load tests, and development:

- 8 sample customers across FREE / PRO / ENTERPRISE tiers (with passwords)
- 2 support employees + the default admin/agent accounts
- 14 orders spanning every OrderStatus (pending → delivered → refunded)
- 6 existing support tickets in different states, with message threads
- A few agent audit-log entries so the admin dashboard isn't empty

Usage::

    python -m scripts.seed_demo           # additive (skips existing emails)
    python -m scripts.seed_demo --reset   # wipe demo data first, then seed

Note: this seeds the SAME admin / agent / demo customer accounts that
``seed_admin.py`` creates (idempotent), so both scripts can run together.

All demo passwords follow the pattern ``<Role>Password123!`` and are printed
at the end of the run — change them before using any of this in production.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from decimal import Decimal
from typing import Any, Dict, List

from sqlalchemy import delete, select

from app.security import get_password_hash
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
from database.session import async_session_maker, init_db_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("seed_demo")

# ---------------------------------------------------------------------------
# Demo dataset definitions
# ---------------------------------------------------------------------------

DEMO_PASSWORD = "CustomerPassword123!"

CUSTOMERS: List[Dict[str, Any]] = [
    # -- Enterprise tier -----------------------------------------------------
    {
        "email": "sarah.connor@cyberdyne.io",
        "name": "Sarah Connor",
        "phone": "+1-555-0192",
        "company": "Cyberdyne Systems",
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "email": "priya.sharma@innovatelogic.com",
        "name": "Priya Sharma",
        "phone": "+1-555-0177",
        "company": "InnovateLogic",
        "tier": CustomerTier.ENTERPRISE,
    },
    # -- Pro tier ------------------------------------------------------------
    {
        "email": "john.miller@acme.org",
        "name": "John Miller",
        "phone": "+1-555-0144",
        "company": "Acme Corporation",
        "tier": CustomerTier.PRO,
    },
    {
        "email": "elena.rodriguez@brightpath.co",
        "name": "Elena Rodriguez",
        "phone": "+1-555-0163",
        "company": "BrightPath Consulting",
        "tier": CustomerTier.PRO,
    },
    {
        "email": "customer@example.com",  # also seeded by seed_admin.py
        "name": "Jane Customer",
        "phone": "+1-555-0100",
        "company": "Example Retailers",
        "tier": CustomerTier.PRO,
    },
    # -- Free tier -----------------------------------------------------------
    {
        "email": "alex.chen@personal.com",
        "name": "Alex Chen",
        "phone": None,
        "company": None,
        "tier": CustomerTier.FREE,
    },
    {
        "email": "marcus.webb@gmail.com",
        "name": "Marcus Webb",
        "phone": "+1-555-0121",
        "company": None,
        "tier": CustomerTier.FREE,
    },
    {
        "email": "yuki.tanaka@studio-nk.jp",
        "name": "Yuki Tanaka",
        "phone": "+81-3-5555-0134",
        "company": "Studio N-K",
        "tier": CustomerTier.FREE,
    },
]

# (customer_email, order_number, status, total, items)
ORDERS: List[Dict[str, Any]] = [
    {
        "email": "sarah.connor@cyberdyne.io",
        "order_number": "ORD-2026-8812",
        "status": OrderStatus.DELIVERED,
        "total": "499.50",
        "items": [{"sku": "SRV-RACK-01", "name": "Server Rack Enclosure", "quantity": 1, "price": 499.50}],
    },
    {
        "email": "john.miller@acme.org",
        "order_number": "ORD-2026-9915",
        "status": OrderStatus.SHIPPED,
        "total": "129.99",
        "items": [{"sku": "DEV-HUB-04", "name": "Thunderbolt 4 Dock", "quantity": 1, "price": 129.99}],
    },
    {
        "email": "alex.chen@personal.com",
        "order_number": "ORD-2026-1055",
        "status": OrderStatus.PROCESSING,
        "total": "89.00",
        "items": [
            {"sku": "AUD-CAN-02", "name": "Noise-Cancelling Headphones", "quantity": 1, "price": 89.00},
        ],
    },
    {
        "email": "alex.chen@personal.com",
        "order_number": "ORD-2026-1201",
        "status": OrderStatus.PENDING,
        "total": "34.97",
        "items": [
            {"sku": "CBL-USBC-1M", "name": "USB-C Cable 1m", "quantity": 2, "price": 12.99},
            {"sku": "ACC-MOUSE-03", "name": "Wireless Mouse", "quantity": 1, "price": 8.99},
        ],
    },
    {
        "email": "priya.sharma@innovatelogic.com",
        "order_number": "ORD-2026-4471",
        "status": OrderStatus.SHIPPED,
        "total": "2418.00",
        "items": [
            {"sku": "LAP-PRO-16", "name": 'Pro Laptop 16" M3 Max', "quantity": 2, "price": 1209.00},
        ],
    },
    {
        "email": "elena.rodriguez@brightpath.co",
        "order_number": "ORD-2026-2280",
        "status": OrderStatus.DELIVERED,
        "total": "249.99",
        "items": [{"sku": "MON-27-4K", "name": '27" 4K Monitor', "quantity": 1, "price": 249.99}],
    },
    {
        "email": "marcus.webb@gmail.com",
        "order_number": "ORD-2026-3345",
        "status": OrderStatus.CANCELLED,
        "total": "59.99",
        "items": [{"sku": "KB-MECH-87", "name": "Mechanical Keyboard 87-key", "quantity": 1, "price": 59.99}],
    },
    {
        "email": "yuki.tanaka@studio-nk.jp",
        "order_number": "ORD-2026-5102",
        "status": OrderStatus.REFUNDED,
        "total": "189.00",
        "items": [{"sku": "CAM-WEBCAM-4K", "name": "4K Streaming Webcam", "quantity": 1, "price": 189.00}],
    },
    {
        "email": "customer@example.com",
        "order_number": "ORD-2026-6604",
        "status": OrderStatus.DELIVERED,
        "total": "745.50",
        "items": [
            {"sku": "TAB-11-64", "name": "Tablet 11-inch 64GB", "quantity": 1, "price": 429.00},
            {"sku": "ACC-STYLUS-2", "name": "Active Stylus", "quantity": 1, "price": 89.50},
            {"sku": "CAS-TAB-11", "name": "Tablet Folio Case", "quantity": 1, "price": 227.00},
        ],
    },
    {
        "email": "priya.sharma@innovatelogic.com",
        "order_number": "ORD-2026-7018",
        "status": OrderStatus.PROCESSING,
        "total": "199.00",
        "items": [{"sku": "NET-SW-8P", "name": "8-Port Managed Switch", "quantity": 1, "price": 199.00}],
    },
    {
        "email": "elena.rodriguez@brightpath.co",
        "order_number": "ORD-2026-7741",
        "status": OrderStatus.PENDING,
        "total": "99.00",
        "items": [{"sku": "PWR-BANK-20K", "name": "20,000mAh Power Bank", "quantity": 2, "price": 49.50}],
    },
    {
        "email": "sarah.connor@cyberdyne.io",
        "order_number": "ORD-2026-8390",
        "status": OrderStatus.SHIPPED,
        "total": "3299.00",
        "items": [{"sku": "SRV-BLADE-X", "name": "Blade Server X-Series", "quantity": 1, "price": 3299.00}],
    },
    {
        "email": "marcus.webb@gmail.com",
        "order_number": "ORD-2026-9023",
        "status": OrderStatus.DELIVERED,
        "total": "45.00",
        "items": [{"sku": "HUB-USBC-7", "name": "7-in-1 USB-C Hub", "quantity": 1, "price": 45.00}],
    },
    # The order used by scripts/demo_conversation.py — must exist for this customer.
    {
        "email": "customer@example.com",
        "order_number": "ORD-2026-12345",
        "status": OrderStatus.SHIPPED,
        "total": "59.98",
        "items": [
            {"sku": "VPN-ROUTE-01", "name": "Travel Wi-Fi Router", "quantity": 1, "price": 59.98},
        ],
    },
    {
        "email": "customer@example.com",
        "order_number": "ORD-2026-9548",
        "status": OrderStatus.SHIPPED,
        "total": "119.99",
        "items": [{"sku": "SPK-BT-ROOM", "name": "Bluetooth Conference Speaker", "quantity": 1, "price": 119.99}],
    },
]

# (customer_email, ticket_number, subject, description, status, priority,
#  assigned_to_email_or_None, created_by_agent, messages)
TICKETS: List[Dict[str, Any]] = [
    {
        "email": "sarah.connor@cyberdyne.io",
        "ticket_number": "TICK-101",
        "subject": "Damaged shipment replacement",
        "description": "Rack enclosure arrived dented in transit during freight delivery.",
        "status": TicketStatus.IN_PROGRESS,
        "priority": TicketPriority.HIGH,
        "assigned_to": "david.lee@support.company.com",
        "created_by_agent": True,
        "messages": [
            ("customer", "Hello, my package arrived with a dented exterior. Can I get a replacement?"),
            ("agent", "I have opened ticket TICK-101 and notified Tier-2 freight logistics."),
            ("human", "Hi Sarah, I am David from logistics. A courier will collect the damaged unit tomorrow."),
        ],
    },
    {
        "email": "john.miller@acme.org",
        "ticket_number": "TICK-102",
        "subject": "Invoice billing query",
        "description": "Need updated VAT tax receipt for enterprise tax deduction.",
        "status": TicketStatus.OPEN,
        "priority": TicketPriority.LOW,
        "assigned_to": None,
        "created_by_agent": False,
        "messages": [
            ("customer", "Could you resend the VAT receipt for order ORD-2026-9915? Accounting needs it."),
        ],
    },
    {
        "email": "yuki.tanaka@studio-nk.jp",
        "ticket_number": "TICK-103",
        "subject": "Refund not received for returned webcam",
        "description": "Returned the 4K webcam under RMA-55210 twelve days ago but no refund has appeared yet.",
        "status": TicketStatus.ESCALATED,
        "priority": TicketPriority.HIGH,
        "assigned_to": "david.lee@support.company.com",
        "created_by_agent": True,
        "messages": [
            ("customer", "I returned my webcam two weeks ago and still have no refund. This is unacceptable."),
            ("agent", "I understand your frustration. I have escalated this to our payments team with priority."),
        ],
    },
    {
        "email": "elena.rodriguez@brightpath.co",
        "ticket_number": "TICK-104",
        "subject": "Monitor flickering after firmware update",
        "description": '27" 4K monitor flickers at 144Hz since installing the latest firmware.',
        "status": TicketStatus.RESOLVED,
        "priority": TicketPriority.MEDIUM,
        "assigned_to": "david.lee@support.company.com",
        "created_by_agent": False,
        "messages": [
            ("customer", "The monitor flickers on 144Hz after the firmware update."),
            ("human", "Rolling back to firmware 2.1.3 fixes it — we shipped a patch to all affected units."),
            ("customer", "Confirmed fixed after the patch. Thanks!"),
        ],
    },
    {
        "email": "alex.chen@personal.com",
        "ticket_number": "TICK-105",
        "subject": "Change shipping address on pending order",
        "description": "ORD-2026-1201 needs to ship to a new address before it leaves the warehouse.",
        "status": TicketStatus.OPEN,
        "priority": TicketPriority.MEDIUM,
        "assigned_to": None,
        "created_by_agent": True,
        "messages": [],
    },
    {
        "email": "customer@example.com",
        "ticket_number": "TICK-106",
        "subject": "Warranty claim for tablet battery",
        "description": "Tablet battery drains within 3 hours; device is 8 months old and under warranty.",
        "status": TicketStatus.IN_PROGRESS,
        "priority": TicketPriority.MEDIUM,
        "assigned_to": "david.lee@support.company.com",
        "created_by_agent": True,
        "messages": [
            ("customer", "My tablet battery drains in about 3 hours now, it's only 8 months old."),
            ("agent", "That is within the 1-year limited warranty. I have started a warranty diagnostic."),
        ],
    },
]

AUDIT_LOGS: List[Dict[str, Any]] = [
    {
        "email": "sarah.connor@cyberdyne.io",
        "action_type": "knowledge_base_retrieval",
        "details": {"query": "damaged freight policy", "score": 0.94},
        "success": True,
    },
    {
        "email": "sarah.connor@cyberdyne.io",
        "action_type": "ticket_auto_creation",
        "details": {"ticket_number": "TICK-101", "assigned_to": "David Lee"},
        "success": True,
    },
    {
        "email": "yuki.tanaka@studio-nk.jp",
        "action_type": "escalate_ticket",
        "details": {"ticket_number": "TICK-103", "reason": "Refund delay beyond SLA"},
        "success": True,
    },
    {
        "email": "customer@example.com",
        "action_type": "get_order_status",
        "details": {"order_number": "ORD-2026-6604"},
        "success": True,
    },
    {
        "email": "marcus.webb@gmail.com",
        "action_type": "get_order_status",
        "details": {"order_number": "ORD-2026-3345"},
        "success": False,
    },
]

# ---------------------------------------------------------------------------
# Seeding logic
# ---------------------------------------------------------------------------

EMPLOYEES: List[Dict[str, Any]] = [
    {"email": "david.lee@support.company.com", "name": "David Lee", "role": EmployeeRole.SUPPORT, "password": "AgentPassword123!"},
    {"email": "maria.garcia@support.company.com", "name": "Maria Garcia", "role": EmployeeRole.ADMIN, "password": "AdminPassword123!"},
    # Also covers the default admin/agent accounts from seed_admin.py:
    {"email": "admin@support.company.com", "name": "System Administrator", "role": EmployeeRole.ADMIN, "password": "AdminPassword123!"},
    {"email": "agent@support.company.com", "name": "Alex Support Specialist", "role": EmployeeRole.SUPPORT, "password": "AgentPassword123!"},
]


async def _seed_demo(reset: bool) -> None:
    logger.info("Ensuring database schema...")
    await init_db_models()

    async with async_session_maker() as session:
        if reset:
            logger.info("RESET: removing existing demo data...")
            for model in (AgentActionLog, TicketMessage, SupportTicket, Order, Customer, Employee):
                result = await session.execute(delete(model))
                logger.info("  deleted %d row(s) from %s", result.rowcount or 0, model.__tablename__)

        # -- 1. Employees ----------------------------------------------------
        logger.info("1. Seeding support employees...")
        employee_ids: Dict[str, int] = {}
        for spec in EMPLOYEES:
            existing = (
                await session.execute(select(Employee).where(Employee.email == spec["email"]))
            ).scalar_one_or_none()
            if existing:
                employee_ids[spec["email"]] = existing.id
                logger.info("  employee %s already exists", spec["email"])
                continue
            emp = Employee(
                email=spec["email"],
                name=spec["name"],
                role=spec["role"],
                hashed_password=get_password_hash(spec["password"]),
                is_active=True,
            )
            session.add(emp)
            await session.flush()
            employee_ids[spec["email"]] = emp.id

        # -- 2. Customers ----------------------------------------------------
        logger.info("2. Seeding %d customers (mixed tiers)...", len(CUSTOMERS))
        customer_ids: Dict[str, int] = {}
        for spec in CUSTOMERS:
            existing = (
                await session.execute(select(Customer).where(Customer.email == spec["email"]))
            ).scalar_one_or_none()
            if existing:
                customer_ids[spec["email"]] = existing.id
                logger.info("  customer %s already exists", spec["email"])
                continue
            cust = Customer(
                email=spec["email"],
                name=spec["name"],
                phone=spec["phone"],
                company=spec["company"],
                tier=spec["tier"],
                hashed_password=get_password_hash(DEMO_PASSWORD),
                is_active=True,
            )
            session.add(cust)
            await session.flush()
            customer_ids[spec["email"]] = cust.id

        # -- 3. Orders ---------------------------------------------------------
        logger.info("3. Seeding %d orders across all statuses...", len(ORDERS))
        orders_by_number: Dict[str, Order] = {}
        for spec in ORDERS:
            existing = (
                await session.execute(select(Order).where(Order.order_number == spec["order_number"]))
            ).scalar_one_or_none()
            if existing:
                orders_by_number[spec["order_number"]] = existing
                logger.info("  order %s already exists", spec["order_number"])
                continue
            order = Order(
                customer_id=customer_ids[spec["email"]],
                order_number=spec["order_number"],
                status=spec["status"],
                total_amount=Decimal(spec["total"]),
                items=spec["items"],
            )
            session.add(order)
            await session.flush()
            orders_by_number[spec["order_number"]] = order
        status_counts: Dict[str, int] = {}
        for spec in ORDERS:
            status_counts[spec["status"].value] = status_counts.get(spec["status"].value, 0) + 1
        logger.info("  order status distribution: %s", status_counts)

        # -- 4. Tickets + message threads --------------------------------------
        logger.info("4. Seeding %d support tickets with message threads...", len(TICKETS))
        tickets_by_number: Dict[str, SupportTicket] = {}
        for spec in TICKETS:
            existing = (
                await session.execute(select(SupportTicket).where(SupportTicket.ticket_number == spec["ticket_number"]))
            ).scalar_one_or_none()
            if existing:
                tickets_by_number[spec["ticket_number"]] = existing
                logger.info("  ticket %s already exists", spec["ticket_number"])
                continue

            assigned_email = spec.get("assigned_to")
            ticket = SupportTicket(
                ticket_number=spec["ticket_number"],
                customer_id=customer_ids[spec["email"]],
                subject=spec["subject"],
                description=spec["description"],
                status=spec["status"],
                priority=spec["priority"],
                assigned_to=employee_ids[assigned_email] if assigned_email else None,
                created_by_agent=spec.get("created_by_agent", False),
            )
            session.add(ticket)
            await session.flush()

            for sender, content in spec["messages"]:
                session.add(
                    TicketMessage(
                        ticket_id=ticket.id,
                        sender_type=MessageSenderType(sender),
                        content=content,
                    )
                )
            await session.flush()
            tickets_by_number[spec["ticket_number"]] = ticket

        # -- 5. Agent audit logs ------------------------------------------------
        logger.info("5. Seeding agent audit-log entries...")
        demo_emails = list(customer_ids.values())
        existing_logs = (
            await session.execute(select(AgentActionLog).where(AgentActionLog.customer_id.in_(demo_emails)))
        ).scalars().all()
        if existing_logs:
            logger.info("  %d audit-log row(s) already exist for demo customers, skipping", len(existing_logs))
        else:
            for spec in AUDIT_LOGS:
                session.add(
                    AgentActionLog(
                        customer_id=customer_ids[spec["email"]],
                        action_type=spec["action_type"],
                        details=spec["details"],
                        success=spec["success"],
                    )
                )
            await session.flush()

        await session.commit()

    # -- Summary -------------------------------------------------------------
    logger.info("=" * 62)
    logger.info("DEMO SEED COMPLETE")
    logger.info("=" * 62)
    logger.info("Employees : %d (%s)", len(EMPLOYEES), ", ".join(e["email"] for e in EMPLOYEES))
    logger.info("Customers : %d (%d enterprise, %d pro, %d free)",
                len(CUSTOMERS),
                sum(1 for c in CUSTOMERS if c["tier"] == CustomerTier.ENTERPRISE),
                sum(1 for c in CUSTOMERS if c["tier"] == CustomerTier.PRO),
                sum(1 for c in CUSTOMERS if c["tier"] == CustomerTier.FREE))
    logger.info("Orders    : %d across statuses: %s", len(ORDERS), sorted({o["status"].value for o in ORDERS}))
    logger.info("Tickets   : %d across statuses: %s", len(TICKETS), sorted({t["status"].value for t in TICKETS}))
    logger.info("-" * 62)
    logger.info("Customer login (all demo customers): customer@example.com / %s", DEMO_PASSWORD)
    logger.info("Support agent: agent@support.company.com / AgentPassword123!")
    logger.info("Admin        : admin@support.company.com  / AdminPassword123!")
    logger.info("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed realistic demo data for the AI Support Agent.")
    parser.add_argument("--reset", action="store_true", help="Delete existing demo data before seeding.")
    args = parser.parse_args()
    asyncio.run(_seed_demo(reset=args.reset))


if __name__ == "__main__":
    main()
