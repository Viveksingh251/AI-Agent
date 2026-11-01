"""Database Seed Script for Local Development and Staging Environments."""

import asyncio
import logging
from decimal import Decimal
from database.session import async_session_maker, init_db_models
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_db")


async def seed_data() -> None:
    """Populates relational database with comprehensive test records across all entities."""
    logger.info("Initializing schema tables...")
    await init_db_models()

    async with async_session_maker() as session:
        logger.info("1. Seeding customer accounts...")
        cust1 = Customer(
            email="sarah.connor@cyberdyne.io",
            name="Sarah Connor",
            phone="+1-555-0192",
            company="Cyberdyne Systems",
            tier=CustomerTier.ENTERPRISE,
        )
        cust2 = Customer(
            email="john.miller@acme.org",
            name="John Miller",
            phone="+1-555-0144",
            company="Acme Corporation",
            tier=CustomerTier.PRO,
        )
        cust3 = Customer(
            email="alex.chen@personal.com",
            name="Alex Chen",
            phone="+1-555-0188",
            company=None,
            tier=CustomerTier.FREE,
        )
        session.add_all([cust1, cust2, cust3])
        await session.flush()

        logger.info("2. Seeding support employees...")
        emp1 = Employee(
            email="david.lee@support.company.com",
            name="David Lee",
            role=EmployeeRole.SUPPORT,
            is_active=True,
        )
        emp2 = Employee(
            email="maria.garcia@support.company.com",
            name="Maria Garcia",
            role=EmployeeRole.ADMIN,
            is_active=True,
        )
        session.add_all([emp1, emp2])
        await session.flush()

        logger.info("3. Seeding customer orders...")
        ord1 = Order(
            customer_id=cust1.id,
            order_number="ORD-2026-8812",
            status=OrderStatus.DELIVERED,
            total_amount=Decimal("499.50"),
            items=[
                {"sku": "SRV-RACK-01", "name": "Server Rack Enclosure", "quantity": 1, "price": 499.50}
            ],
        )
        ord2 = Order(
            customer_id=cust2.id,
            order_number="ORD-2026-9915",
            status=OrderStatus.SHIPPED,
            total_amount=Decimal("129.99"),
            items=[
                {"sku": "DEV-HUB-04", "name": "Thunderbolt 4 Dock", "quantity": 1, "price": 129.99}
            ],
        )
        session.add_all([ord1, ord2])
        await session.flush()

        logger.info("4. Seeding support tickets...")
        ticket1 = SupportTicket(
            ticket_number="TICK-101",
            customer_id=cust1.id,
            subject="Damaged shipment replacement",
            description="Rack enclosure arrived dented in transit during freight delivery.",
            status=TicketStatus.IN_PROGRESS,
            priority=TicketPriority.HIGH,
            assigned_to=emp1.id,
            created_by_agent=True,
        )
        ticket2 = SupportTicket(
            ticket_number="TICK-102",
            customer_id=cust2.id,
            subject="Invoice billing query",
            description="Need updated VAT tax receipt for enterprise tax deduction.",
            status=TicketStatus.OPEN,
            priority=TicketPriority.LOW,
            assigned_to=None,
            created_by_agent=False,
        )
        session.add_all([ticket1, ticket2])
        await session.flush()

        logger.info("5. Seeding ticket messages...")
        msg1 = TicketMessage(
            ticket_id=ticket1.id,
            sender_type=MessageSenderType.CUSTOMER,
            content="Hello, my package arrived with a dented exterior. Can I get a replacement?",
        )
        msg2 = TicketMessage(
            ticket_id=ticket1.id,
            sender_type=MessageSenderType.AGENT,
            content="I have opened ticket TICK-101 and notified Tier-2 freight logistics.",
        )
        msg3 = TicketMessage(
            ticket_id=ticket1.id,
            sender_type=MessageSenderType.HUMAN,
            content="Hi Sarah, I am David from logistics. A courier will collect the damaged unit tomorrow.",
        )
        session.add_all([msg1, msg2, msg3])
        await session.flush()

        logger.info("6. Seeding agent action audit logs...")
        log1 = AgentActionLog(
            customer_id=cust1.id,
            action_type="knowledge_base_retrieval",
            details={"query": "damaged freight policy", "score": 0.94},
            success=True,
        )
        log2 = AgentActionLog(
            customer_id=cust1.id,
            action_type="ticket_auto_creation",
            details={"ticket_number": "TICK-101", "assigned_to": "David Lee"},
            success=True,
        )
        session.add_all([log1, log2])

        await session.commit()
        logger.info("Database seeding successfully completed for all entities!")


if __name__ == "__main__":
    asyncio.run(seed_data())
