"""Seed Script to Initialize Default Admin User and Support Staff with Hashed Passwords."""

import asyncio
import logging
from app.security import get_password_hash
from database.session import async_session_maker, init_db_models
from database.models import Customer, CustomerTier, Employee, EmployeeRole
from database import crud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_admin")


async def seed_admin() -> None:
    """Creates default administrator, support agent, and test customer accounts."""
    logger.info("Verifying database schema...")
    await init_db_models()

    async with async_session_maker() as session:
        # 1. Default Admin User
        admin_email = "admin@support.company.com"
        admin_pwd = "AdminPassword123!"
        existing_admin = await crud.get_employee_by_email(session, admin_email)

        if not existing_admin:
            logger.info(f"Creating default Administrator: {admin_email}")
            admin_emp = Employee(
                email=admin_email,
                name="System Administrator",
                role=EmployeeRole.ADMIN,
                hashed_password=get_password_hash(admin_pwd),
                is_active=True,
            )
            session.add(admin_emp)
        else:
            logger.info(f"Admin '{admin_email}' already exists.")

        # 2. Default Support Agent
        support_email = "agent@support.company.com"
        support_pwd = "AgentPassword123!"
        existing_agent = await crud.get_employee_by_email(session, support_email)

        if not existing_agent:
            logger.info(f"Creating default Support Agent: {support_email}")
            agent_emp = Employee(
                email=support_email,
                name="Alex Support Specialist",
                role=EmployeeRole.SUPPORT,
                hashed_password=get_password_hash(support_pwd),
                is_active=True,
            )
            session.add(agent_emp)
        else:
            logger.info(f"Support Agent '{support_email}' already exists.")

        # 3. Default Customer
        customer_email = "customer@example.com"
        customer_pwd = "CustomerPassword123!"
        existing_cust = await crud.get_customer_by_email(session, customer_email)

        if not existing_cust:
            logger.info(f"Creating default Customer: {customer_email}")
            cust = Customer(
                email=customer_email,
                name="Jane Customer",
                phone="+1-555-0100",
                company="Example Retailers",
                tier=CustomerTier.PRO,
                hashed_password=get_password_hash(customer_pwd),
                is_active=True,
            )
            session.add(cust)
        else:
            logger.info(f"Customer '{customer_email}' already exists.")

        await session.commit()

        logger.info("\n" + "=" * 60)
        logger.info("SEED CREDENTIALS INITIALIZED FOR AUTHENTICATION & RBAC:")
        logger.info("=" * 60)
        logger.info(f"1. Admin Role        : {admin_email} / {admin_pwd}")
        logger.info(f"2. Support Agent Role: {support_email} / {support_pwd}")
        logger.info(f"3. Customer Role     : {customer_email} / {customer_pwd}")
        logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(seed_admin())

