"""Asynchronous Database Session Management and Connection Pooling (SQLAlchemy 2.0)."""

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import get_settings

logger = logging.getLogger("support_agent.database.session")
settings = get_settings()


def get_engine_kwargs(db_url: str) -> dict:
    """Builds appropriate engine keyword parameters based on target database dialect."""
    kwargs: dict = {
        "echo": settings.DB_ECHO,
        "future": True,
    }

    # SQLite (used in automated tests or local ephemeral testing) does not support pool sizing
    if "sqlite" in db_url.lower():
        kwargs["poolclass"] = NullPool
    else:
        # PostgreSQL production connection pooling configuration
        kwargs.update({
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": settings.DB_POOL_TIMEOUT,
            "pool_pre_ping": True,
        })

    return kwargs


# Global Async Engine Singleton
async_engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    **get_engine_kwargs(settings.DATABASE_URL),
)

# Async Session Factory
async_session_maker = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session wrapped in an automatic transaction."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            logger.exception("Database transaction failed, rolling back active changes.")
            await session.rollback()
            raise exc
        finally:
            await session.close()


async def init_db_models() -> None:
    """Initializes database schema tables in development or test environments.
    
    In production environments, schema migrations should be managed via Alembic.
    """
    from database.models import Base

    logger.info("Verifying database schema models...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")


async def close_db_connections() -> None:
    """Closes all pooled database connections cleanly during application shutdown."""
    logger.info("Disposing async database connection engine pool...")
    await async_engine.dispose()
