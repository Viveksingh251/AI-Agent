"""FastAPI Request Dependencies and Resource Injections."""

from typing import AsyncGenerator
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.security import (
    AuthenticatedUser,
    UserRole,
    check_tool_access,
    get_current_active_user,
    get_current_user,
    oauth2_scheme,
    require_permission,
    require_roles,
    verify_tool_access,
)
from database.session import get_db_session


async def get_app_settings() -> Settings:
    """Dependency that returns application settings."""
    return get_settings()


async def get_database(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[AsyncSession, None]:
    """Dependency providing an asynchronous database session."""
    yield session


async def get_agent_runner(request: Request):
    """Dependency providing the cached compiled LangGraph support agent.

    The graph is compiled once in the app lifespan (``app.state.agent``);
    falls back to compiling on demand (e.g. tests building routers directly).
    """
    cached = getattr(request.app.state, "agent", None)
    if cached is not None:
        return cached

    from agent.graph import create_support_graph

    return create_support_graph()
