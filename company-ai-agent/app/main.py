"""Main FastAPI Application Entry Point for Company AI Agent."""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import CompanyAIAgent, get_agent
from app.config import Settings, get_settings
from app.models import get_db, init_db_and_seed
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthCheckResponse,
    MessageItem,
    SessionHistoryResponse,
    SessionSummary,
    ToolDefinition,
)
from app.services.memory_service import MemoryService, get_memory_service

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("company_agent.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for application startup and shutdown events."""
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} in '{settings.APP_ENV}' environment...")
    
    # Initialize and seed database with default enterprise data
    await init_db_and_seed()
    logger.info("Database initialized and verified.")
    
    yield
    
    logger.info(f"Shutting down {settings.APP_NAME}...")


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise AI Agent API with dynamic tool calling, database reasoning, and persistent memory.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root Status Endpoint", tags=["General"])
async def root() -> Dict[str, Any]:
    """Provides basic service metadata and API navigation info."""
    return {
        "service": settings.APP_NAME,
        "status": "online",
        "environment": settings.APP_ENV,
        "docs_url": "/docs",
        "api_v1_prefix": "/api/v1",
    }


@app.get("/health", response_model=HealthCheckResponse, summary="Service Health Check", tags=["General"])
async def health_check(
    agent: CompanyAIAgent = Depends(get_agent),
) -> HealthCheckResponse:
    """Performs deep health check across Agent, Database, and LLM subsystem."""
    return await agent.health_check()


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Agent Turn",
    tags=["Agent"],
)
async def chat_with_agent(
    request: ChatRequest,
    agent: CompanyAIAgent = Depends(get_agent),
) -> ChatResponse:
    """Processes a user prompt through the Company AI Agent.
    
    The agent dynamically reasons, executes database or API tools if needed,
    updates persistent session memory, and yields a synthesized final answer.
    """
    try:
        response = await agent.run(
            message=request.message,
            session_id=request.session_id,
            system_override=request.system_override,
        )
        return response
    except Exception as exc:
        logger.exception("Error occurred while processing agent turn")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(exc)}",
        ) from exc


@app.get(
    "/api/v1/sessions",
    response_model=List[SessionSummary],
    summary="List Conversation Sessions",
    tags=["Memory"],
)
async def list_sessions(
    limit: int = 50,
    memory_service: MemoryService = Depends(get_memory_service),
) -> List[SessionSummary]:
    """Returns a list of conversation sessions stored in memory."""
    return await memory_service.list_sessions(limit=limit)


@app.get(
    "/api/v1/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get Conversation History",
    tags=["Memory"],
)
async def get_session_history(
    session_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
) -> SessionHistoryResponse:
    """Retrieves full turn history for a given conversation session."""
    messages = await memory_service.get_history(session_id=session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        total_messages=len(messages),
    )


@app.delete(
    "/api/v1/sessions/{session_id}",
    summary="Clear Conversation History",
    tags=["Memory"],
)
async def clear_session(
    session_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
) -> Dict[str, str]:
    """Clears all stored conversation messages for the specified session."""
    deleted_count = await memory_service.clear_session(session_id=session_id)
    return {
        "session_id": session_id,
        "status": "cleared",
        "deleted_messages": str(deleted_count),
    }


@app.get(
    "/api/v1/tools",
    response_model=List[ToolDefinition],
    summary="List Registered Tools",
    tags=["Tools"],
)
async def list_tools(
    agent: CompanyAIAgent = Depends(get_agent),
) -> List[ToolDefinition]:
    """Exposes all tools currently available to the AI agent."""
    return agent.get_available_tools()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

