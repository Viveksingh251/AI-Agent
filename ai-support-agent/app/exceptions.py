"""Custom Application Exceptions and Exception Handlers."""

from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse


class SupportAgentException(Exception):
    """Base exception for all domain-specific errors in AI Support Agent."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentExecutionError(SupportAgentException):
    """Raised when the LangGraph agent graph fails during step execution."""
    pass


class TicketNotFoundError(SupportAgentException):
    """Raised when a support ticket cannot be located."""
    pass


class CustomerNotFoundError(SupportAgentException):
    """Raised when customer record does not exist."""
    pass


class RAGRetrievalError(SupportAgentException):
    """Raised when vector database search or knowledge retrieval fails."""
    pass


async def support_agent_exception_handler(request: Request, exc: SupportAgentException) -> JSONResponse:
    """Global exception handler converting domain exceptions into standardized JSON responses."""
    # TODO: Log exception trace with structured logger and Sentry integration
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_type": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "path": request.url.path,
        },
    )

