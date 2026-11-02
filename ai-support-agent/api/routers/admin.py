"""Admin & Observability API Router — restricted to support_agent / admin roles.

Endpoints:
    GET /api/v1/admin/performance-summary — aggregate operational metrics
    GET /api/v1/admin/agent-logs          — paginated agent action audit trail
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    AgentActionLogResponse,
    PaginatedLogsResponse,
    PerformanceReportResponse,
)
from app.dependencies import (
    AuthenticatedUser,
    get_database,
    require_roles,
    UserRole,
)
from database.models import AgentActionLog, SupportTicket

logger = logging.getLogger("support_agent.api.admin")
router = APIRouter()


@router.get(
    "/performance-summary",
    response_model=PerformanceReportResponse,
    summary="Operational Performance Summary",
)
async def performance_summary(
    window_hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
    include_report: bool = Query(False, description="Also generate an LLM narrative report"),
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(require_roles(UserRole.SUPPORT_AGENT, UserRole.ADMIN)),
) -> PerformanceReportResponse:
    """Aggregate agent/support metrics over the requested time window."""
    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    logger.info("Admin performance summary | window=%dh by=%s", window_hours, current_user.email)

    # --- Agent action log aggregates -------------------------------------
    result = await db.execute(
        select(AgentActionLog).where(AgentActionLog.created_at >= since)
    )
    logs = list(result.scalars().all())

    by_type = Counter(log.action_type for log in logs)
    succeeded = sum(1 for log in logs if log.success)
    failed = len(logs) - succeeded

    # --- Ticket aggregates ------------------------------------------------
    total_result = await db.execute(select(func.count(SupportTicket.id)))
    open_result = await db.execute(
        select(func.count(SupportTicket.id)).where(
            SupportTicket.status.in_(["open", "in_progress", "escalated"])
        )
    )
    created_result = await db.execute(
        select(func.count(SupportTicket.id)).where(SupportTicket.created_at >= since)
    )
    escalated_result = await db.execute(
        select(func.count(SupportTicket.id)).where(
            SupportTicket.status == "escalated", SupportTicket.created_at >= since
        )
    )

    summary = PerformanceReportResponse(
        window_hours=window_hours,
        total_actions=len(logs),
        successful_actions=succeeded,
        failed_actions=failed,
        success_rate=round(succeeded / len(logs), 4) if logs else 0.0,
        actions_by_type=dict(by_type),
        escalations=int(escalated_result.scalar() or 0),
        open_tickets=int(open_result.scalar() or 0),
        tickets_created=int(created_result.scalar() or 0),
        narrative_report=None,
    )

    if include_report:
        summary.narrative_report = await _narrative_report(summary)

    return summary


async def _narrative_report(summary: PerformanceReportResponse) -> Optional[str]:
    """LLM-generated qualitative summary; falls back to a deterministic one."""
    try:
        from agent.nodes import _OPENAI_AVAILABLE, _get_llm  # noqa: PLC0415

        if not _OPENAI_AVAILABLE:
            return None
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        prompt = (
            "Write a concise (max 150 words) operational performance report for a "
            "customer-support AI platform from these metrics:\n"
            f"- Window: last {summary.window_hours}h\n"
            f"- Agent actions: {summary.total_actions} ({summary.success_rate:.0%} success)\n"
            f"- Actions by type: {summary.actions_by_type}\n"
            f"- Tickets created: {summary.tickets_created}, open: {summary.open_tickets}, "
            f"escalated: {summary.escalations}\n"
            "Highlight anomalies and one concrete recommendation."
        )
        response = await _get_llm(temperature=0.2).ainvoke([HumanMessage(content=prompt)])
        text = getattr(response, "content", "")
        return text.strip() or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Narrative report generation failed: %s", exc)
        return None


@router.get(
    "/agent-logs",
    response_model=PaginatedLogsResponse,
    summary="Agent Action Audit Logs",
)
async def list_agent_logs(
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    customer_id: Optional[int] = Query(None, description="Filter by customer"),
    success: Optional[bool] = Query(None, description="Filter by success flag"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(require_roles(UserRole.SUPPORT_AGENT, UserRole.ADMIN)),
) -> PaginatedLogsResponse:
    """Paginated audit trail of every action the AI agent performed."""
    logger.info("Admin agent-logs listing | by=%s action_type=%s", current_user.email, action_type)

    stmt = select(AgentActionLog).order_by(AgentActionLog.created_at.desc())
    count_stmt = select(func.count(AgentActionLog.id))

    if action_type:
        stmt = stmt.where(AgentActionLog.action_type == action_type)
        count_stmt = count_stmt.where(AgentActionLog.action_type == action_type)
    if customer_id is not None:
        stmt = stmt.where(AgentActionLog.customer_id == customer_id)
        count_stmt = count_stmt.where(AgentActionLog.customer_id == customer_id)
    if success is not None:
        stmt = stmt.where(AgentActionLog.success.is_(success))
        count_stmt = count_stmt.where(AgentActionLog.success.is_(success))

    total_result = await db.execute(count_stmt)
    total = int(total_result.scalar() or 0)

    result = await db.execute(stmt.offset(skip).limit(limit))
    logs = list(result.scalars().all())

    return PaginatedLogsResponse(
        total=total,
        skip=skip,
        limit=limit,
        logs=[AgentActionLogResponse.model_validate(log) for log in logs],
    )
