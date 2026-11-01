"""Support Ticket Management Tools for LangChain / LangGraph Agent with Role Clearance."""

from typing import Any, Dict, Optional
from langchain_core.tools import tool


@tool
def create_support_ticket_tool(
    customer_id: str,
    subject: str,
    description: str,
    priority: str = "medium",
) -> Dict[str, Any]:
    """Creates a new customer support ticket in the internal ticketing system.
    
    Permitted for all authenticated roles (customer, support_agent, admin).
    """
    return {
        "status": "created",
        "ticket_id": "TICK-90210",
        "customer_id": customer_id,
        "subject": subject,
        "priority": priority,
    }


@tool
def get_ticket_status_tool(ticket_id: str) -> Dict[str, Any]:
    """Retrieves current resolution status, assignee, and history for an existing ticket.
    
    Permitted for all authenticated roles.
    """
    return {
        "ticket_id": ticket_id,
        "status": "in_progress",
        "assigned_team": "Tier 1 Technical Support",
        "last_updated": "2026-11-01T12:00:00Z",
    }


@tool
def escalate_ticket_tool(
    ticket_id: str,
    escalation_reason: str,
    target_tier: str = "tier_2_support",
    caller_role: str = "customer",
) -> Dict[str, Any]:
    """Escalates an incident ticket to senior engineering or management.
    
    Requires 'support_agent' or 'admin' role clearance. Regular customers cannot self-escalate tickets without review.
    """
    if caller_role not in ["support_agent", "admin"]:
        return {
            "error": "Permission Denied: Only support_agent or admin staff can directly execute ticket escalations.",
            "authorized": False,
        }

    return {
        "ticket_id": ticket_id,
        "status": "escalated",
        "assigned_tier": target_tier,
        "escalation_reason": escalation_reason,
        "authorized_by_role": caller_role,
        "timestamp": "2026-11-01T12:30:00Z",
    }
