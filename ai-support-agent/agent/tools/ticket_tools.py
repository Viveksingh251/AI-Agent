"""Support-ticket tools — real implementations backed by the database.

Covers: create, status-update, escalate, history, and follow-up email
drafting.  All customer-scoped operations enforce ownership checks; the
email drafting tool uses the LLM when available with a deterministic
template fallback.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from agent.tools.base import (
    STAFF_ROLES,
    audit_log,
    db_session,
    deny,
    is_authorized,
    tool_error,
)

logger = logging.getLogger("support_agent.tools.ticket")

_VALID_PRIORITIES = {"low", "medium", "high", "critical"}
_VALID_STATUSES = {"open", "in_progress", "resolved", "escalated", "closed"}


def _ticket_dict(ticket: Any) -> Dict[str, Any]:
    return {
        "id": str(ticket.id),
        "ticket_number": ticket.ticket_number,
        "customer_id": str(ticket.customer_id),
        "subject": ticket.subject,
        "status": ticket.status.value if hasattr(ticket.status, "value") else str(ticket.status),
        "priority": ticket.priority.value if hasattr(ticket.priority, "value") else str(ticket.priority),
        "created_at": str(ticket.created_at),
        "updated_at": str(ticket.updated_at) if getattr(ticket, "updated_at", None) else None,
    }


# ---------------------------------------------------------------------------
# Load a ticket (by numeric ID or ticket number)
# ---------------------------------------------------------------------------


async def _load_ticket(session: Any, ticket_id: str) -> Optional[Any]:
    from database.crud import get_ticket_by_id, get_ticket_by_number  # noqa: PLC0415

    key = str(ticket_id).strip()
    if key.isdigit():
        ticket = await get_ticket_by_id(session, int(key))
        if ticket is not None:
            return ticket
    return await get_ticket_by_number(session, key)


# ---------------------------------------------------------------------------
# 1. create_support_ticket
# ---------------------------------------------------------------------------


@tool
async def create_support_ticket(
    subject: str,
    description: str,
    priority: str = "medium",
    customer_id: Optional[str] = None,
    requesting_customer_id: Optional[str] = None,
    role: str = "customer",
) -> Dict[str, Any]:
    """Create a new support ticket for a customer.

    Args:
        subject: Short summary of the issue (max ~120 chars).
        description: Detailed description of the problem.
        priority: One of 'low', 'medium', 'high', 'critical'.
        customer_id: ID of the customer the ticket is for (defaults to the caller).
        requesting_customer_id: ID of the authenticated caller (system-injected).
        role: Caller role: 'customer', 'support_agent', or 'admin'.

    Returns:
        {"success": bool, "ticket": {...}} or {"success": False, "error": str}
    """
    action = "create_support_ticket"
    try:
        if not subject or not subject.strip():
            return {"success": False, "error": "A ticket subject is required."}
        if not description or not description.strip():
            return {"success": False, "error": "A ticket description is required."}
        priority = (priority or "medium").lower()
        if priority not in _VALID_PRIORITIES:
            return {
                "success": False,
                "error": f"Invalid priority '{priority}'. Valid values: {sorted(_VALID_PRIORITIES)}.",
            }

        # Ownership: customers can only create tickets for themselves.
        target_customer_id = str(customer_id) if customer_id else requesting_customer_id
        if not target_customer_id:
            return {"success": False, "error": "A customer ID is required to create a ticket."}
        if not is_authorized(requesting_customer_id, target_customer_id, role):
            return deny(action, "Customers can only create tickets for their own account")

        ticket_number = f"TCK-{uuid.uuid4().hex[:8].upper()}"

        async with db_session() as session:
            from database.crud import create_support_ticket  # noqa: PLC0415

            ticket = await create_support_ticket(
                session,
                ticket_number=ticket_number,
                customer_id=int(target_customer_id),
                subject=subject.strip()[:200],
                description=description.strip(),
                priority=priority,
                created_by_agent=True,
            )
            data = _ticket_dict(ticket)

        await audit_log(action, {"ticket_number": data["ticket_number"], "priority": priority}, target_customer_id)
        logger.info("Ticket created: %s (customer=%s)", data["ticket_number"], target_customer_id)
        return {"success": True, "ticket": data}
    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)


# ---------------------------------------------------------------------------
# 2. update_ticket_status
# ---------------------------------------------------------------------------


@tool
async def update_ticket_status(
    ticket_id: str,
    status: str,
    note: Optional[str] = None,
    requesting_customer_id: Optional[str] = None,
    role: str = "customer",
) -> Dict[str, Any]:
    """Update the status of an existing support ticket and optionally attach a note.

    Args:
        ticket_id: Ticket ID or ticket number.
        status: One of 'open', 'in_progress', 'resolved', 'escalated', 'closed'.
        note: Optional comment explaining the change (appended to the ticket thread).
        requesting_customer_id: ID of the authenticated caller (system-injected).
        role: Caller role: 'customer', 'support_agent', or 'admin'.

    Returns:
        {"success": bool, "ticket": {...}} or {"success": False, "error": str}
    """
    action = "update_ticket_status"
    try:
        status = (status or "").lower()
        if status not in _VALID_STATUSES:
            return {
                "success": False,
                "error": f"Invalid status '{status}'. Valid values: {sorted(_VALID_STATUSES)}.",
            }

        async with db_session() as session:
            ticket = await _load_ticket(session, ticket_id)
            if ticket is None:
                await audit_log(action, {"ticket_id": ticket_id, "found": False}, requesting_customer_id, success=False)
                return {"success": False, "error": f"No ticket found with identifier '{ticket_id}'."}

            owner_id = str(ticket.customer_id)
            if not is_authorized(requesting_customer_id, owner_id, role):
                await audit_log(action, {"ticket_id": ticket_id}, requesting_customer_id, success=False)
                return deny(action, "Customers may only update their own tickets")

            from database.crud import add_ticket_message, update_ticket  # noqa: PLC0415

            updated = await update_ticket(session, ticket_id=int(ticket.id), status=status)
            if updated is None:
                return {"success": False, "error": "Ticket update could not be applied."}

            if note and note.strip():
                await add_ticket_message(
                    session,
                    ticket_id=int(ticket.id),
                    sender_type="agent",
                    content=note.strip(),
                )

            data = _ticket_dict(updated)

        await audit_log(action, {"ticket_id": data["id"], "new_status": status, "note_added": bool(note)}, owner_id)
        return {"success": True, "ticket": data}
    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)


# ---------------------------------------------------------------------------
# 3. escalate_ticket
# ---------------------------------------------------------------------------


@tool
async def escalate_ticket(
    ticket_id: str,
    reason: str,
    priority: str = "high",
    requesting_customer_id: Optional[str] = None,
    role: str = "customer",
) -> Dict[str, Any]:
    """Escalate a ticket to senior support staff with an elevated priority.

    Requires 'support_agent' or 'admin' clearance — customers request
    escalation conversationally instead of calling this tool directly.

    Args:
        ticket_id: Ticket ID or ticket number.
        reason: Why the ticket is being escalated.
        priority: New priority ('high' by default, or 'critical').
        requesting_customer_id: ID of the authenticated caller (system-injected).
        role: Caller role — must be 'support_agent' or 'admin'.

    Returns:
        {"success": bool, "ticket": {...}} or {"success": False, "error": str}
    """
    action = "escalate_ticket"
    try:
        if role not in STAFF_ROLES:
            return deny(action, "Only support_agent or admin staff can escalate tickets")
        if not reason or not reason.strip():
            return {"success": False, "error": "An escalation reason is required."}

        priority = (priority or "high").lower()
        if priority not in {"high", "critical"}:
            return {"success": False, "error": "Escalated tickets must have priority 'high' or 'critical'."}

        async with db_session() as session:
            ticket = await _load_ticket(session, ticket_id)
            if ticket is None:
                await audit_log(action, {"ticket_id": ticket_id, "found": False}, requesting_customer_id, success=False)
                return {"success": False, "error": f"No ticket found with identifier '{ticket_id}'."}

            owner_id = str(ticket.customer_id)

            from database.crud import add_ticket_message, update_ticket  # noqa: PLC0415

            updated = await update_ticket(
                session,
                ticket_id=int(ticket.id),
                status="escalated",
                priority=priority,
            )
            if updated is None:
                return {"success": False, "error": "Ticket escalation could not be applied."}

            await add_ticket_message(
                session,
                ticket_id=int(ticket.id),
                sender_type="agent",
                content=f"[ESCALATED] {reason.strip()}",
            )
            data = _ticket_dict(updated)

        await audit_log(action, {"ticket_id": data["id"], "reason": reason, "priority": priority}, owner_id)
        logger.warning("Ticket %s escalated: %s", data["ticket_number"], reason)
        return {"success": True, "ticket": data}
    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)


# ---------------------------------------------------------------------------
# 4. get_ticket_history
# ---------------------------------------------------------------------------


@tool
async def get_ticket_history(
    ticket_id: str,
    requesting_customer_id: Optional[str] = None,
    role: str = "customer",
) -> Dict[str, Any]:
    """Retrieve a ticket's full details and its complete message thread,
    oldest first.

    Authorization: customers may only view history for their own tickets.

    Args:
        ticket_id: Ticket ID or ticket number.
        requesting_customer_id: ID of the authenticated caller (system-injected).
        role: Caller role: 'customer', 'support_agent', or 'admin'.

    Returns:
        {"success": bool, "ticket": {...}, "messages": [...]}
    """
    action = "get_ticket_history"
    try:
        async with db_session() as session:
            ticket = await _load_ticket(session, ticket_id)
            if ticket is None:
                await audit_log(action, {"ticket_id": ticket_id, "found": False}, requesting_customer_id, success=False)
                return {"success": False, "error": f"No ticket found with identifier '{ticket_id}'."}

            owner_id = str(ticket.customer_id)
            if not is_authorized(requesting_customer_id, owner_id, role):
                await audit_log(action, {"ticket_id": ticket_id}, requesting_customer_id, success=False)
                return deny(action, "Customers may only view their own ticket history")

            from database.crud import get_ticket_messages  # noqa: PLC0415

            messages = await get_ticket_messages(session, ticket_id=int(ticket.id))
            data = _ticket_dict(ticket)
            data["description"] = ticket.description

        await audit_log(action, {"ticket_id": data["id"], "messages": len(messages)}, owner_id)
        return {
            "success": True,
            "ticket": data,
            "messages": [
                {
                    "sender": m.sender_type.value if hasattr(m.sender_type, "value") else str(m.sender_type),
                    "content": m.content,
                    "created_at": str(m.created_at),
                }
                for m in messages
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)


# ---------------------------------------------------------------------------
# 5. draft_followup_email
# ---------------------------------------------------------------------------

_EMAIL_TEMPLATE = """Subject: {subject}

Dear {name},

Thank you for contacting our support team regarding your ticket
({ticket_ref}).

{body}

If you have any further questions, simply reply to this message and we
will be happy to assist.

Kind regards,
Customer Support Team
"""


@tool
async def draft_followup_email(
    ticket_id: str,
    tone: str = "professional",
    requesting_customer_id: Optional[str] = None,
    role: str = "customer",
) -> Dict[str, Any]:
    """Draft a follow-up email to the customer about a support ticket.

    Uses the configured LLM when available; otherwise falls back to a
    deterministic template so the workflow never breaks.

    Args:
        ticket_id: Ticket ID or ticket number.
        tone: Desired tone — 'professional', 'friendly', or 'apologetic'.
        requesting_customer_id: ID of the authenticated caller (system-injected).
        role: Caller role — drafting internal correspondence requires staff clearance.

    Returns:
        {"success": bool, "email": str, "tone": str, "used_llm": bool}
    """
    action = "draft_followup_email"
    try:
        if role not in STAFF_ROLES:
            return deny(action, "Drafting follow-up emails requires staff clearance")

        tone = (tone or "professional").lower()
        if tone not in {"professional", "friendly", "apologetic"}:
            return {"success": False, "error": "Tone must be 'professional', 'friendly', or 'apologetic'."}

        async with db_session() as session:
            ticket = await _load_ticket(session, ticket_id)
            if ticket is None:
                await audit_log(action, {"ticket_id": ticket_id, "found": False}, requesting_customer_id, success=False)
                return {"success": False, "error": f"No ticket found with identifier '{ticket_id}'."}

            owner_id = str(ticket.customer_id)
            customer_name = "Valued Customer"
            customer = None
            try:
                from database.crud import get_customer_by_id  # noqa: PLC0415

                customer = await get_customer_by_id(session, int(owner_id))
                if customer:
                    customer_name = customer.name
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load customer for email draft: %s", exc)

        ticket_ref = ticket.ticket_number
        subject = ticket.subject
        status = ticket.status.value if hasattr(ticket.status, "value") else str(ticket.status)
        description = ticket.description

        email: Optional[str] = None
        used_llm = False
        try:
            email = await _llm_draft(
                tone=tone,
                name=customer_name,
                ticket_ref=ticket_ref,
                subject=subject,
                status=status,
                description=description,
            )
            used_llm = email is not None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM email draft failed, using template: %s", exc)

        if not email:
            tone_openers = {
                "professional": "We are writing to update you on the status of your request.",
                "friendly": "Just a quick note to keep you in the loop on your request!",
                "apologetic": "We sincerely apologise for the inconvenience this issue has caused you.",
            }
            email = _EMAIL_TEMPLATE.format(
                subject=f"Update on your support ticket {ticket_ref}",
                name=customer_name,
                ticket_ref=ticket_ref,
                body=(
                    f"{tone_openers[tone]}\n\n"
                    f"Issue: {description}\n\n"
                    f"Current status: {status.replace('_', ' ').title()}. "
                    f"Our team is actively working on it and we will update you as soon as "
                    f"there is progress."
                ),
            )

        await audit_log(action, {"ticket_id": ticket_ref, "tone": tone, "used_llm": used_llm}, owner_id)
        return {"success": True, "email": email.strip(), "tone": tone, "used_llm": used_llm}
    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)


async def _llm_draft(**ctx: Any) -> Optional[str]:
    """Attempt an LLM-generated email; return None when unavailable or stubbed."""
    try:
        from langchain_core.messages import HumanMessage  # noqa: PLC0415
    except ImportError:
        return None

    try:
        from agent.nodes import _OPENAI_AVAILABLE, _get_llm  # noqa: PLC0415

        # The stub LLM (no langchain-openai installed) returns canned triage
        # JSON — detecting that and falling back to the template is safer
        # than shipping garbage to the customer.
        if not _OPENAI_AVAILABLE:
            return None

        llm = _get_llm(temperature=0.4)
        prompt = (
            f"Draft a {ctx['tone']} follow-up email to a customer about their support ticket.\n"
            f"Customer name: {ctx['name']}\n"
            f"Ticket reference: {ctx['ticket_ref']}\n"
            f"Ticket subject: {ctx['subject']}\n"
            f"Issue description: {ctx['description']}\n"
            f"Current ticket status: {ctx['status']}\n"
            "Return ONLY the email body including a subject line. Sign off as "
            "'Customer Support Team'. Do not invent facts or promises."
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = getattr(response, "content", "")
        return text.strip() if text and text.strip() else None
    except Exception:  # noqa: BLE001
        return None
