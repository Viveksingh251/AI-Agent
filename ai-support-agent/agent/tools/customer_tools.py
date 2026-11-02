"""Customer record and order lookup tools, backed by the real database.

Authorization model:
- ``requesting_customer_id`` is the authenticated caller (injected from the
  graph state, never taken from the LLM).
- Customers may only view their own profile / orders.
- Staff roles (support_agent, admin) may view any record.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from agent.tools.base import (
    audit_log,
    db_session,
    deny,
    is_authorized,
    tool_error,
)

logger = logging.getLogger("support_agent.tools.customer")


@tool
async def get_customer_info(
    email_or_id: str,
    requesting_customer_id: Optional[str] = None,
    role: str = "customer",
) -> Dict[str, Any]:
    """Look up a customer account by email address or numeric customer ID.

    Authorization: customers can only retrieve their own profile.
    Support agents and admins may look up any customer.

    Args:
        email_or_id: Email address or numeric ID of the customer to fetch.
        requesting_customer_id: ID of the authenticated caller (system-injected).
        role: Caller role: 'customer', 'support_agent', or 'admin'.

    Returns:
        {"success": bool, "customer": {...}} or {"success": False, "error": str}
    """
    action = "get_customer_info"
    try:
        if not email_or_id or not str(email_or_id).strip():
            return {"success": False, "error": "An email or customer ID is required."}

        key = str(email_or_id).strip()

        async with db_session() as session:
            from database.crud import get_customer_by_email, get_customer_by_id  # noqa: PLC0415

            customer = None
            if key.isdigit():
                customer = await get_customer_by_id(session, int(key))
            if customer is None and "@" in key:
                customer = await get_customer_by_email(session, key)
            if customer is None and not key.isdigit() and "@" not in key:
                return {
                    "success": False,
                    "error": "Invalid identifier — provide a numeric ID or an email address.",
                }

            if customer is None:
                await audit_log(action, {"lookup": key, "found": False}, requesting_customer_id, success=False)
                return {"success": False, "error": "No customer found with that email or ID."}

            # Authorization: customer may only see their own record.
            if not is_authorized(requesting_customer_id, str(customer.id), role):
                await audit_log(
                    action,
                    {"lookup": key, "target_customer_id": str(customer.id)},
                    requesting_customer_id,
                    success=False,
                )
                return deny(action, "Customers may only access their own account information")

            data = {
                "id": str(customer.id),
                "email": customer.email,
                "name": customer.name,
                "phone": customer.phone,
                "company": customer.company,
                "tier": customer.tier.value if hasattr(customer.tier, "value") else str(customer.tier),
                "is_active": bool(customer.is_active),
                "created_at": str(customer.created_at),
            }

        await audit_log(action, {"lookup": key, "found": True}, requesting_customer_id)
        return {"success": True, "customer": data}
    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)


@tool
async def get_order_status(
    order_number: str,
    customer_id: str,
    requesting_customer_id: Optional[str] = None,
    role: str = "customer",
) -> Dict[str, Any]:
    """Fetch the full details and fulfilment status of an order by its order number.

    Authorization: customers can only view orders that belong to them.

    Args:
        order_number: The public order number (e.g. 'ORD-8821').
        customer_id: ID of the customer who owns the order.
        requesting_customer_id: ID of the authenticated caller (system-injected).
        role: Caller role: 'customer', 'support_agent', or 'admin'.

    Returns:
        {"success": bool, "order": {...}} or {"success": False, "error": str}
    """
    action = "get_order_status"
    try:
        if not order_number or not str(order_number).strip():
            return {"success": False, "error": "An order number is required."}

        async with db_session() as session:
            from database.crud import get_order_by_number  # noqa: PLC0415

            order = await get_order_by_number(session, str(order_number).strip())
            if order is None:
                await audit_log(action, {"order_number": order_number, "found": False}, requesting_customer_id, success=False)
                return {"success": False, "error": f"No order found with number '{order_number}'."}

            owner_id = str(order.customer_id)
            # The stated customer_id must match the order owner, and the
            # caller must be authorized to see that customer's data.
            if customer_id and str(customer_id) != owner_id:
                await audit_log(action, {"order_number": order_number, "claimed_owner": str(customer_id)}, requesting_customer_id, success=False)
                return deny(action, "This order does not belong to the provided customer")

            if not is_authorized(requesting_customer_id, owner_id, role):
                await audit_log(action, {"order_number": order_number, "owner": owner_id}, requesting_customer_id, success=False)
                return deny(action, "Customers may only view their own orders")

            data = {
                "id": str(order.id),
                "order_number": order.order_number,
                "customer_id": owner_id,
                "status": order.status.value if hasattr(order.status, "value") else str(order.status),
                "total_amount": float(order.total_amount),
                "items": order.items,
                "created_at": str(order.created_at),
            }

        await audit_log(action, {"order_number": order_number}, requesting_customer_id)
        return {"success": True, "order": data}
    except Exception as exc:  # noqa: BLE001
        return tool_error(action, exc)
