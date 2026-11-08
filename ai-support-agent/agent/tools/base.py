"""Shared infrastructure for agent tools: DB access, authorization, audit logging.

Every tool in ``agent/tools`` uses the helpers here so that all tool
invocations share one consistent pattern:

1. **Authorization** — customers may only touch their own records; staff
   (support agents / admins) may act on any customer's records.
2. **Audit logging** — each invocation writes to the ``agent_action_logs``
   table via ``database.crud.log_agent_action`` *and* emits a structured
   log line.  Logging failures never break the tool: audit is best-effort.
3. **Structured results** — every tool returns a dict with at least
   ``success: bool``; errors carry ``success=False`` plus a human-readable
   ``error`` message (never raw exceptions leaked to the LLM).
4. **Injection-proof roles** — ``normalize_role`` whitelists the role
   string *before* any authorization decision, so a hostile or corrupted
   value can never be interpreted as a staff role.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger("support_agent.tools")

from app.input_sanitizer import redact_mapping  # noqa: E402

# Roles that may access any customer's data.
STAFF_ROLES = {"support_agent", "admin", "staff"}

# The ONLY roles any caller may claim. Anything else (including values
# arriving from an LLM, a tampered payload, or a bug) is coerced to the
# least-privileged default.
VALID_ROLES = {"customer", "support_agent", "admin", "staff"}
DEFAULT_ROLE = "customer"


def normalize_role(role: Any) -> str:
    """Whitelist-normalise a role string.

    Deny-by-default: unknown / missing / hostile values become 'customer'
    so an invalid role can never grant staff clearance.
    """
    if not isinstance(role, str):
        return DEFAULT_ROLE
    value = role.strip().lower()
    return value if value in VALID_ROLES else DEFAULT_ROLE


# ---------------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------------


@asynccontextmanager
async def db_session() -> AsyncIterator[Any]:
    """Yield an async SQLAlchemy session with auto commit/rollback.

    Imports are done lazily so tools remain importable (and unit-testable
    with mocks) even when the app/database packages or DB drivers are not
    fully configured.
    """
    from database.session import async_session_maker  # noqa: PLC0415

    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


def is_authorized(
    requesting_customer_id: Optional[str],
    resource_customer_id: Optional[str],
    role: str = "customer",
) -> bool:
    """Return True if *requesting_customer_id* may access a resource owned
    by *resource_customer_id*.

    Rules:
    - staff roles (support_agent / admin) → always authorized
    - customers → only when the resource belongs to them
    - anonymous (no id) → never authorized for customer-scoped resources

    The role is normalised through :func:`normalize_role` first — an
    unrecognised role value is treated as 'customer', never as staff.
    """
    role = normalize_role(role)
    if role in STAFF_ROLES:
        return True
    if not requesting_customer_id:
        return False
    return str(requesting_customer_id) == str(resource_customer_id)


def deny(action: str, reason: str = "Permission Denied") -> Dict[str, Any]:
    """Build the canonical 'authorization denied' structured result."""
    logger.warning("AUTH DENIED | action=%s reason=%s", action, reason)
    return {
        "success": False,
        "error": f"{reason}: you are not authorized to perform '{action}'.",
        "authorized": False,
    }


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


async def audit_log(
    action: str,
    details: Dict[str, Any],
    customer_id: Optional[str] = None,
    success: bool = True,
) -> None:
    """Best-effort audit trail: DB row + structured application log.

    Never raises — a broken audit sink must not fail the customer-facing
    tool call.
    """
    logger.info(
        "TOOL ACTION | action=%s customer=%s success=%s details=%s",
        action,
        customer_id,
        success,
        redact_mapping(details),
    )
    try:
        from database.crud import log_agent_action  # noqa: PLC0415

        parsed_id: Optional[int] = None
        if customer_id is not None:
            try:
                parsed_id = int(customer_id)
            except (TypeError, ValueError):
                parsed_id = None

        async with db_session() as session:
            await log_agent_action(
                session,
                action_type=action,
                details=details,
                customer_id=parsed_id,
                success=success,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit_log failed for %s (non-fatal): %s", action, exc)


# ---------------------------------------------------------------------------
# Error wrapper
# ---------------------------------------------------------------------------


def tool_error(action: str, exc: Exception) -> Dict[str, Any]:
    """Convert an unexpected exception into a safe structured error result."""
    logger.exception("TOOL ERROR | action=%s error=%s", action, exc)
    return {
        "success": False,
        "error": f"'{action}' failed. Please try again or contact support.",
    }
