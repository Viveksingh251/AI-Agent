"""Unit and Integration Tests for JWT Authentication and Role-Based Access Control."""

import pytest
from httpx import AsyncClient

from app.security import (
    AuthenticatedUser,
    ROLE_PERMISSIONS,
    UserRole,
    check_tool_access,
    create_access_token,
    create_refresh_token,
    decode_jwt,
    get_password_hash,
    verify_password,
)
from app.config import get_settings
from agent.tools.ticket_tools import escalate_ticket_tool
from agent.tools.customer_tools import view_sensitive_financials_tool

settings = get_settings()


def test_password_hashing():
    """Verifies password hashing and verification logic."""
    plain = "SuperSecretPassword123!"
    hashed = get_password_hash(plain)

    assert hashed != plain
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_access_token_generation_and_decoding():
    """Verifies JWT access token encoding and payload claims."""
    claims = {
        "sub": "user@example.com",
        "user_id": 42,
        "role": "support_agent",
        "user_type": "employee",
    }
    token = create_access_token(claims)
    assert isinstance(token, str)

    decoded = decode_jwt(token, secret_key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    assert decoded["sub"] == "user@example.com"
    assert decoded["user_id"] == 42
    assert decoded["role"] == "support_agent"
    assert decoded["type"] == "access"
    assert "exp" in decoded


def test_refresh_token_generation_and_decoding():
    """Verifies refresh token carries 'refresh' type tag."""
    claims = {
        "sub": "customer@example.com",
        "user_id": 99,
        "role": "customer",
        "user_type": "customer",
    }
    token = create_refresh_token(claims)
    decoded = decode_jwt(token, secret_key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    assert decoded["type"] == "refresh"
    assert decoded["sub"] == "customer@example.com"


def test_role_permissions_mapping():
    """Verifies role permissions grants."""
    customer_user = AuthenticatedUser(
        id=1,
        email="c@example.com",
        name="Customer",
        role=UserRole.CUSTOMER,
        user_type="customer",
        permissions=list(ROLE_PERMISSIONS[UserRole.CUSTOMER]),
    )
    assert customer_user.has_permission("ticket:create") is True
    assert customer_user.has_permission("ticket:escalate") is False

    agent_user = AuthenticatedUser(
        id=2,
        email="a@example.com",
        name="Agent",
        role=UserRole.SUPPORT_AGENT,
        user_type="employee",
        permissions=list(ROLE_PERMISSIONS[UserRole.SUPPORT_AGENT]),
    )
    assert agent_user.has_permission("ticket:escalate") is True
    assert agent_user.has_permission("tools:view_sensitive") is True

    admin_user = AuthenticatedUser(
        id=3,
        email="adm@example.com",
        name="Admin",
        role=UserRole.ADMIN,
        user_type="employee",
        permissions=["*"],
    )
    assert admin_user.has_permission("any_arbitrary_permission") is True


def test_tool_permission_enforcement():
    """Verifies tool level RBAC blocks unauthorized invocations."""
    # 1. Escalation tool
    denied_res = escalate_ticket_tool.invoke({
        "ticket_id": "TICK-101",
        "escalation_reason": "Urgent issue",
        "caller_role": "customer",
    })
    assert denied_res.get("authorized") is False
    assert "Permission Denied" in denied_res.get("error", "")

    allowed_res = escalate_ticket_tool.invoke({
        "ticket_id": "TICK-101",
        "escalation_reason": "Customer needs VIP support",
        "caller_role": "support_agent",
    })
    assert allowed_res.get("status") == "escalated"

    # 2. Sensitive financials tool
    fin_denied = view_sensitive_financials_tool.invoke({
        "customer_id": "1",
        "caller_role": "customer",
    })
    assert fin_denied.get("authorized") is False

    fin_allowed = view_sensitive_financials_tool.invoke({
        "customer_id": "1",
        "caller_role": "admin",
    })
    assert fin_allowed.get("authorized") is True
    assert "credit_balance_usd" in fin_allowed

