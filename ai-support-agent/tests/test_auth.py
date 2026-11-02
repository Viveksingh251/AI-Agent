"""Authentication & JWT unit tests: hashing, tokens, RBAC permissions."""

from __future__ import annotations

import time

import pytest

from app.config import get_settings
from app.security import (
    AuthenticatedUser,
    ROLE_PERMISSIONS,
    UserRole,
    check_tool_access,
    create_access_token,
    create_refresh_token,
    decode_jwt,
    encode_jwt,
    get_password_hash,
    verify_password,
)

settings = get_settings()


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_and_verify_roundtrip(self):
        hashed = get_password_hash("S3curePass!")
        assert hashed != "S3curePass!"
        assert verify_password("S3curePass!", hashed) is True

    def test_wrong_password_rejected(self):
        hashed = get_password_hash("RightPassword")
        assert verify_password("WrongPassword", hashed) is False

    def test_hash_is_salted_unique(self):
        assert get_password_hash("same") != get_password_hash("same")

    def test_empty_inputs_rejected(self):
        assert verify_password("", "somehash") is False
        assert verify_password("pw", "") is False


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


class TestJWT:
    def _claims(self) -> dict:
        return {"sub": "user@example.com", "user_id": 42,
                "role": "support_agent", "user_type": "employee"}

    def test_access_token_roundtrip(self):
        token = create_access_token(self._claims())
        decoded = decode_jwt(token, secret_key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        assert decoded["sub"] == "user@example.com"
        assert decoded["user_id"] == 42
        assert decoded["type"] == "access"
        assert "exp" in decoded

    def test_refresh_token_has_refresh_type(self):
        token = create_refresh_token(self._claims())
        decoded = decode_jwt(token, secret_key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        assert decoded["type"] == "refresh"

    def test_expired_token_rejected(self):
        token = encode_jwt({"sub": "x", "exp": int(time.time()) - 1000},
                           secret_key=settings.SECRET_KEY)
        with pytest.raises(ValueError):
            decode_jwt(token, secret_key=settings.SECRET_KEY)

    def test_tampered_signature_rejected(self):
        token = create_access_token(self._claims())
        bad = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
        with pytest.raises(ValueError):
            decode_jwt(bad, secret_key=settings.SECRET_KEY)

    def test_wrong_secret_rejected(self):
        token = create_access_token(self._claims())
        with pytest.raises(ValueError):
            decode_jwt(token, secret_key="other-secret")

    def test_malformed_token_rejected(self):
        with pytest.raises(ValueError):
            decode_jwt("not-a-token", secret_key=settings.SECRET_KEY)


# ---------------------------------------------------------------------------
# Role-based permissions
# ---------------------------------------------------------------------------


def _user(role: UserRole) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=1, email=f"{role.value}@x.com", name=role.value, role=role,
        user_type="customer" if role == UserRole.CUSTOMER else "employee",
        permissions=list(ROLE_PERMISSIONS.get(role, set())),
    )


class TestRBAC:
    def test_customer_permissions(self):
        u = _user(UserRole.CUSTOMER)
        assert u.has_permission("ticket:create")
        assert not u.has_permission("ticket:escalate")
        assert not u.has_permission("customer:read")

    def test_support_agent_permissions(self):
        u = _user(UserRole.SUPPORT_AGENT)
        assert u.has_permission("ticket:escalate")
        assert u.has_permission("customer:read")
        assert not u.has_permission("*")

    def test_admin_wildcard(self):
        u = AuthenticatedUser(id=3, email="a@x.com", name="admin", role=UserRole.ADMIN,
                              user_type="employee", permissions=["*"])
        assert u.has_permission("literally.anything")

    def test_has_any_role(self):
        u = _user(UserRole.SUPPORT_AGENT)
        assert u.has_any_role(UserRole.SUPPORT_AGENT)
        assert not u.has_any_role(UserRole.CUSTOMER)
        # admins implicitly pass
        assert _user(UserRole.ADMIN).has_any_role(UserRole.CUSTOMER)

    def test_tool_access_matrix(self):
        assert check_tool_access(_user(UserRole.ADMIN), "escalate_ticket_tool")
        assert check_tool_access(_user(UserRole.SUPPORT_AGENT), "escalate_ticket_tool")
        assert not check_tool_access(_user(UserRole.CUSTOMER), "escalate_ticket_tool")
        assert check_tool_access(_user(UserRole.CUSTOMER), "get_order_history_tool")
        # unknown tool defaults to allow (restricted explicitly elsewhere)
        assert check_tool_access(_user(UserRole.CUSTOMER), "unknown_tool")
