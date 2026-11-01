"""Authentication, JWT Token Handling, and Role-Based Access Control (RBAC)."""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from database.session import get_db_session

logger = logging.getLogger("support_agent.security")
settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/login",
    auto_error=True,
)


# ==============================================================================
# 1. Role & Permission Definitions
# ==============================================================================

class UserRole(str, Enum):
    """User access control roles."""

    CUSTOMER = "customer"
    SUPPORT_AGENT = "support_agent"
    ADMIN = "admin"


# Role-to-Permissions Mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.CUSTOMER: {
        "chat:interact",
        "ticket:create",
        "ticket:read_own",
        "order:read_own",
    },
    UserRole.SUPPORT_AGENT: {
        "chat:interact",
        "ticket:create",
        "ticket:read_all",
        "ticket:update",
        "ticket:escalate",
        "order:read_all",
        "customer:read",
        "tools:view_sensitive",
        "tools:escalate",
    },
    UserRole.ADMIN: {
        "*",  # Superuser wildcard grant
    },
}

# Tool-Level Access Authorization
TOOL_PERMISSIONS: Dict[str, Set[UserRole]] = {
    "create_support_ticket_tool": {UserRole.CUSTOMER, UserRole.SUPPORT_AGENT, UserRole.ADMIN},
    "get_ticket_status_tool": {UserRole.CUSTOMER, UserRole.SUPPORT_AGENT, UserRole.ADMIN},
    "search_knowledge_base_tool": {UserRole.CUSTOMER, UserRole.SUPPORT_AGENT, UserRole.ADMIN},
    "get_customer_profile_tool": {UserRole.SUPPORT_AGENT, UserRole.ADMIN},
    "get_order_history_tool": {UserRole.CUSTOMER, UserRole.SUPPORT_AGENT, UserRole.ADMIN},
    "escalate_ticket_tool": {UserRole.SUPPORT_AGENT, UserRole.ADMIN},
    "view_sensitive_financials_tool": {UserRole.SUPPORT_AGENT, UserRole.ADMIN},
}


class AuthenticatedUser(BaseModel):
    """Data transfer model representing the currently authenticated user in request context."""

    id: int
    email: str
    name: str
    role: UserRole
    user_type: str  # 'customer' | 'employee'
    is_active: bool = True
    permissions: List[str] = Field(default_factory=list)

    def has_permission(self, required_permission: str) -> bool:
        """Checks if user possesses the requested permission."""
        if "*" in self.permissions or self.role == UserRole.ADMIN:
            return True
        return required_permission in self.permissions

    def has_any_role(self, *roles: UserRole) -> bool:
        """Checks if user holds any of the specified roles."""
        return self.role in roles or self.role == UserRole.ADMIN


# ==============================================================================
# 2. Cryptographic Password Hashing (Bcrypt with PBKDF2 Fallback)
# ==============================================================================

# Attempt to load passlib for industry standard bcrypt; fallback to hashlib.pbkdf2_hmac
_passlib_context = None
try:
    from passlib.context import CryptContext
    _passlib_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except ImportError:
    _passlib_context = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a stored hash."""
    if not hashed_password or not plain_password:
        return False

    if _passlib_context is not None:
        try:
            return _passlib_context.verify(plain_password, hashed_password)
        except Exception:
            pass

    # Standard library PBKDF2 fallback verification
    if hashed_password.startswith("pbkdf2:"):
        try:
            parts = hashed_password.split("$")
            if len(parts) == 4:
                _, iterations, salt, stored_hash = parts
                test_hash = hashlib.pbkdf2_hmac(
                    "sha256",
                    plain_password.encode("utf-8"),
                    salt.encode("utf-8"),
                    int(iterations),
                ).hex()
                return hmac.compare_digest(test_hash, stored_hash)
        except Exception:
            return False

    return False


def get_password_hash(password: str) -> str:
    """Generates a secure cryptographic password hash."""
    if _passlib_context is not None:
        try:
            return _passlib_context.hash(password)
        except Exception:
            pass

    # Secure standard library PBKDF2-HMAC-SHA256 hash
    salt = secrets.token_hex(16)
    iterations = 100_000
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return f"pbkdf2:sha256${iterations}${salt}${derived_key}"


# ==============================================================================
# 3. JWT Token Generation and Decoding
# ==============================================================================

def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data_str: str) -> bytes:
    padding = "=" * ((4 - len(data_str) % 4) % 4)
    return base64.urlsafe_b64decode(data_str + padding)


def encode_jwt(payload: Dict[str, Any], secret_key: str, algorithm: str = "HS256") -> str:
    """Encodes a JWT claims dictionary into a signed JWT string."""
    header = {"typ": "JWT", "alg": algorithm}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")

    unsigned_token = f"{_base64url_encode(header_bytes)}.{_base64url_encode(payload_bytes)}"
    signature = hmac.new(
        secret_key.encode("utf-8"),
        unsigned_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return f"{unsigned_token}.{_base64url_encode(signature)}"


def decode_jwt(token: str, secret_key: str, algorithm: str = "HS256") -> Dict[str, Any]:
    """Decodes and validates a signed JWT token string."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT token format.")

        unsigned_token = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac.new(
            secret_key.encode("utf-8"),
            unsigned_token.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        provided_sig = _base64url_decode(parts[2])
        if not hmac.compare_digest(expected_sig, provided_sig):
            raise ValueError("JWT signature verification failed.")

        payload_bytes = _base64url_decode(parts[1])
        payload = json.loads(payload_bytes.decode("utf-8"))

        # Verify expiration
        exp = payload.get("exp")
        if exp is not None and time.time() > float(exp):
            raise ValueError("Token has expired.")

        return payload
    except Exception as exc:
        raise ValueError(f"Could not validate credentials: {str(exc)}") from exc


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generates a short-lived access JWT token."""
    to_encode = data.copy()
    expire_duration = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire_timestamp = datetime.now(timezone.utc) + expire_duration

    to_encode.update({
        "exp": int(expire_timestamp.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "type": "access",
    })
    return encode_jwt(to_encode, secret_key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generates a long-lived refresh JWT token."""
    to_encode = data.copy()
    expire_duration = expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    expire_timestamp = datetime.now(timezone.utc) + expire_duration

    to_encode.update({
        "exp": int(expire_timestamp.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "type": "refresh",
    })
    return encode_jwt(to_encode, secret_key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)


# ==============================================================================
# 4. FastAPI Security Dependencies
# ==============================================================================

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> AuthenticatedUser:
    """Validates JWT bearer token and resolves the active user from the database."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_jwt(token, secret_key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        email: Optional[str] = payload.get("sub")
        user_id: Optional[int] = payload.get("user_id")
        user_type: Optional[str] = payload.get("user_type", "customer")
        token_type: Optional[str] = payload.get("type")

        if not email or not user_id or token_type != "access":
            raise credentials_exception

    except ValueError:
        raise credentials_exception

    # Resolve entity from respective database table
    from database import crud

    if user_type == "employee":
        employee = await crud.get_employee_by_id(db, user_id)
        if not employee or not employee.is_active:
            raise credentials_exception

        role = UserRole.ADMIN if employee.role.value == "admin" else UserRole.SUPPORT_AGENT
        permissions = list(ROLE_PERMISSIONS.get(role, set()))

        return AuthenticatedUser(
            id=employee.id,
            email=employee.email,
            name=employee.name,
            role=role,
            user_type="employee",
            is_active=employee.is_active,
            permissions=permissions,
        )

    else:
        customer = await crud.get_customer_by_id(db, user_id)
        if not customer or not customer.is_active:
            raise credentials_exception

        role = UserRole.CUSTOMER
        permissions = list(ROLE_PERMISSIONS.get(role, set()))

        return AuthenticatedUser(
            id=customer.id,
            email=customer.email,
            name=customer.name,
            role=role,
            user_type="customer",
            is_active=customer.is_active,
            permissions=permissions,
        )


async def get_current_active_user(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """Ensures that the authenticated user account is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive account.",
        )
    return current_user


def require_roles(*allowed_roles: UserRole) -> Callable:
    """Dependency factory enforcing that the caller possesses at least one of the allowed roles."""
    async def role_checker(
        current_user: AuthenticatedUser = Depends(get_current_active_user),
    ) -> AuthenticatedUser:
        if current_user.role == UserRole.ADMIN:
            return current_user
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {[r.value for r in allowed_roles]}",
            )
        return current_user

    return role_checker


def require_permission(required_permission: str) -> Callable:
    """Dependency factory enforcing that the caller possesses a specific permission string."""
    async def permission_checker(
        current_user: AuthenticatedUser = Depends(get_current_active_user),
    ) -> AuthenticatedUser:
        if not current_user.has_permission(required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: '{required_permission}'",
            )
        return current_user

    return permission_checker


# ==============================================================================
# 5. Agent Tool Authorization Guard
# ==============================================================================

def check_tool_access(user: AuthenticatedUser, tool_name: str) -> bool:
    """Verifies whether an authenticated user possesses clearance to invoke an agent tool."""
    if user.role == UserRole.ADMIN:
        return True

    allowed_roles = TOOL_PERMISSIONS.get(tool_name)
    if allowed_roles is None:
        # Default allow if not restricted
        return True

    return user.role in allowed_roles


def verify_tool_access(user: AuthenticatedUser, tool_name: str) -> None:
    """Raises HTTP 403 Forbidden if user lacks clearance for the specified agent tool."""
    if not check_tool_access(user, tool_name):
        logger.warning(
            f"Access denied for user {user.email} (role={user.role.value}) to tool '{tool_name}'"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Role '{user.role.value}' is not permitted to execute tool '{tool_name}'.",
        )

