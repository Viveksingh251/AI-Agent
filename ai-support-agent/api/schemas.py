"""Pydantic v2 Request, Response, and Data Transfer Schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field


# ==============================================================================
# Chat & Agent Schemas
# ==============================================================================

class ChatRequest(BaseModel):
    """Payload sent by client to interact with the customer support agent."""

    message: str = Field(..., min_length=1, description="Customer text prompt or query")
    session_id: Optional[str] = Field(default=None, description="Conversation session ID for thread continuity")
    customer_id: Optional[str] = Field(default=None, description="Customer external identifier")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class ChatResponse(BaseModel):
    """Structured response synthesized by the LangGraph support agent."""

    session_id: str = Field(..., description="Active session ID")
    response: str = Field(..., description="Assistant response text")
    intent: Optional[str] = Field(default=None, description="Classified intent")
    sentiment: Optional[str] = Field(default=None, description="Detected sentiment")
    escalation_required: bool = Field(default=False, description="Whether human escalation was triggered")
    ticket_id: Optional[str] = Field(default=None, description="Associated ticket ID if generated")
    confidence_score: float = Field(default=1.0, description="Agent confidence score (0.0 to 1.0)")
    turn_count: int = Field(default=1, description="Turn sequence counter")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC response timestamp")


# ==============================================================================
# Customer Schemas
# ==============================================================================

class CustomerBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    email: EmailStr
    phone: Optional[str] = Field(default=None, max_length=32)
    company: Optional[str] = Field(default=None, max_length=128)
    tier: str = Field(default="free", description="Subscription tier: free | pro | enterprise")


class CustomerCreate(CustomerBase):
    password: Optional[str] = Field(default=None, min_length=6, description="Customer account password")


class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    tier: Optional[str] = None


class CustomerResponse(CustomerBase):
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# Order Schemas
# ==============================================================================

class OrderCreate(BaseModel):
    customer_id: int
    order_number: str
    total_amount: Decimal
    items: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "pending"


class OrderResponse(BaseModel):
    id: int
    customer_id: int
    order_number: str
    status: str
    total_amount: Decimal
    items: List[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# Support Ticket Schemas
# ==============================================================================

class TicketCreate(BaseModel):
    customer_id: int
    subject: str = Field(..., min_length=3, max_length=255)
    description: str = Field(..., min_length=5)
    priority: str = Field(default="medium", description="Urgency: low | medium | high | critical")
    assigned_to: Optional[int] = Field(default=None, description="Employee ID")
    created_by_agent: bool = Field(default=False)


class TicketUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    assigned_to: Optional[int] = None


class SupportTicketResponse(BaseModel):
    id: int
    ticket_number: str
    customer_id: int
    subject: str
    description: str
    status: str
    priority: str
    assigned_to: Optional[int] = None
    created_by_agent: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Backwards compatibility alias
TicketResponse = SupportTicketResponse


# ==============================================================================
# Ticket Message Schemas
# ==============================================================================

class TicketMessageCreate(BaseModel):
    ticket_id: int
    sender_type: str = Field(..., description="customer | agent | human")
    content: str = Field(..., min_length=1)


class TicketMessageResponse(BaseModel):
    id: int
    ticket_id: int
    sender_type: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# Employee Schemas
# ==============================================================================

class EmployeeCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str
    role: str = Field(default="support", description="support | admin")
    is_active: bool = True


class EmployeeResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# Agent Action Audit Log Schemas
# ==============================================================================

class AgentActionLogResponse(BaseModel):
    id: int
    customer_id: Optional[int] = None
    action_type: str
    details: Dict[str, Any]
    success: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ==============================================================================
# Authentication & RBAC Schemas
# ==============================================================================

class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User account email address")
    password: str = Field(..., min_length=1, description="Account password")


class CustomerRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    email: EmailStr
    password: str = Field(..., min_length=6, description="Plaintext password")
    phone: Optional[str] = None
    company: Optional[str] = None


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Short-lived JWT access token")
    refresh_token: str = Field(..., description="Long-lived JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiry duration in seconds")
    role: str = Field(..., description="Assigned user role (customer, support_agent, admin)")
    user_type: str = Field(..., description="Account category (customer | employee)")


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Valid JWT refresh token")


class UserProfileResponse(BaseModel):
    user_id: int
    email: str
    name: str
    role: str
    user_type: str
    is_active: bool = True
    permissions: List[str] = Field(default_factory=list)


# ==============================================================================
# System & Diagnostic Health Schemas
# ==============================================================================

class ComponentHealth(BaseModel):
    """Health status details of an individual subsystem component."""

    status: str = Field(..., description="Component status: 'healthy', 'degraded', or 'disabled'")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic metadata")


class HealthCheckResponse(BaseModel):
    """Comprehensive service health and diagnostic status."""

    status: str = Field(..., description="Overall service status: 'healthy' or 'degraded'")
    app_name: str = Field(..., description="Service application name")
    environment: str = Field(..., description="Current deployment environment")
    version: str = Field(..., description="Service build version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC diagnostic timestamp")
    components: Dict[str, ComponentHealth] = Field(default_factory=dict, description="Status of subsystems")
    feature_flags: Dict[str, bool] = Field(default_factory=dict, description="Active feature flag values")
