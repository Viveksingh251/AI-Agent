"""Customer Profiles and Account History API Router with RBAC."""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import CustomerCreate, CustomerResponse, OrderResponse, TicketResponse
from app.dependencies import (
    AuthenticatedUser,
    get_current_active_user,
    get_database,
    require_roles,
    UserRole,
)
from database import crud

logger = logging.getLogger("support_agent.api.customers")
router = APIRouter()


@router.post(
    "",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Customer Account (Admin Only)",
)
async def create_customer_account(
    payload: CustomerCreate,
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(require_roles(UserRole.ADMIN)),
) -> CustomerResponse:
    """Creates a customer profile via administrative interface. Restricted to Admins."""
    existing = await crud.get_customer_by_email(db, payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Customer with email '{payload.email}' already exists.",
        )

    customer = await crud.create_customer(
        session=db,
        email=payload.email,
        name=payload.name,
        password=payload.password,
        phone=payload.phone,
        company=payload.company,
        tier=payload.tier,
    )
    return CustomerResponse.model_validate(customer)


@router.get(
    "/{customer_id}",
    response_model=CustomerResponse,
    summary="Get Customer Profile",
)
async def get_customer_profile(
    customer_id: int,
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> CustomerResponse:
    """Retrieves customer record by ID.
    
    Customers can only view their own record; Support staff and Admins can view any profile.
    """
    if current_user.role == UserRole.CUSTOMER and current_user.id != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot view other customer profiles.",
        )

    customer = await crud.get_customer_by_id(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with id={customer_id} was not found.",
        )
    return CustomerResponse.model_validate(customer)


@router.get(
    "/{customer_id}/tickets",
    response_model=List[TicketResponse],
    summary="List Customer Tickets",
)
async def list_customer_tickets(
    customer_id: int,
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> List[TicketResponse]:
    """Returns support tickets for the specified customer."""
    if current_user.role == UserRole.CUSTOMER and current_user.id != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot view other customers' tickets.",
        )

    tickets = await crud.list_customer_tickets(db, customer_id=customer_id)
    return [TicketResponse.model_validate(t) for t in tickets]


@router.get(
    "/{customer_id}/orders",
    response_model=List[OrderResponse],
    summary="List Customer Orders",
)
async def list_customer_orders(
    customer_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> List[OrderResponse]:
    """Returns recent orders for the specified customer.

    Customers can only view their own orders; Support staff and Admins can
    view orders for any customer.
    """
    if current_user.role == UserRole.CUSTOMER and current_user.id != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot view other customers' orders.",
        )

    orders = await crud.list_customer_orders(db, customer_id=customer_id, limit=limit)
    return [OrderResponse.model_validate(o) for o in orders]
