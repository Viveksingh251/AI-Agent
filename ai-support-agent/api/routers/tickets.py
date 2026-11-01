"""Support Ticket Lifecycle API Router with Role-Based Access Control."""

import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import TicketCreate, TicketResponse, TicketUpdate
from app.dependencies import (
    AuthenticatedUser,
    get_current_active_user,
    get_database,
    require_roles,
    UserRole,
)
from database import crud

logger = logging.getLogger("support_agent.api.tickets")
router = APIRouter()


@router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Support Ticket",
)
async def create_ticket(
    payload: TicketCreate,
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> TicketResponse:
    """Creates a new customer support ticket.
    
    Customers may only open tickets for their own account; Support Agents and Admins can create on behalf of any customer.
    """
    if current_user.role == UserRole.CUSTOMER and payload.customer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Customers can only create support tickets for their own account.",
        )

    ticket_number = f"TICK-{uuid.uuid4().hex[:6].upper()}"
    logger.info(f"Creating ticket {ticket_number} by user={current_user.email} for customer_id={payload.customer_id}")

    ticket = await crud.create_support_ticket(
        session=db,
        ticket_number=ticket_number,
        customer_id=payload.customer_id,
        subject=payload.subject,
        description=payload.description,
        priority=payload.priority,
        assigned_to=payload.assigned_to,
        created_by_agent=(current_user.role != UserRole.CUSTOMER),
    )
    return TicketResponse.model_validate(ticket)


@router.get(
    "/{ticket_number}",
    response_model=TicketResponse,
    summary="Get Ticket by Number",
)
async def get_ticket(
    ticket_number: str,
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> TicketResponse:
    """Retrieves ticket details by ticket number.
    
    Customers are restricted to their own tickets; staff can inspect all tickets.
    """
    logger.info(f"User {current_user.email} fetching ticket {ticket_number}")
    ticket = await crud.get_ticket_by_number(db, ticket_number)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket '{ticket_number}' was not found.",
        )

    # Ownership access check
    if current_user.role == UserRole.CUSTOMER and ticket.customer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not possess clearance to view this ticket.",
        )

    return TicketResponse.model_validate(ticket)


@router.patch(
    "/{ticket_id}",
    response_model=TicketResponse,
    summary="Update Ticket Status or Assignee (Staff Only)",
)
async def update_ticket(
    ticket_id: int,
    payload: TicketUpdate,
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(require_roles(UserRole.SUPPORT_AGENT, UserRole.ADMIN)),
) -> TicketResponse:
    """Updates ticket status, priority, or assigned agent.
    
    Restricted to support_agent and admin roles.
    """
    logger.info(f"Staff {current_user.email} updating ticket_id={ticket_id}")
    ticket = await crud.update_ticket(
        session=db,
        ticket_id=ticket_id,
        status=payload.status,
        priority=payload.priority,
        assigned_to=payload.assigned_to,
    )
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket id={ticket_id} was not found.",
        )
    return TicketResponse.model_validate(ticket)


@router.get(
    "",
    response_model=List[TicketResponse],
    summary="List Support Tickets",
)
async def list_tickets(
    status_filter: Optional[str] = Query(None, alias="status"),
    priority_filter: Optional[str] = Query(None, alias="priority"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_database),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> List[TicketResponse]:
    """Lists support tickets. Customers see only their own; staff see all matching tickets."""
    if current_user.role == UserRole.CUSTOMER:
        tickets = await crud.list_customer_tickets(
            db, customer_id=current_user.id, status=status_filter, limit=limit
        )
    else:
        tickets = await crud.list_tickets(
            db, status=status_filter, priority=priority_filter, skip=skip, limit=limit
        )
    return [TicketResponse.model_validate(t) for t in tickets]
