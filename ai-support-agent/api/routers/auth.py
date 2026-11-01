"""Authentication and Token Management API Router with RBAC."""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    CustomerRegisterRequest,
    CustomerResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserLoginRequest,
    UserProfileResponse,
)
from app.config import Settings, get_settings
from app.dependencies import get_current_active_user, get_database
from app.security import (
    AuthenticatedUser,
    create_access_token,
    create_refresh_token,
    decode_jwt,
    UserRole,
)
from database import crud

logger = logging.getLogger("support_agent.api.auth")
router = APIRouter()


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User Login (Returns Access & Refresh Tokens)",
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_database),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Authenticates both Employees (support/admin) and Customers using OAuth2 Form Data.
    
    Generates a short-lived Access Token and a long-lived Refresh Token.
    """
    email = form_data.username.strip()
    password = form_data.password

    logger.info(f"Authenticating login request for: {email}")
    auth_result = await crud.authenticate_user(db, email=email, password=password)

    if not auth_result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_obj, user_type, role_str = auth_result

    token_claims = {
        "sub": user_obj.email,
        "user_id": user_obj.id,
        "name": user_obj.name,
        "role": role_str,
        "user_type": user_type,
    }

    access_token = create_access_token(token_claims)
    refresh_token = create_refresh_token(token_claims)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        role=role_str,
        user_type=user_type,
    )


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2 Token Endpoint Alias",
    include_in_schema=False,
)
async def token_alias(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_database),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Backwards-compatibility alias for Swagger OAuth2 login."""
    return await login(form_data=form_data, db=db, settings=settings)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh Access Token",
)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_database),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Validates refresh token and issues a fresh Access and Refresh Token pair."""
    try:
        claims = decode_jwt(payload.refresh_token, secret_key=settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        if claims.get("type") != "refresh":
            raise ValueError("Provided token is not a refresh token.")

        user_id = claims.get("user_id")
        user_type = claims.get("user_type")
        email = claims.get("sub")

        if not user_id or not email or not user_type:
            raise ValueError("Missing subject identity in token payload.")

    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired refresh token: {str(err)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Re-validate that user account still exists and is active
    if user_type == "employee":
        user_obj = await crud.get_employee_by_id(db, user_id)
        if not user_obj or not user_obj.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive or removed.")
        role_str = "admin" if user_obj.role.value == "admin" else "support_agent"
    else:
        user_obj = await crud.get_customer_by_id(db, user_id)
        if not user_obj or not user_obj.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Customer account is inactive or removed.")
        role_str = "customer"

    new_claims = {
        "sub": user_obj.email,
        "user_id": user_obj.id,
        "name": user_obj.name,
        "role": role_str,
        "user_type": user_type,
    }

    new_access_token = create_access_token(new_claims)
    new_refresh_token = create_refresh_token(new_claims)

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        role=role_str,
        user_type=user_type,
    )


@router.post(
    "/register",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Customer Self-Registration",
)
async def register_customer(
    payload: CustomerRegisterRequest,
    db: AsyncSession = Depends(get_database),
) -> CustomerResponse:
    """Public customer self-registration endpoint."""
    existing = await crud.get_customer_by_email(db, payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An account with email '{payload.email}' already exists.",
        )

    customer = await crud.create_customer(
        session=db,
        email=payload.email,
        name=payload.name,
        password=payload.password,
        phone=payload.phone,
        company=payload.company,
        tier="free",
    )
    return CustomerResponse.model_validate(customer)


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get Current Authenticated User Profile",
)
async def get_current_user_profile(
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> UserProfileResponse:
    """Returns the caller's active user credentials, role, and permission grants."""
    return UserProfileResponse(
        user_id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        role=current_user.role.value,
        user_type=current_user.user_type,
        is_active=current_user.is_active,
        permissions=current_user.permissions,
    )
