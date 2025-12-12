"""Authentication routes for Google OAuth integration."""
from http.client import HTTPException
from app.services.user_service import user_service
from sqlalchemy.orm import Session
from fastapi import APIRouter, Query , Depends
from app.db.db import get_db
from app.core.dependencies import CurrentUser
from app.core.security import verify_password,create_access_token
from app.models.auth import (
    CurrentUserResponse,
    LoginResponse,
    LogoutResponse,
    TokenResponse, UserCreate,LoginRequest
)
from app.settings import settings
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login",
    description="Check user credentials",
)
def login(request: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    """
    Validate user credentials using SQLAlchemy session.
    """

    user = user_service.get_user_by_username(db, request.user_name)
    if not user or not verify_password(request.password, user.password):
        raise HTTPException(
        )

    access_token = create_access_token(data={"sub": user.id},expires_minutes=settings.jwt_access_token_expire_minutes)

    return LoginResponse(access_token=access_token)

@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout",
    description="Log out the current user and invalidate their session.",
)
async def logout(current_user: CurrentUser) -> LogoutResponse:
    """Log out the current user."""
    raise NotImplementedError()


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="Get Current User",
    description="Get information about the currently authenticated user.",
)
async def get_current_user(current_user: CurrentUser) -> CurrentUserResponse:
    """Get the current authenticated user's information."""
    raise NotImplementedError()


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh Token",
    description="Refresh an expired access token using a refresh token.",
)
async def refresh_token(
    refresh_token: str = Query(..., description="Refresh token"),
) -> TokenResponse:
    """Refresh an expired access token."""
    raise NotImplementedError()
